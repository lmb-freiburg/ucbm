"""Non-negative matrix factorization."""

# Author: Vlad Niculae
#         Lars Buitinck
#         Mathieu Blondel <mathieu@mblondel.org>
#         Tom Dupre la Tour
# License: BSD 3 clause

import itertools
import time
import warnings
from abc import ABC
from math import sqrt
from numbers import Integral, Real
import torch
from functools import partial
from tqdm import trange
import os

import numpy as np
import scipy.sparse as sp
from scipy import linalg, sparse

from sklearn._config import config_context
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array, check_random_state, gen_batches, metadata_routing
from sklearn.utils._param_validation import (
    Hidden,
    Interval,
    StrOptions,
    validate_params,
)
from sklearn.utils.deprecation import _deprecate_Xt_in_inverse_transform
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from sklearn.utils.validation import (
    check_is_fitted,
    check_non_negative,
)
from sklearn.decomposition._cdnmf_fast import _update_cdnmf_fast
from sklearn.utils._array_api import device, get_namespace

EPSILON = np.finfo(np.float32).eps
EPSILON_TORCH = torch.finfo(torch.float32).eps

def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    If u_based_decision is False, then the same sign correction is applied to
    so that the rows in v that are largest in absolute value are always
    positive.

    Parameters
    ----------
    u : ndarray
        Parameters u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
        u can be None if `u_based_decision` is False.

    v : ndarray
        Parameters u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`. The input v should
        really be called vt to be consistent with scipy's output.
        v can be None if `u_based_decision` is True.

    u_based_decision : bool, default=True
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.

    Returns
    -------
    u_adjusted : ndarray
        Array u with adjusted columns and the same dimensions as u.

    v_adjusted : ndarray
        Array v with adjusted rows and the same dimensions as v.
    """
    xp, _ = get_namespace(*[a for a in [u, v] if a is not None])

    if u_based_decision:
        # columns of u, rows of v, or equivalently rows of u.T and v
        max_abs_u_cols = xp.argmax(xp.abs(u.T), axis=1)
        shift = xp.arange(u.T.shape[0], device=device(u))
        indices = max_abs_u_cols + shift * u.T.shape[1]
        signs = xp.sign(xp.take(xp.reshape(u.T, (-1,)), indices, axis=0))
        u *= signs[np.newaxis, :]
        if v is not None:
            v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_v_rows = xp.argmax(xp.abs(v), axis=1)
        shift = xp.arange(v.shape[0], device=device(v))
        indices = max_abs_v_rows + shift * v.shape[1]
        signs = xp.sign(xp.take(xp.reshape(v, (-1,)), indices, axis=0))
        if u is not None:
            u *= signs[np.newaxis, :]
        v *= signs[:, np.newaxis]
    return u, v

def blockwise_lu_decomposition_with_pivoting(A, device, block_size=int(1e5)):
    n = A.shape[0]
    L = torch.zeros_like(A).to(device)
    U = torch.zeros_like(A).to(device)
    P = torch.eye(n, dtype=A.dtype, device=device)  # Permutation matrix

    for i in range(0, n, block_size):
        end_i = min(i + block_size, n)

        # Extract the current block
        A_block = torch.from_numpy(A[i:end_i, i:end_i]).to(A.device)

        # Perform LU decomposition with pivoting on the block
        P_block, L_block, U_block = linalg.lu(A_block, pivot=True)
        
        # Update the permutation matrix
        P[i:end_i, i:end_i] = P_block.cpu()

        # Apply the permutation to the entire matrix A
        A = torch.matmul(P.T, A)

        # Set the values in L and U
        L[i:end_i, i:end_i] = L_block
        U[i:end_i, i:end_i] = U_block

        if end_i == n:
            break

        # Extract the off-diagonal blocks
        A21 = A[end_i:n, i:end_i]
        A12 = A[i:end_i, end_i:n]

        # Compute the off-diagonal blocks in L and U
        L[end_i:n, i:end_i] = torch.triangular_solve(A21.T, U_block, upper=False)[0].T
        U[i:end_i, end_i:n] = torch.triangular_solve(A12, L_block, upper=True)[0]

        # Update the remaining block of A
        A[end_i:n, end_i:n] -= torch.matmul(L[end_i:n, i:end_i], U[i:end_i, end_i:n])

    return P, L, U

def randomized_range_finder(
    A, *, size, n_iter, power_iteration_normalizer="auto", random_state=None, device="cpu"
):
    """Compute an orthonormal matrix whose range approximates the range of A.

    Parameters
    ----------
    A : 2D array
        The input data matrix.

    size : int
        Size of the return array.

    n_iter : int
        Number of power iterations used to stabilize the result.

    power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.

        .. versionadded:: 0.18

    random_state : int, RandomState instance or None, default=None
        The seed of the pseudo random number generator to use when shuffling
        the data, i.e. getting the random vectors to initialize the algorithm.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    Q : ndarray
        A (size x size) projection matrix, the range of which
        approximates well the range of the input matrix A.

    Notes
    -----

    Follows Algorithm 4.3 of
    :arxiv:`"Finding structure with randomness:
    Stochastic algorithms for constructing approximate matrix decompositions"
    <0909.4061>`
    Halko, et al. (2009)

    An implementation of a randomized algorithm for principal component
    analysis
    A. Szlam et al. 2014

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils.extmath import randomized_range_finder
    >>> A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> randomized_range_finder(A, size=2, n_iter=2, random_state=42)
    array([[-0.21...,  0.88...],
           [-0.52...,  0.24...],
           [-0.82..., -0.38...]])
    """
    xp, is_array_api_compliant = get_namespace(A)
    random_state = check_random_state(random_state)

    # Generating normal random vectors with shape: (A.shape[1], size)
    # XXX: generate random number directly from xp if it's possible
    # one day.
    Q = xp.asarray(random_state.normal(size=(A.shape[1], size)))
    Q = torch.from_numpy(Q).to(device)
    if hasattr(A, "dtype") and xp.isdtype(A.dtype, kind="real floating"):
        # Use float32 computation and components if A has a float32 dtype.
        # Q = xp.astype(Q, A.dtype, copy=False)
        Q = Q.float()

    # Move Q to device if needed only after converting to float32 if needed to
    # avoid allocating unnecessary memory on the device.

    # Note: we cannot combine the astype and to_device operations in one go
    # using xp.asarray(..., dtype=dtype, device=device) because downcasting
    # from float64 to float32 in asarray might not always be accepted as only
    # casts following type promotion rules are guarateed to work.
    # https://github.com/data-apis/array-api/issues/647
    if is_array_api_compliant:
        raise NotImplementedError
        Q = xp.asarray(Q, device=device(A))

    # Deal with "auto" mode
    if power_iteration_normalizer == "auto":
        if n_iter <= 2:
            power_iteration_normalizer = "none"
        elif is_array_api_compliant:
            # XXX: https://github.com/data-apis/array-api/issues/627
            warnings.warn(
                "Array API does not support LU factorization, falling back to QR"
                " instead. Set `power_iteration_normalizer='QR'` explicitly to silence"
                " this warning."
            )
            power_iteration_normalizer = "QR"
        else:
            power_iteration_normalizer = "LU"
    elif power_iteration_normalizer == "LU" and is_array_api_compliant:
        raise ValueError(
            "Array API does not support LU factorization. Set "
            "`power_iteration_normalizer='QR'` instead."
        )

    if is_array_api_compliant:
        raise NotImplementedError
        qr_normalizer = partial(xp.linalg.qr, mode="reduced")
    else:
        # Use scipy.linalg instead of numpy.linalg when not explicitly
        # using the Array API.
        # qr_normalizer = partial(linalg.qr, mode="economic", check_finite=False)
        qr_normalizer = partial(torch.linalg.qr, mode="reduced")

    if power_iteration_normalizer == "QR":
        normalizer = qr_normalizer
    elif power_iteration_normalizer == "LU":
        # normalizer = partial(linalg.lu, permute_l=True, check_finite=False)
        normalizer = partial(torch.linalg.lu, pivot=True)
    else:
        normalizer = lambda x: (x, None)

    # Perform power iterations with Q to further 'imprint' the top
    # singular vectors of A in Q
    for _ in range(n_iter):
        # Q, _ = normalizer(A @ Q)
        # Q, _ = normalizer(A.T @ Q)
        # P, L, U = normalizer(torch.from_numpy(np.array(A)).to(device) @ Q)
        P, L, U = blockwise_lu_decomposition_with_pivoting(torch.from_numpy(np.array(A)).to(device) @ Q, device)
        Q = P @ L
        P, L, U = normalizer(A.T @ Q)
        # P, L, U = blockwise_lu_decomposition_with_pivoting(A.T @ Q, device)
        Q = P @ L
        

    # Sample the range of A using by linear projection of Q
    # Extract an orthonormal basis
    Q, _ = qr_normalizer(A @ Q)

    return Q

def randomized_svd_gpu(
    M,
    n_components,
    *,
    n_oversamples=10,
    n_iter="auto",
    power_iteration_normalizer="auto",
    transpose="auto",
    flip_sign=True,
    random_state=None,
    svd_lapack_driver="gesdd",
    device="cpu",
):
    """Compute a truncated randomized SVD.

    This method solves the fixed-rank approximation problem described in [1]_
    (problem (1.5), p5).

    Parameters
    ----------
    M : {ndarray, sparse matrix}
        Matrix to decompose.

    n_components : int
        Number of singular values and vectors to extract.

    n_oversamples : int, default=10
        Additional number of random vectors to sample the range of `M` so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of `M` is `n_components + n_oversamples`. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of singular vectors and singular values. Users might wish
        to increase this parameter up to `2*k - n_components` where k is the
        effective rank, for large matrices, noisy problems, matrices with
        slowly decaying spectrums, or to increase precision accuracy. See [1]_
        (pages 5, 23 and 26).

    n_iter : int or 'auto', default='auto'
        Number of power iterations. It can be used to deal with very noisy
        problems. When 'auto', it is set to 4, unless `n_components` is small
        (< .1 * min(X.shape)) in which case `n_iter` is set to 7.
        This improves precision with few components. Note that in general
        users should rather increase `n_oversamples` before increasing `n_iter`
        as the principle of the randomized method is to avoid usage of these
        more costly power iterations steps. When `n_components` is equal
        or greater to the effective matrix rank and the spectrum does not
        present a slow decay, `n_iter=0` or `1` should even work fine in theory
        (see [1]_ page 9).

        .. versionchanged:: 0.18

    power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.

        .. versionadded:: 0.18

    transpose : bool or 'auto', default='auto'
        Whether the algorithm should be applied to M.T instead of M. The
        result should approximately be the same. The 'auto' mode will
        trigger the transposition if M.shape[1] > M.shape[0] since this
        implementation of randomized SVD tend to be a little faster in that
        case.

        .. versionchanged:: 0.18

    flip_sign : bool, default=True
        The output of a singular value decomposition is only unique up to a
        permutation of the signs of the singular vectors. If `flip_sign` is
        set to `True`, the sign ambiguity is resolved by making the largest
        loadings for each component in the left singular vectors positive.

    random_state : int, RandomState instance or None, default='warn'
        The seed of the pseudo random number generator to use when
        shuffling the data, i.e. getting the random vectors to initialize
        the algorithm. Pass an int for reproducible results across multiple
        function calls. See :term:`Glossary <random_state>`.

        .. versionchanged:: 1.2
            The default value changed from 0 to None.

    svd_lapack_driver : {"gesdd", "gesvd"}, default="gesdd"
        Whether to use the more efficient divide-and-conquer approach
        (`"gesdd"`) or more general rectangular approach (`"gesvd"`) to compute
        the SVD of the matrix B, which is the projection of M into a low
        dimensional subspace, as described in [1]_.

        .. versionadded:: 1.2

    Returns
    -------
    u : ndarray of shape (n_samples, n_components)
        Unitary matrix having left singular vectors with signs flipped as columns.
    s : ndarray of shape (n_components,)
        The singular values, sorted in non-increasing order.
    vh : ndarray of shape (n_components, n_features)
        Unitary matrix having right singular vectors with signs flipped as rows.

    Notes
    -----
    This algorithm finds a (usually very good) approximate truncated
    singular value decomposition using randomization to speed up the
    computations. It is particularly fast on large matrices on which
    you wish to extract only a small number of components. In order to
    obtain further speed up, `n_iter` can be set <=2 (at the cost of
    loss of precision). To increase the precision it is recommended to
    increase `n_oversamples`, up to `2*k-n_components` where k is the
    effective rank. Usually, `n_components` is chosen to be greater than k
    so increasing `n_oversamples` up to `n_components` should be enough.

    References
    ----------
    .. [1] :arxiv:`"Finding structure with randomness:
      Stochastic algorithms for constructing approximate matrix decompositions"
      <0909.4061>`
      Halko, et al. (2009)

    .. [2] A randomized algorithm for the decomposition of matrices
      Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert

    .. [3] An implementation of a randomized algorithm for principal component
      analysis A. Szlam et al. 2014

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils.extmath import randomized_svd
    >>> a = np.array([[1, 2, 3, 5],
    ...               [3, 4, 5, 6],
    ...               [7, 8, 9, 10]])
    >>> U, s, Vh = randomized_svd(a, n_components=2, random_state=0)
    >>> U.shape, s.shape, Vh.shape
    ((3, 2), (2,), (2, 4))
    """
    if sparse.issparse(M) and M.format in ("lil", "dok"):
        warnings.warn(
            "Calculating SVD of a {} is expensive. "
            "csr_matrix is more efficient.".format(type(M).__name__),
            sparse.SparseEfficiencyWarning,
        )

    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples
    n_samples, n_features = M.shape

    if n_iter == "auto":
        # Checks if the number of iterations is explicitly specified
        # Adjust n_iter. 7 was found a good compromise for PCA. See #5299
        n_iter = 7 if n_components < 0.1 * min(M.shape) else 4

    if transpose == "auto":
        transpose = n_samples < n_features
    if transpose:
        # this implementation is a bit faster with smaller shape[1]
        raise NotImplementedError
        M = M.T

    Q = randomized_range_finder(
        M,
        size=n_random,
        n_iter=n_iter,
        power_iteration_normalizer=power_iteration_normalizer,
        random_state=random_state,
        device=device,
    )

    # project M to the (k + p) dimensional space using the basis vectors
    B = Q.T @ M

    # compute the SVD on the thin matrix: (k + p) wide
    xp, is_array_api_compliant = get_namespace(B)
    if is_array_api_compliant:
        Uhat, s, Vt = xp.linalg.svd(B, full_matrices=False)
    else:
        # When when array_api_dispatch is disabled, rely on scipy.linalg
        # instead of numpy.linalg to avoid introducing a behavior change w.r.t.
        # previous versions of scikit-learn.
        Uhat, s, Vt = linalg.svd(
            B, full_matrices=False, lapack_driver=svd_lapack_driver
        )
    del B
    U = Q @ Uhat

    if flip_sign:
        if not transpose:
            U, Vt = svd_flip(U, Vt)
        else:
            # In case of transpose u_based_decision=false
            # to actually flip based on u and not v.
            U, Vt = svd_flip(U, Vt, u_based_decision=False)

    if transpose:
        # transpose back the results according to the input convention
        return Vt[:n_components, :].T, s[:n_components], U[:, :n_components].T
    else:
        return U[:, :n_components], s[:n_components], Vt[:n_components, :]

def norm(x):
    """Dot product-based Euclidean norm implementation.

    See: http://fa.bianp.net/blog/2011/computing-the-vector-norm/

    Parameters
    ----------
    x : array-like
        Vector for which to compute the norm.
    """
    return sqrt(squared_norm(x))


def trace_dot(X, Y):
    """Trace of np.dot(X, Y.T).

    Parameters
    ----------
    X : array-like
        First matrix.
    Y : array-like
        Second matrix.
    """
    return np.dot(X.ravel(), Y.ravel())


def _check_init(A, shape, whom):
    A = check_array(A)
    if shape[0] != "auto" and A.shape[0] != shape[0]:
        raise ValueError(
            f"Array with wrong first dimension passed to {whom}. Expected {shape[0]}, "
            f"but got {A.shape[0]}."
        )
    if shape[1] != "auto" and A.shape[1] != shape[1]:
        raise ValueError(
            f"Array with wrong second dimension passed to {whom}. Expected {shape[1]}, "
            f"but got {A.shape[1]}."
        )
    check_non_negative(A, whom)
    if np.max(A) == 0:
        raise ValueError(f"Array passed to {whom} is full of zeros.")


def _beta_divergence(X, W, H, beta, square_root=False):
    """Compute the beta-divergence of X and dot(W, H).

    Parameters
    ----------
    X : float or array-like of shape (n_samples, n_features)

    W : float or array-like of shape (n_samples, n_components)

    H : float or array-like of shape (n_components, n_features)

    beta : float or {'frobenius', 'kullback-leibler', 'itakura-saito'}
        Parameter of the beta-divergence.
        If beta == 2, this is half the Frobenius *squared* norm.
        If beta == 1, this is the generalized Kullback-Leibler divergence.
        If beta == 0, this is the Itakura-Saito divergence.
        Else, this is the general beta-divergence.

    square_root : bool, default=False
        If True, return np.sqrt(2 * res)
        For beta == 2, it corresponds to the Frobenius norm.

    Returns
    -------
        res : float
            Beta divergence of X and np.dot(X, H).
    """
    beta = _beta_loss_to_float(beta)

    # The method can be called with scalars
    # if not sp.issparse(X):
    #     X = np.atleast_2d(X)
    # W = np.atleast_2d(W)
    # H = np.atleast_2d(H)

    # Frobenius norm
    if beta == 2:
        # Avoid the creation of the dense np.dot(W, H) if X is sparse.
        if False and X.is_sparse:
            norm_X = np.dot(X.data, X.data)
            norm_WH = trace_dot(np.linalg.multi_dot([W.T, W, H]), H)
            cross_prod = trace_dot((X @ H.T), W)
            res = (norm_X + norm_WH - 2.0 * cross_prod) / 2.0
        else:
            # res = squared_norm(X - np.dot(W, H)) / 2.0
            chunk_size = int(1e5)
            res = 0
            for i in range(0, X.shape[0], chunk_size):
                chunk = slice(i, i + chunk_size)
                diff = (X[chunk] - W[chunk] @ H).flatten()
                res += diff @ diff / 2.0
            # diff = (X - W @ H).flatten()
            # old_res = diff @ diff / 2.0

        if square_root:
            # return np.sqrt(res * 2)
            if isinstance(res, torch.Tensor):
                return torch.sqrt(res * 2)
            return np.sqrt(res * 2)
        else:
            return res

    raise NotImplementedError

    if sp.issparse(X):
        # compute np.dot(W, H) only where X is nonzero
        WH_data = _special_sparse_dot(W, H, X).data
        X_data = X.data
    else:
        WH = np.dot(W, H)
        WH_data = WH.ravel()
        X_data = X.ravel()

    # do not affect the zeros: here 0 ** (-1) = 0 and not infinity
    indices = X_data > EPSILON
    WH_data = WH_data[indices]
    X_data = X_data[indices]

    # used to avoid division by zero
    WH_data[WH_data < EPSILON] = EPSILON

    # generalized Kullback-Leibler divergence
    if beta == 1:
        # fast and memory efficient computation of np.sum(np.dot(W, H))
        sum_WH = np.dot(np.sum(W, axis=0), np.sum(H, axis=1))
        # computes np.sum(X * log(X / WH)) only where X is nonzero
        div = X_data / WH_data
        res = np.dot(X_data, np.log(div))
        # add full np.sum(np.dot(W, H)) - np.sum(X)
        res += sum_WH - X_data.sum()

    # Itakura-Saito divergence
    elif beta == 0:
        div = X_data / WH_data
        res = np.sum(div) - np.prod(X.shape) - np.sum(np.log(div))

    # beta-divergence, beta not in (0, 1, 2)
    else:
        if sp.issparse(X):
            # slow loop, but memory efficient computation of :
            # np.sum(np.dot(W, H) ** beta)
            sum_WH_beta = 0
            for i in range(X.shape[1]):
                sum_WH_beta += np.sum(np.dot(W, H[:, i]) ** beta)

        else:
            sum_WH_beta = np.sum(WH**beta)

        sum_X_WH = np.dot(X_data, WH_data ** (beta - 1))
        res = (X_data**beta).sum() - beta * sum_X_WH
        res += sum_WH_beta * (beta - 1)
        res /= beta * (beta - 1)

    if square_root:
        res = max(res, 0)  # avoid negative number due to rounding errors
        return np.sqrt(2 * res)
    else:
        return res


def _special_sparse_dot(W, H, X):
    """Computes np.dot(W, H), only where X is non zero."""
    if sp.issparse(X):
        ii, jj = X.nonzero()
        n_vals = ii.shape[0]
        dot_vals = np.empty(n_vals)
        n_components = W.shape[1]

        batch_size = max(n_components, n_vals // n_components)
        for start in range(0, n_vals, batch_size):
            batch = slice(start, start + batch_size)
            dot_vals[batch] = np.multiply(W[ii[batch], :], H.T[jj[batch], :]).sum(
                axis=1
            )

        WH = sp.coo_matrix((dot_vals, (ii, jj)), shape=X.shape)
        return WH.tocsr()
    else:
        return np.dot(W, H)


def _beta_loss_to_float(beta_loss):
    """Convert string beta_loss to float."""
    beta_loss_map = {"frobenius": 2, "kullback-leibler": 1, "itakura-saito": 0}
    if isinstance(beta_loss, str):
        beta_loss = beta_loss_map[beta_loss]
    return beta_loss


def _initialize_nmf(X, n_components, init=None, eps=1e-6, random_state=None, all_non_negative=False, W_filepath="tmp.mmap"):
    """Algorithms for NMF initialization.

    Computes an initial guess for the non-negative
    rank k matrix approximation for X: X = WH.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data matrix to be decomposed.

    n_components : int
        The number of components desired in the approximation.

    init :  {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}, default=None
        Method used to initialize the procedure.
        Valid options:

        - None: 'nndsvda' if n_components <= min(n_samples, n_features),
            otherwise 'random'.

        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)

        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

        - 'custom': use custom matrices W and H

        .. versionchanged:: 1.1
            When `init=None` and n_components is less than n_samples and n_features
            defaults to `nndsvda` instead of `nndsvd`.

    eps : float, default=1e-6
        Truncate all values less then this in output to zero.

    random_state : int, RandomState instance or None, default=None
        Used when ``init`` == 'nndsvdar' or 'random'. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    W : array-like of shape (n_samples, n_components)
        Initial guesses for solving X ~= WH.

    H : array-like of shape (n_components, n_features)
        Initial guesses for solving X ~= WH.

    References
    ----------
    C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    """
    if not all_non_negative:
        check_non_negative(X, "NMF initialization")
    n_samples, n_features = X.shape

    if (
        init is not None
        and init != "random"
        and n_components > min(n_samples, n_features)
    ):
        raise ValueError(
            "init = '{}' can only be used when "
            "n_components <= min(n_samples, n_features)".format(init)
        )

    if init is None:
        if n_components <= min(n_samples, n_features):
            init = "nndsvda"
        else:
            init = "random"

    # Random initialization
    if init == "random":
        n_rows, _ = X.shape
        sum_value = 0.0
        total_elements = 0
        chunk_size = int(1e5)
        for i in trange(0, n_rows, chunk_size):
            chunk = X[i:i + chunk_size]
            sum_value += chunk.sum().item()
            total_elements += chunk.shape[0]*chunk.shape[1]
        Xmean = sum_value / total_elements
        avg = np.sqrt(Xmean / n_components)
        # rng = check_random_state(random_state)
        rng = np.random.default_rng()
        H = avg * rng.standard_normal(size=(n_components, n_features), dtype=X.dtype)
        np.abs(H, out=H)
        if W_filepath is not None:
            print(n_samples, n_components, n_rows, chunk_size)
            W = np.memmap(W_filepath, dtype=X.dtype, mode='w+', shape=(n_samples, n_components))
            for i in trange(0, n_rows, chunk_size):
                j = min(i + chunk_size, n_samples)
                W[i:j] = np.abs(avg * rng.standard_normal(size=(j-i, n_components)).astype(X.dtype, copy=False))
            # rng.standard_normal(size=(n_samples, n_components), dtype=X.dtype, out=W)
            # W *= avg
            # np.abs(W, out=W)
        else:
            # W = avg * rng.standard_normal(size=(n_samples, n_components), dtype=X.dtype)
            W = np.empty((n_samples, n_components), dtype=X.dtype)
            rng.standard_normal(size=(n_samples, n_components), dtype=X.dtype, out=W)
            W *= avg
            np.abs(W, out=W)
        return W, H

    # NNDSVD initialization
    raise NotImplementedError
    U, S, V = randomized_svd(np.array(X), n_components, random_state=random_state)
    # U, S, V = randomized_svd_gpu(X, n_components, random_state=random_state, device=device)
    W = np.zeros_like(U)
    H = np.zeros_like(V)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if init == "nndsvd":
        pass
    elif init == "nndsvda":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif init == "nndsvdar":
        rng = check_random_state(random_state)
        avg = X.mean()
        W[W == 0] = abs(avg * rng.standard_normal(size=len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * rng.standard_normal(size=len(H[H == 0])) / 100)
    else:
        raise ValueError(
            "Invalid init parameter: got %r instead of one of %r"
            % (init, (None, "random", "nndsvd", "nndsvda", "nndsvdar"))
        )

    return W, H


def _update_coordinate_descent(X, W, Ht, l1_reg, l2_reg, shuffle, random_state):
    """Helper function for _fit_coordinate_descent.

    Update W to minimize the objective function, iterating once over all
    coordinates. By symmetry, to update H, one can call
    _update_coordinate_descent(X.T, Ht, W, ...).

    """
    n_components = Ht.shape[1]

    HHt = np.dot(Ht.T, Ht)
    XHt = safe_sparse_dot(X, Ht)

    # L2 regularization corresponds to increase of the diagonal of HHt
    if l2_reg != 0.0:
        # adds l2_reg only on the diagonal
        HHt.flat[:: n_components + 1] += l2_reg
    # L1 regularization corresponds to decrease of each element of XHt
    if l1_reg != 0.0:
        XHt -= l1_reg

    if shuffle:
        permutation = random_state.permutation(n_components)
    else:
        permutation = np.arange(n_components)
    # The following seems to be required on 64-bit Windows w/ Python 3.5.
    permutation = np.asarray(permutation, dtype=np.intp)
    return _update_cdnmf_fast(W, HHt, XHt, permutation)


def _fit_coordinate_descent(
    X,
    W,
    H,
    tol=1e-4,
    max_iter=200,
    l1_reg_W=0,
    l1_reg_H=0,
    l2_reg_W=0,
    l2_reg_H=0,
    update_H=True,
    verbose=0,
    shuffle=False,
    random_state=None,
):
    """Compute Non-negative Matrix Factorization (NMF) with Coordinate Descent

    The objective function is minimized with an alternating minimization of W
    and H. Each minimization is done with a cyclic (up to a permutation of the
    features) Coordinate Descent.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Constant matrix.

    W : array-like of shape (n_samples, n_components)
        Initial guess for the solution.

    H : array-like of shape (n_components, n_features)
        Initial guess for the solution.

    tol : float, default=1e-4
        Tolerance of the stopping condition.

    max_iter : int, default=200
        Maximum number of iterations before timing out.

    l1_reg_W : float, default=0.
        L1 regularization parameter for W.

    l1_reg_H : float, default=0.
        L1 regularization parameter for H.

    l2_reg_W : float, default=0.
        L2 regularization parameter for W.

    l2_reg_H : float, default=0.
        L2 regularization parameter for H.

    update_H : bool, default=True
        Set to True, both W and H will be estimated from initial guesses.
        Set to False, only W will be estimated.

    verbose : int, default=0
        The verbosity level.

    shuffle : bool, default=False
        If true, randomize the order of coordinates in the CD solver.

    random_state : int, RandomState instance or None, default=None
        Used to randomize the coordinates in the CD solver, when
        ``shuffle`` is set to ``True``. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    W : ndarray of shape (n_samples, n_components)
        Solution to the non-negative least squares problem.

    H : ndarray of shape (n_components, n_features)
        Solution to the non-negative least squares problem.

    n_iter : int
        The number of iterations done by the algorithm.

    References
    ----------
    .. [1] :doi:`"Fast local algorithms for large scale nonnegative matrix and tensor
       factorizations" <10.1587/transfun.E92.A.708>`
       Cichocki, Andrzej, and P. H. A. N. Anh-Huy. IEICE transactions on fundamentals
       of electronics, communications and computer sciences 92.3: 708-721, 2009.
    """
    # so W and Ht are both in C order in memory
    Ht = check_array(H.T, order="C")
    X = check_array(X, accept_sparse="csr")

    rng = check_random_state(random_state)

    for n_iter in range(1, max_iter + 1):
        violation = 0.0

        # Update W
        violation += _update_coordinate_descent(
            X, W, Ht, l1_reg_W, l2_reg_W, shuffle, rng
        )
        # Update H
        if update_H:
            violation += _update_coordinate_descent(
                X.T, Ht, W, l1_reg_H, l2_reg_H, shuffle, rng
            )

        if n_iter == 1:
            violation_init = violation

        if violation_init == 0:
            break

        if verbose:
            print("violation:", violation / violation_init)

        if violation / violation_init <= tol:
            if verbose:
                print("Converged at iteration", n_iter + 1)
            break

    return W, Ht.T, n_iter


def _multiplicative_update_w(
    X,
    W,
    H,
    beta_loss,
    l1_reg_W,
    l2_reg_W,
    gamma,
    H_sum=None,
    HHt=None,
    XHt=None,
    update_H=True,
):
    """Update W in Multiplicative Update NMF."""
    if beta_loss == 2:
        # Numerator
        if XHt is None:
            # XHt = safe_sparse_dot(X, H.T)
            XHt = X @ H.T
        if update_H:
            # avoid a copy of XHt, which will be re-computed (update_H=True)
            numerator = XHt
        else:
            # preserve the XHt, which is not re-computed (update_H=False)
            # numerator = XHt.copy()
            numerator = XHt.clone()

        # Denominator
        if HHt is None:
            # HHt = np.dot(H, H.T)
            HHt = H @ H.T
        # denominator = np.dot(W, HHt)
        denominator = W @ HHt

    else:
        raise NotImplementedError
        # Numerator
        # if X is sparse, compute WH only where X is non zero
        WH_safe_X = _special_sparse_dot(W, H, X)
        if sp.issparse(X):
            WH_safe_X_data = WH_safe_X.data
            X_data = X.data
        else:
            WH_safe_X_data = WH_safe_X
            X_data = X
            # copy used in the Denominator
            WH = WH_safe_X.copy()
            if beta_loss - 1.0 < 0:
                WH[WH < EPSILON] = EPSILON

        # to avoid taking a negative power of zero
        if beta_loss - 2.0 < 0:
            WH_safe_X_data[WH_safe_X_data < EPSILON] = EPSILON

        if beta_loss == 1:
            np.divide(X_data, WH_safe_X_data, out=WH_safe_X_data)
        elif beta_loss == 0:
            # speeds up computation time
            # refer to /numpy/numpy/issues/9363
            WH_safe_X_data **= -1
            WH_safe_X_data **= 2
            # element-wise multiplication
            WH_safe_X_data *= X_data
        else:
            WH_safe_X_data **= beta_loss - 2
            # element-wise multiplication
            WH_safe_X_data *= X_data

        # here numerator = dot(X * (dot(W, H) ** (beta_loss - 2)), H.T)
        numerator = safe_sparse_dot(WH_safe_X, H.T)

        # Denominator
        if beta_loss == 1:
            if H_sum is None:
                H_sum = np.sum(H, axis=1)  # shape(n_components, )
            denominator = H_sum[np.newaxis, :]

        else:
            # computation of WHHt = dot(dot(W, H) ** beta_loss - 1, H.T)
            if sp.issparse(X):
                # memory efficient computation
                # (compute row by row, avoiding the dense matrix WH)
                WHHt = np.empty(W.shape)
                for i in range(X.shape[0]):
                    WHi = np.dot(W[i, :], H)
                    if beta_loss - 1 < 0:
                        WHi[WHi < EPSILON] = EPSILON
                    WHi **= beta_loss - 1
                    WHHt[i, :] = np.dot(WHi, H.T)
            else:
                WH **= beta_loss - 1
                WHHt = np.dot(WH, H.T)
            denominator = WHHt

    # Add L1 and L2 regularization
    if l1_reg_W > 0:
        denominator += l1_reg_W
    if l2_reg_W > 0:
        denominator = denominator + l2_reg_W * W
    # denominator[denominator == 0] = EPSILON
    denominator[denominator == 0] = EPSILON_TORCH

    numerator /= denominator
    delta_W = numerator

    # gamma is in ]0, 1]
    if gamma != 1:
        delta_W **= gamma

    W *= delta_W

    return W, H_sum, HHt, XHt


def _multiplicative_update_h(
    X, W, H, beta_loss, l1_reg_H, l2_reg_H, gamma, A=None, B=None, rho=None
):
    """update H in Multiplicative Update NMF."""
    if beta_loss == 2:
        # numerator = safe_sparse_dot(W.T, X)
        # denominator = np.linalg.multi_dot([W.T, W, H])
        numerator = W.T @ X
        denominator = W.T @ W @ H

    else:
        raise NotImplementedError
        # Numerator
        WH_safe_X = _special_sparse_dot(W, H, X)
        if sp.issparse(X):
            WH_safe_X_data = WH_safe_X.data
            X_data = X.data
        else:
            WH_safe_X_data = WH_safe_X
            X_data = X
            # copy used in the Denominator
            WH = WH_safe_X.copy()
            if beta_loss - 1.0 < 0:
                WH[WH < EPSILON] = EPSILON

        # to avoid division by zero
        if beta_loss - 2.0 < 0:
            WH_safe_X_data[WH_safe_X_data < EPSILON] = EPSILON

        if beta_loss == 1:
            np.divide(X_data, WH_safe_X_data, out=WH_safe_X_data)
        elif beta_loss == 0:
            # speeds up computation time
            # refer to /numpy/numpy/issues/9363
            WH_safe_X_data **= -1
            WH_safe_X_data **= 2
            # element-wise multiplication
            WH_safe_X_data *= X_data
        else:
            WH_safe_X_data **= beta_loss - 2
            # element-wise multiplication
            WH_safe_X_data *= X_data

        # here numerator = dot(W.T, (dot(W, H) ** (beta_loss - 2)) * X)
        numerator = safe_sparse_dot(W.T, WH_safe_X)

        # Denominator
        if beta_loss == 1:
            W_sum = np.sum(W, axis=0)  # shape(n_components, )
            W_sum[W_sum == 0] = 1.0
            denominator = W_sum[:, np.newaxis]

        # beta_loss not in (1, 2)
        else:
            # computation of WtWH = dot(W.T, dot(W, H) ** beta_loss - 1)
            if sp.issparse(X):
                # memory efficient computation
                # (compute column by column, avoiding the dense matrix WH)
                WtWH = np.empty(H.shape)
                for i in range(X.shape[1]):
                    WHi = np.dot(W, H[:, i])
                    if beta_loss - 1 < 0:
                        WHi[WHi < EPSILON] = EPSILON
                    WHi **= beta_loss - 1
                    WtWH[:, i] = np.dot(W.T, WHi)
            else:
                WH **= beta_loss - 1
                WtWH = np.dot(W.T, WH)
            denominator = WtWH

    # Add L1 and L2 regularization
    if l1_reg_H > 0:
        denominator += l1_reg_H
    if l2_reg_H > 0:
        denominator = denominator + l2_reg_H * H
    # denominator[denominator == 0] = EPSILON
    denominator[denominator == 0] = EPSILON_TORCH

    if A is not None and B is not None:
        # Updates for the online nmf
        if gamma != 1:
            H **= 1 / gamma
        numerator *= H
        A *= rho
        B *= rho
        A += numerator
        B += denominator
        H = A / B

        if gamma != 1:
            H **= gamma
    else:
        delta_H = numerator
        delta_H /= denominator
        if gamma != 1:
            delta_H **= gamma
        H *= delta_H

    return H


def _fit_multiplicative_update(
    X,
    W,
    H,
    beta_loss="frobenius",
    max_iter=200,
    tol=1e-4,
    l1_reg_W=0,
    l1_reg_H=0,
    l2_reg_W=0,
    l2_reg_H=0,
    update_H=True,
    verbose=0,
):
    """Compute Non-negative Matrix Factorization with Multiplicative Update.

    The objective function is _beta_divergence(X, WH) and is minimized with an
    alternating minimization of W and H. Each minimization is done with a
    Multiplicative Update.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Constant input matrix.

    W : array-like of shape (n_samples, n_components)
        Initial guess for the solution.

    H : array-like of shape (n_components, n_features)
        Initial guess for the solution.

    beta_loss : float or {'frobenius', 'kullback-leibler', \
            'itakura-saito'}, default='frobenius'
        String must be in {'frobenius', 'kullback-leibler', 'itakura-saito'}.
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros.

    max_iter : int, default=200
        Number of iterations.

    tol : float, default=1e-4
        Tolerance of the stopping condition.

    l1_reg_W : float, default=0.
        L1 regularization parameter for W.

    l1_reg_H : float, default=0.
        L1 regularization parameter for H.

    l2_reg_W : float, default=0.
        L2 regularization parameter for W.

    l2_reg_H : float, default=0.
        L2 regularization parameter for H.

    update_H : bool, default=True
        Set to True, both W and H will be estimated from initial guesses.
        Set to False, only W will be estimated.

    verbose : int, default=0
        The verbosity level.

    Returns
    -------
    W : ndarray of shape (n_samples, n_components)
        Solution to the non-negative least squares problem.

    H : ndarray of shape (n_components, n_features)
        Solution to the non-negative least squares problem.

    n_iter : int
        The number of iterations done by the algorithm.

    References
    ----------
    Lee, D. D., & Seung, H., S. (2001). Algorithms for Non-negative Matrix
    Factorization. Adv. Neural Inform. Process. Syst.. 13.
    Fevotte, C., & Idier, J. (2011). Algorithms for nonnegative matrix
    factorization with the beta-divergence. Neural Computation, 23(9).
    """
    start_time = time.time()

    beta_loss = _beta_loss_to_float(beta_loss)

    # gamma for Maximization-Minimization (MM) algorithm [Fevotte 2011]
    if beta_loss < 1:
        gamma = 1.0 / (2.0 - beta_loss)
    elif beta_loss > 2:
        gamma = 1.0 / (beta_loss - 1.0)
    else:
        gamma = 1.0

    # used for the convergence criterion
    error_at_init = _beta_divergence(X, W, H, beta_loss, square_root=True)
    previous_error = error_at_init

    H_sum, HHt, XHt = None, None, None
    for n_iter in range(1, max_iter + 1):
        # update W
        # H_sum, HHt and XHt are saved and reused if not update_H
        W, H_sum, HHt, XHt = _multiplicative_update_w(
            X,
            W,
            H,
            beta_loss=beta_loss,
            l1_reg_W=l1_reg_W,
            l2_reg_W=l2_reg_W,
            gamma=gamma,
            H_sum=H_sum,
            HHt=HHt,
            XHt=XHt,
            update_H=update_H,
        )

        # necessary for stability with beta_loss < 1
        if beta_loss < 1:
            W[W < np.finfo(np.float64).eps] = 0.0

        # update H (only at fit or fit_transform)
        if update_H:
            H = _multiplicative_update_h(
                X,
                W,
                H,
                beta_loss=beta_loss,
                l1_reg_H=l1_reg_H,
                l2_reg_H=l2_reg_H,
                gamma=gamma,
            )

            # These values will be recomputed since H changed
            H_sum, HHt, XHt = None, None, None

            # necessary for stability with beta_loss < 1
            if beta_loss <= 1:
                H[H < np.finfo(np.float64).eps] = 0.0

        # test convergence criterion every 10 iterations
        if tol > 0 and n_iter % 10 == 0:
            error = _beta_divergence(X, W, H, beta_loss, square_root=True)

            if verbose:
                iter_time = time.time()
                print(
                    "Epoch %02d reached after %.3f seconds, error: %f"
                    % (n_iter, iter_time - start_time, error)
                )

            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    # do not print if we have already printed in the convergence test
    if verbose and (tol == 0 or n_iter % 10 != 0):
        end_time = time.time()
        print(
            "Epoch %02d reached after %.3f seconds." % (n_iter, end_time - start_time)
        )

    return W, H, n_iter


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "W": ["array-like", None],
        "H": ["array-like", None],
        "update_H": ["boolean"],
    },
    prefer_skip_nested_validation=False,
)
def non_negative_factorization(
    X,
    W=None,
    H=None,
    n_components="warn",
    *,
    init=None,
    update_H=True,
    solver="cd",
    beta_loss="frobenius",
    tol=1e-4,
    max_iter=200,
    alpha_W=0.0,
    alpha_H="same",
    l1_ratio=0.0,
    random_state=None,
    verbose=0,
    shuffle=False,
):
    """Compute Non-negative Matrix Factorization (NMF).

    Find two non-negative matrices (W, H) whose product approximates the non-
    negative matrix X. This factorization can be used for example for
    dimensionality reduction, source separation or topic extraction.

    The objective function is:

        .. math::

            L(W, H) &= 0.5 * ||X - WH||_{loss}^2

            &+ alpha\\_W * l1\\_ratio * n\\_features * ||vec(W)||_1

            &+ alpha\\_H * l1\\_ratio * n\\_samples * ||vec(H)||_1

            &+ 0.5 * alpha\\_W * (1 - l1\\_ratio) * n\\_features * ||W||_{Fro}^2

            &+ 0.5 * alpha\\_H * (1 - l1\\_ratio) * n\\_samples * ||H||_{Fro}^2

    Where:

    :math:`||A||_{Fro}^2 = \\sum_{i,j} A_{ij}^2` (Frobenius norm)

    :math:`||vec(A)||_1 = \\sum_{i,j} abs(A_{ij})` (Elementwise L1 norm)

    The generic norm :math:`||X - WH||_{loss}^2` may represent
    the Frobenius norm or another supported beta-divergence loss.
    The choice between options is controlled by the `beta_loss` parameter.

    The regularization terms are scaled by `n_features` for `W` and by `n_samples` for
    `H` to keep their impact balanced with respect to one another and to the data fit
    term as independent as possible of the size `n_samples` of the training set.

    The objective function is minimized with an alternating minimization of W
    and H. If H is given and update_H=False, it solves for W only.

    Note that the transformed data is named W and the components matrix is named H. In
    the NMF literature, the naming convention is usually the opposite since the data
    matrix X is transposed.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Constant matrix.

    W : array-like of shape (n_samples, n_components), default=None
        If `init='custom'`, it is used as initial guess for the solution.
        If `update_H=False`, it is initialised as an array of zeros, unless
        `solver='mu'`, then it is filled with values calculated by
        `np.sqrt(X.mean() / self._n_components)`.
        If `None`, uses the initialisation method specified in `init`.

    H : array-like of shape (n_components, n_features), default=None
        If `init='custom'`, it is used as initial guess for the solution.
        If `update_H=False`, it is used as a constant, to solve for W only.
        If `None`, uses the initialisation method specified in `init`.

    n_components : int or {'auto'} or None, default=None
        Number of components, if n_components is not set all features
        are kept.
        If `n_components='auto'`, the number of components is automatically inferred
        from `W` or `H` shapes.

        .. versionchanged:: 1.4
            Added `'auto'` value.

    init : {'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'}, default=None
        Method used to initialize the procedure.

        Valid options:

        - None: 'nndsvda' if n_components < n_features, otherwise 'random'.
        - 'random': non-negative random matrices, scaled with:
          `sqrt(X.mean() / n_components)`
        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
          initialization (better for sparseness)
        - 'nndsvda': NNDSVD with zeros filled with the average of X
          (better when sparsity is not desired)
        - 'nndsvdar': NNDSVD with zeros filled with small random values
          (generally faster, less accurate alternative to NNDSVDa
          for when sparsity is not desired)
        - 'custom': If `update_H=True`, use custom matrices W and H which must both
          be provided. If `update_H=False`, then only custom matrix H is used.

        .. versionchanged:: 0.23
            The default value of `init` changed from 'random' to None in 0.23.

        .. versionchanged:: 1.1
            When `init=None` and n_components is less than n_samples and n_features
            defaults to `nndsvda` instead of `nndsvd`.

    update_H : bool, default=True
        Set to True, both W and H will be estimated from initial guesses.
        Set to False, only W will be estimated.

    solver : {'cd', 'mu'}, default='cd'
        Numerical solver to use:

        - 'cd' is a Coordinate Descent solver that uses Fast Hierarchical
          Alternating Least Squares (Fast HALS).
        - 'mu' is a Multiplicative Update solver.

        .. versionadded:: 0.17
           Coordinate Descent solver.

        .. versionadded:: 0.19
           Multiplicative Update solver.

    beta_loss : float or {'frobenius', 'kullback-leibler', \
            'itakura-saito'}, default='frobenius'
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros. Used only in 'mu' solver.

        .. versionadded:: 0.19

    tol : float, default=1e-4
        Tolerance of the stopping condition.

    max_iter : int, default=200
        Maximum number of iterations before timing out.

    alpha_W : float, default=0.0
        Constant that multiplies the regularization terms of `W`. Set it to zero
        (default) to have no regularization on `W`.

        .. versionadded:: 1.0

    alpha_H : float or "same", default="same"
        Constant that multiplies the regularization terms of `H`. Set it to zero to
        have no regularization on `H`. If "same" (default), it takes the same value as
        `alpha_W`.

        .. versionadded:: 1.0

    l1_ratio : float, default=0.0
        The regularization mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an elementwise L2 penalty
        (aka Frobenius Norm).
        For l1_ratio = 1 it is an elementwise L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

    random_state : int, RandomState instance or None, default=None
        Used for NMF initialisation (when ``init`` == 'nndsvdar' or
        'random'), and in Coordinate Descent. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : int, default=0
        The verbosity level.

    shuffle : bool, default=False
        If true, randomize the order of coordinates in the CD solver.

    Returns
    -------
    W : ndarray of shape (n_samples, n_components)
        Solution to the non-negative least squares problem.

    H : ndarray of shape (n_components, n_features)
        Solution to the non-negative least squares problem.

    n_iter : int
        Actual number of iterations.

    References
    ----------
    .. [1] :doi:`"Fast local algorithms for large scale nonnegative matrix and tensor
       factorizations" <10.1587/transfun.E92.A.708>`
       Cichocki, Andrzej, and P. H. A. N. Anh-Huy. IEICE transactions on fundamentals
       of electronics, communications and computer sciences 92.3: 708-721, 2009.

    .. [2] :doi:`"Algorithms for nonnegative matrix factorization with the
       beta-divergence" <10.1162/NECO_a_00168>`
       Fevotte, C., & Idier, J. (2011). Neural Computation, 23(9).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> from sklearn.decomposition import non_negative_factorization
    >>> W, H, n_iter = non_negative_factorization(
    ...     X, n_components=2, init='random', random_state=0)
    """
    est = NMF(
        n_components=n_components,
        init=init,
        solver=solver,
        beta_loss=beta_loss,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        l1_ratio=l1_ratio,
        verbose=verbose,
        shuffle=shuffle,
    )
    est._validate_params()

    X = check_array(X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32])

    with config_context(assume_finite=True):
        W, H, n_iter = est._fit_transform(X, W=W, H=H, update_H=update_H)

    return W, H, n_iter


class _BaseNMF(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator, ABC):
    """Base class for NMF and MiniBatchNMF."""

    # This prevents ``set_split_inverse_transform`` to be generated for the
    # non-standard ``Xt`` arg on ``inverse_transform``.
    # TODO(1.7): remove when Xt is removed in v1.7 for inverse_transform
    __metadata_request__inverse_transform = {"Xt": metadata_routing.UNUSED}

    _parameter_constraints: dict = {
        "n_components": [
            Interval(Integral, 1, None, closed="left"),
            None,
            StrOptions({"auto"}),
            Hidden(StrOptions({"warn"})),
        ],
        "init": [
            StrOptions({"random", "nndsvd", "nndsvda", "nndsvdar", "custom"}),
            None,
        ],
        "beta_loss": [
            StrOptions({"frobenius", "kullback-leibler", "itakura-saito"}),
            Real,
        ],
        "tol": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
        "alpha_W": [Interval(Real, 0, None, closed="left")],
        "alpha_H": [Interval(Real, 0, None, closed="left"), StrOptions({"same"})],
        "l1_ratio": [Interval(Real, 0, 1, closed="both")],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        n_components="warn",
        *,
        init=None,
        beta_loss="frobenius",
        tol=1e-4,
        max_iter=200,
        random_state=None,
        alpha_W=0.0,
        alpha_H="same",
        l1_ratio=0.0,
        verbose=0,
        device="cpu",
        all_non_negative=False,
    ):
        self.n_components = n_components
        self.init = init
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H
        self.l1_ratio = l1_ratio
        self.verbose = verbose
        self.device = device
        self.all_non_negative = all_non_negative

    def _check_params(self, X):
        # n_components
        self._n_components = self.n_components
        if self.n_components == "warn":
            warnings.warn(
                (
                    "The default value of `n_components` will change from `None` to"
                    " `'auto'` in 1.6. Set the value of `n_components` to `None`"
                    " explicitly to suppress the warning."
                ),
                FutureWarning,
            )
            self._n_components = None  # Keeping the old default value
        if self._n_components is None:
            self._n_components = X.shape[1]

        # beta_loss
        self._beta_loss = _beta_loss_to_float(self.beta_loss)

    def _check_w_h(self, X, W, H, update_H, W_filepath="tmp.mmap"):
        """Check W and H, or initialize them."""
        n_samples, n_features = X.shape

        if self.init == "custom" and update_H:
            raise NotImplementedError
            _check_init(H, (self._n_components, n_features), "NMF (input H)")
            _check_init(W, (n_samples, self._n_components), "NMF (input W)")
            if self._n_components == "auto":
                self._n_components = H.shape[0]

            if H.dtype != X.dtype or W.dtype != X.dtype:
                raise TypeError(
                    "H and W should have the same dtype as X. Got "
                    "H.dtype = {} and W.dtype = {}.".format(H.dtype, W.dtype)
                )

        elif not update_H:
            raise NotImplementedError
            if W is not None:
                warnings.warn(
                    "When update_H=False, the provided initial W is not used.",
                    RuntimeWarning,
                )

            _check_init(H, (self._n_components, n_features), "NMF (input H)")
            if self._n_components == "auto":
                self._n_components = H.shape[0]

            if H.dtype != X.dtype:
                raise TypeError(
                    "H should have the same dtype as X. Got H.dtype = {}.".format(
                        H.dtype
                    )
                )

            # 'mu' solver should not be initialized by zeros
            if self.solver == "mu":
                avg = np.sqrt(X.mean() / self._n_components)
                W = np.full((n_samples, self._n_components), avg, dtype=X.dtype)
            else:
                W = np.zeros((n_samples, self._n_components), dtype=X.dtype)

        else:
            if W is not None or H is not None:
                warnings.warn(
                    (
                        "When init!='custom', provided W or H are ignored. Set "
                        " init='custom' to use them as initialization."
                    ),
                    RuntimeWarning,
                )

            if self._n_components == "auto":
                self._n_components = X.shape[1]

            W, H = _initialize_nmf(
                X, self._n_components, init=self.init, random_state=self.random_state, all_non_negative=self.all_non_negative, W_filepath=W_filepath,
            )

        return W, H

    def _compute_regularization(self, X):
        """Compute scaled regularization terms."""
        n_samples, n_features = X.shape
        alpha_W = self.alpha_W
        alpha_H = self.alpha_W if self.alpha_H == "same" else self.alpha_H

        l1_reg_W = n_features * alpha_W * self.l1_ratio
        l1_reg_H = n_samples * alpha_H * self.l1_ratio
        l2_reg_W = n_features * alpha_W * (1.0 - self.l1_ratio)
        l2_reg_H = n_samples * alpha_H * (1.0 - self.l1_ratio)

        return l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H

    def fit(self, X, y=None, **params):
        """Learn a NMF model for the data X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        **params : kwargs
            Parameters (keyword arguments) and values passed to
            the fit_transform instance.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # param validation is done in fit_transform

        self.fit_transform(X, **params)
        return self

    def inverse_transform(self, X=None, *, Xt=None):
        """Transform data back to its original space.

        .. versionadded:: 0.18

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_components)
            Transformed data matrix.

        Xt : {ndarray, sparse matrix} of shape (n_samples, n_components)
            Transformed data matrix.

            .. deprecated:: 1.5
                `Xt` was deprecated in 1.5 and will be removed in 1.7. Use `X` instead.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Returns a data matrix of the original shape.
        """

        X = _deprecate_Xt_in_inverse_transform(X, Xt)

        check_is_fitted(self)
        return X @ self.components_

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        return self.components_.shape[0]

    def _more_tags(self):
        return {
            "requires_positive_X": True,
            "preserves_dtype": [np.float64, np.float32],
        }


class NMF(_BaseNMF):
    """Non-Negative Matrix Factorization (NMF).

    Find two non-negative matrices, i.e. matrices with all non-negative elements, (W, H)
    whose product approximates the non-negative matrix X. This factorization can be used
    for example for dimensionality reduction, source separation or topic extraction.

    The objective function is:

        .. math::

            L(W, H) &= 0.5 * ||X - WH||_{loss}^2

            &+ alpha\\_W * l1\\_ratio * n\\_features * ||vec(W)||_1

            &+ alpha\\_H * l1\\_ratio * n\\_samples * ||vec(H)||_1

            &+ 0.5 * alpha\\_W * (1 - l1\\_ratio) * n\\_features * ||W||_{Fro}^2

            &+ 0.5 * alpha\\_H * (1 - l1\\_ratio) * n\\_samples * ||H||_{Fro}^2

    Where:

    :math:`||A||_{Fro}^2 = \\sum_{i,j} A_{ij}^2` (Frobenius norm)

    :math:`||vec(A)||_1 = \\sum_{i,j} abs(A_{ij})` (Elementwise L1 norm)

    The generic norm :math:`||X - WH||_{loss}` may represent
    the Frobenius norm or another supported beta-divergence loss.
    The choice between options is controlled by the `beta_loss` parameter.

    The regularization terms are scaled by `n_features` for `W` and by `n_samples` for
    `H` to keep their impact balanced with respect to one another and to the data fit
    term as independent as possible of the size `n_samples` of the training set.

    The objective function is minimized with an alternating minimization of W
    and H.

    Note that the transformed data is named W and the components matrix is named H. In
    the NMF literature, the naming convention is usually the opposite since the data
    matrix X is transposed.

    Read more in the :ref:`User Guide <NMF>`.

    Parameters
    ----------
    n_components : int or {'auto'} or None, default=None
        Number of components, if n_components is not set all features
        are kept.
        If `n_components='auto'`, the number of components is automatically inferred
        from W or H shapes.

        .. versionchanged:: 1.4
            Added `'auto'` value.

    init : {'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'}, default=None
        Method used to initialize the procedure.
        Valid options:

        - `None`: 'nndsvda' if n_components <= min(n_samples, n_features),
          otherwise random.

        - `'random'`: non-negative random matrices, scaled with:
          `sqrt(X.mean() / n_components)`

        - `'nndsvd'`: Nonnegative Double Singular Value Decomposition (NNDSVD)
          initialization (better for sparseness)

        - `'nndsvda'`: NNDSVD with zeros filled with the average of X
          (better when sparsity is not desired)

        - `'nndsvdar'` NNDSVD with zeros filled with small random values
          (generally faster, less accurate alternative to NNDSVDa
          for when sparsity is not desired)

        - `'custom'`: Use custom matrices `W` and `H` which must both be provided.

        .. versionchanged:: 1.1
            When `init=None` and n_components is less than n_samples and n_features
            defaults to `nndsvda` instead of `nndsvd`.

    solver : {'cd', 'mu'}, default='cd'
        Numerical solver to use:

        - 'cd' is a Coordinate Descent solver.
        - 'mu' is a Multiplicative Update solver.

        .. versionadded:: 0.17
           Coordinate Descent solver.

        .. versionadded:: 0.19
           Multiplicative Update solver.

    beta_loss : float or {'frobenius', 'kullback-leibler', \
            'itakura-saito'}, default='frobenius'
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros. Used only in 'mu' solver.

        .. versionadded:: 0.19

    tol : float, default=1e-4
        Tolerance of the stopping condition.

    max_iter : int, default=200
        Maximum number of iterations before timing out.

    random_state : int, RandomState instance or None, default=None
        Used for initialisation (when ``init`` == 'nndsvdar' or
        'random'), and in Coordinate Descent. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    alpha_W : float, default=0.0
        Constant that multiplies the regularization terms of `W`. Set it to zero
        (default) to have no regularization on `W`.

        .. versionadded:: 1.0

    alpha_H : float or "same", default="same"
        Constant that multiplies the regularization terms of `H`. Set it to zero to
        have no regularization on `H`. If "same" (default), it takes the same value as
        `alpha_W`.

        .. versionadded:: 1.0

    l1_ratio : float, default=0.0
        The regularization mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an elementwise L2 penalty
        (aka Frobenius Norm).
        For l1_ratio = 1 it is an elementwise L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

        .. versionadded:: 0.17
           Regularization parameter *l1_ratio* used in the Coordinate Descent
           solver.

    verbose : int, default=0
        Whether to be verbose.

    shuffle : bool, default=False
        If true, randomize the order of coordinates in the CD solver.

        .. versionadded:: 0.17
           *shuffle* parameter used in the Coordinate Descent solver.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Factorization matrix, sometimes called 'dictionary'.

    n_components_ : int
        The number of components. It is same as the `n_components` parameter
        if it was given. Otherwise, it will be same as the number of
        features.

    reconstruction_err_ : float
        Frobenius norm of the matrix difference, or beta-divergence, between
        the training data ``X`` and the reconstructed data ``WH`` from
        the fitted model.

    n_iter_ : int
        Actual number of iterations.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    DictionaryLearning : Find a dictionary that sparsely encodes data.
    MiniBatchSparsePCA : Mini-batch Sparse Principal Components Analysis.
    PCA : Principal component analysis.
    SparseCoder : Find a sparse representation of data from a fixed,
        precomputed dictionary.
    SparsePCA : Sparse Principal Components Analysis.
    TruncatedSVD : Dimensionality reduction using truncated SVD.

    References
    ----------
    .. [1] :doi:`"Fast local algorithms for large scale nonnegative matrix and tensor
       factorizations" <10.1587/transfun.E92.A.708>`
       Cichocki, Andrzej, and P. H. A. N. Anh-Huy. IEICE transactions on fundamentals
       of electronics, communications and computer sciences 92.3: 708-721, 2009.

    .. [2] :doi:`"Algorithms for nonnegative matrix factorization with the
       beta-divergence" <10.1162/NECO_a_00168>`
       Fevotte, C., & Idier, J. (2011). Neural Computation, 23(9).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> from sklearn.decomposition import NMF
    >>> model = NMF(n_components=2, init='random', random_state=0)
    >>> W = model.fit_transform(X)
    >>> H = model.components_
    """

    _parameter_constraints: dict = {
        **_BaseNMF._parameter_constraints,
        "solver": [StrOptions({"mu", "cd"})],
        "shuffle": ["boolean"],
    }

    def __init__(
        self,
        n_components="warn",
        *,
        init=None,
        solver="cd",
        beta_loss="frobenius",
        tol=1e-4,
        max_iter=200,
        random_state=None,
        alpha_W=0.0,
        alpha_H="same",
        l1_ratio=0.0,
        verbose=0,
        shuffle=False,
    ):
        super().__init__(
            n_components=n_components,
            init=init,
            beta_loss=beta_loss,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            l1_ratio=l1_ratio,
            verbose=verbose,
        )

        self.solver = solver
        self.shuffle = shuffle

    def _check_params(self, X):
        super()._check_params(X)

        # solver
        if self.solver != "mu" and self.beta_loss not in (2, "frobenius"):
            # 'mu' is the only solver that handles other beta losses than 'frobenius'
            raise ValueError(
                f"Invalid beta_loss parameter: solver {self.solver!r} does not handle "
                f"beta_loss = {self.beta_loss!r}"
            )
        if self.solver == "mu" and self.init == "nndsvd":
            warnings.warn(
                (
                    "The multiplicative update ('mu') solver cannot update "
                    "zeros present in the initialization, and so leads to "
                    "poorer results when used jointly with init='nndsvd'. "
                    "You may try init='nndsvda' or init='nndsvdar' instead."
                ),
                UserWarning,
            )

        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None, W=None, H=None):
        """Learn a NMF model for the data X and returns the transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        W : array-like of shape (n_samples, n_components), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `None`, uses the initialisation method specified in `init`.

        H : array-like of shape (n_components, n_features), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `None`, uses the initialisation method specified in `init`.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        X = self._validate_data(
            X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32]
        )

        with config_context(assume_finite=True):
            W, H, n_iter = self._fit_transform(X, W=W, H=H)

        self.reconstruction_err_ = _beta_divergence(
            X, W, H, self._beta_loss, square_root=True
        )

        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_iter_ = n_iter

        return W

    def _fit_transform(self, X, y=None, W=None, H=None, update_H=True):
        """Learn a NMF model for the data X and returns the transformed data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed

        y : Ignored

        W : array-like of shape (n_samples, n_components), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `update_H=False`, it is initialised as an array of zeros, unless
            `solver='mu'`, then it is filled with values calculated by
            `np.sqrt(X.mean() / self._n_components)`.
            If `None`, uses the initialisation method specified in `init`.

        H : array-like of shape (n_components, n_features), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `update_H=False`, it is used as a constant, to solve for W only.
            If `None`, uses the initialisation method specified in `init`.

        update_H : bool, default=True
            If True, both W and H will be estimated from initial guesses,
            this corresponds to a call to the 'fit_transform' method.
            If False, only W will be estimated, this corresponds to a call
            to the 'transform' method.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.

        H : ndarray of shape (n_components, n_features)
            Factorization matrix, sometimes called 'dictionary'.

        n_iter_ : int
            Actual number of iterations.
        """
        if not self.all_non_negative:
            check_non_negative(X, "NMF (input X)")

        # check parameters
        self._check_params(X)

        if X.min() == 0 and self._beta_loss <= 0:
            raise ValueError(
                "When beta_loss <= 0 and X contains zeros, "
                "the solver may diverge. Please add small values "
                "to X, or use a positive beta_loss."
            )

        # initialize or check W and H
        W, H = self._check_w_h(X, W, H, update_H)

        # scale the regularization terms
        l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H = self._compute_regularization(X)

        if self.solver == "cd":
            W, H, n_iter = _fit_coordinate_descent(
                X,
                W,
                H,
                self.tol,
                self.max_iter,
                l1_reg_W,
                l1_reg_H,
                l2_reg_W,
                l2_reg_H,
                update_H=update_H,
                verbose=self.verbose,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
        elif self.solver == "mu":
            W, H, n_iter, *_ = _fit_multiplicative_update(
                X,
                W,
                H,
                self._beta_loss,
                self.max_iter,
                self.tol,
                l1_reg_W,
                l1_reg_H,
                l2_reg_W,
                l2_reg_H,
                update_H,
                self.verbose,
            )
        else:
            raise ValueError("Invalid solver parameter '%s'." % self.solver)

        if n_iter == self.max_iter and self.tol > 0:
            warnings.warn(
                "Maximum number of iterations %d reached. Increase "
                "it to improve convergence." % self.max_iter,
                ConvergenceWarning,
            )

        return W, H, n_iter

    def transform(self, X):
        """Transform the data X according to the fitted NMF model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self)
        X = self._validate_data(
            X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32], reset=False
        )

        with config_context(assume_finite=True):
            W, *_ = self._fit_transform(X, H=self.components_, update_H=False)

        return W


class MiniBatchNMF(_BaseNMF):
    """Mini-Batch Non-Negative Matrix Factorization (NMF).

    .. versionadded:: 1.1

    Find two non-negative matrices, i.e. matrices with all non-negative elements,
    (`W`, `H`) whose product approximates the non-negative matrix `X`. This
    factorization can be used for example for dimensionality reduction, source
    separation or topic extraction.

    The objective function is:

        .. math::

            L(W, H) &= 0.5 * ||X - WH||_{loss}^2

            &+ alpha\\_W * l1\\_ratio * n\\_features * ||vec(W)||_1

            &+ alpha\\_H * l1\\_ratio * n\\_samples * ||vec(H)||_1

            &+ 0.5 * alpha\\_W * (1 - l1\\_ratio) * n\\_features * ||W||_{Fro}^2

            &+ 0.5 * alpha\\_H * (1 - l1\\_ratio) * n\\_samples * ||H||_{Fro}^2

    Where:

    :math:`||A||_{Fro}^2 = \\sum_{i,j} A_{ij}^2` (Frobenius norm)

    :math:`||vec(A)||_1 = \\sum_{i,j} abs(A_{ij})` (Elementwise L1 norm)

    The generic norm :math:`||X - WH||_{loss}^2` may represent
    the Frobenius norm or another supported beta-divergence loss.
    The choice between options is controlled by the `beta_loss` parameter.

    The objective function is minimized with an alternating minimization of `W`
    and `H`.

    Note that the transformed data is named `W` and the components matrix is
    named `H`. In the NMF literature, the naming convention is usually the opposite
    since the data matrix `X` is transposed.

    Read more in the :ref:`User Guide <MiniBatchNMF>`.

    Parameters
    ----------
    n_components : int or {'auto'} or None, default=None
        Number of components, if `n_components` is not set all features
        are kept.
        If `n_components='auto'`, the number of components is automatically inferred
        from W or H shapes.

        .. versionchanged:: 1.4
            Added `'auto'` value.

    init : {'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'}, default=None
        Method used to initialize the procedure.
        Valid options:

        - `None`: 'nndsvda' if `n_components <= min(n_samples, n_features)`,
          otherwise random.

        - `'random'`: non-negative random matrices, scaled with:
          `sqrt(X.mean() / n_components)`

        - `'nndsvd'`: Nonnegative Double Singular Value Decomposition (NNDSVD)
          initialization (better for sparseness).

        - `'nndsvda'`: NNDSVD with zeros filled with the average of X
          (better when sparsity is not desired).

        - `'nndsvdar'` NNDSVD with zeros filled with small random values
          (generally faster, less accurate alternative to NNDSVDa
          for when sparsity is not desired).

        - `'custom'`: Use custom matrices `W` and `H` which must both be provided.

    batch_size : int, default=1024
        Number of samples in each mini-batch. Large batch sizes
        give better long-term convergence at the cost of a slower start.

    beta_loss : float or {'frobenius', 'kullback-leibler', \
            'itakura-saito'}, default='frobenius'
        Beta divergence to be minimized, measuring the distance between `X`
        and the dot product `WH`. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for `beta_loss <= 0` (or 'itakura-saito'), the input
        matrix `X` cannot contain zeros.

    tol : float, default=1e-4
        Control early stopping based on the norm of the differences in `H`
        between 2 steps. To disable early stopping based on changes in `H`, set
        `tol` to 0.0.

    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini batches
        that does not yield an improvement on the smoothed cost function.
        To disable convergence detection based on cost function, set
        `max_no_improvement` to None.

    max_iter : int, default=200
        Maximum number of iterations over the complete dataset before
        timing out.

    alpha_W : float, default=0.0
        Constant that multiplies the regularization terms of `W`. Set it to zero
        (default) to have no regularization on `W`.

    alpha_H : float or "same", default="same"
        Constant that multiplies the regularization terms of `H`. Set it to zero to
        have no regularization on `H`. If "same" (default), it takes the same value as
        `alpha_W`.

    l1_ratio : float, default=0.0
        The regularization mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an elementwise L2 penalty
        (aka Frobenius Norm).
        For l1_ratio = 1 it is an elementwise L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

    forget_factor : float, default=0.7
        Amount of rescaling of past information. Its value could be 1 with
        finite datasets. Choosing values < 1 is recommended with online
        learning as more recent batches will weight more than past batches.

    fresh_restarts : bool, default=False
        Whether to completely solve for W at each step. Doing fresh restarts will likely
        lead to a better solution for a same number of iterations but it is much slower.

    fresh_restarts_max_iter : int, default=30
        Maximum number of iterations when solving for W at each step. Only used when
        doing fresh restarts. These iterations may be stopped early based on a small
        change of W controlled by `tol`.

    transform_max_iter : int, default=None
        Maximum number of iterations when solving for W at transform time.
        If None, it defaults to `max_iter`.

    random_state : int, RandomState instance or None, default=None
        Used for initialisation (when ``init`` == 'nndsvdar' or
        'random'), and in Coordinate Descent. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : bool, default=False
        Whether to be verbose.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Factorization matrix, sometimes called 'dictionary'.

    n_components_ : int
        The number of components. It is same as the `n_components` parameter
        if it was given. Otherwise, it will be same as the number of
        features.

    reconstruction_err_ : float
        Frobenius norm of the matrix difference, or beta-divergence, between
        the training data `X` and the reconstructed data `WH` from
        the fitted model.

    n_iter_ : int
        Actual number of started iterations over the whole dataset.

    n_steps_ : int
        Number of mini-batches processed.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    NMF : Non-negative matrix factorization.
    MiniBatchDictionaryLearning : Finds a dictionary that can best be used to represent
        data using a sparse code.

    References
    ----------
    .. [1] :doi:`"Fast local algorithms for large scale nonnegative matrix and tensor
       factorizations" <10.1587/transfun.E92.A.708>`
       Cichocki, Andrzej, and P. H. A. N. Anh-Huy. IEICE transactions on fundamentals
       of electronics, communications and computer sciences 92.3: 708-721, 2009.

    .. [2] :doi:`"Algorithms for nonnegative matrix factorization with the
       beta-divergence" <10.1162/NECO_a_00168>`
       Fevotte, C., & Idier, J. (2011). Neural Computation, 23(9).

    .. [3] :doi:`"Online algorithms for nonnegative matrix factorization with the
       Itakura-Saito divergence" <10.1109/ASPAA.2011.6082314>`
       Lefevre, A., Bach, F., Fevotte, C. (2011). WASPA.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> from sklearn.decomposition import MiniBatchNMF
    >>> model = MiniBatchNMF(n_components=2, init='random', random_state=0)
    >>> W = model.fit_transform(X)
    >>> H = model.components_
    """

    _parameter_constraints: dict = {
        **_BaseNMF._parameter_constraints,
        "max_no_improvement": [Interval(Integral, 1, None, closed="left"), None],
        "batch_size": [Interval(Integral, 1, None, closed="left")],
        "forget_factor": [Interval(Real, 0, 1, closed="both")],
        "fresh_restarts": ["boolean"],
        "fresh_restarts_max_iter": [Interval(Integral, 1, None, closed="left")],
        "transform_max_iter": [Interval(Integral, 1, None, closed="left"), None],
    }

    def __init__(
        self,
        n_components="warn",
        *,
        init=None,
        batch_size=1024,
        beta_loss="frobenius",
        tol=1e-4,
        max_no_improvement=10,
        max_iter=200,
        alpha_W=0.0,
        alpha_H="same",
        l1_ratio=0.0,
        forget_factor=0.7,
        fresh_restarts=False,
        fresh_restarts_max_iter=30,
        transform_max_iter=None,
        random_state=None,
        verbose=0,
        device="cpu",
        all_non_negative=False,
        W_filepath="tmp.mmap",
    ):
        super().__init__(
            n_components=n_components,
            init=init,
            beta_loss=beta_loss,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            l1_ratio=l1_ratio,
            verbose=verbose,
            device=device,
            all_non_negative=all_non_negative,
        )

        self.max_no_improvement = max_no_improvement
        self.batch_size = batch_size
        self.forget_factor = forget_factor
        self.fresh_restarts = fresh_restarts
        self.fresh_restarts_max_iter = fresh_restarts_max_iter
        self.transform_max_iter = transform_max_iter
        self.W_filepath = W_filepath

    def _check_params(self, X):
        super()._check_params(X)

        # batch_size
        self._batch_size = min(self.batch_size, X.shape[0])

        # forget_factor
        self._rho = self.forget_factor ** (self._batch_size / X.shape[0])

        # gamma for Maximization-Minimization (MM) algorithm [Fevotte 2011]
        if self._beta_loss < 1:
            self._gamma = 1.0 / (2.0 - self._beta_loss)
        elif self._beta_loss > 2:
            self._gamma = 1.0 / (self._beta_loss - 1.0)
        else:
            self._gamma = 1.0

        # transform_max_iter
        self._transform_max_iter = (
            self.max_iter
            if self.transform_max_iter is None
            else self.transform_max_iter
        )

        return self

    def _solve_W(self, X, H, max_iter):
        """Minimize the objective function w.r.t W.

        Update W with H being fixed, until convergence. This is the heart
        of `transform` but it's also used during `fit` when doing fresh restarts.
        """
        avg = np.sqrt(X.mean() / self._n_components)
        W = np.full((X.shape[0], self._n_components), avg, dtype=X.dtype)
        W_buffer = W.copy()

        # Get scaled regularization terms. Done for each minibatch to take into account
        # variable sizes of minibatches.
        l1_reg_W, _, l2_reg_W, _ = self._compute_regularization(X)

        for _ in range(max_iter):
            W, *_ = _multiplicative_update_w(
                X, W, H, self._beta_loss, l1_reg_W, l2_reg_W, self._gamma
            )

            W_diff = linalg.norm(W - W_buffer) / linalg.norm(W)
            if self.tol > 0 and W_diff <= self.tol:
                break

            W_buffer[:] = W

        return W

    def _minibatch_step(self, X, W, H, update_H):
        """Perform the update of W and H for one minibatch."""
        batch_size = X.shape[0]

        # get scaled regularization terms. Done for each minibatch to take into account
        # variable sizes of minibatches.
        l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H = self._compute_regularization(X)

        # update W
        if self.fresh_restarts or W is None:
            raise NotImplementedError
            W = self._solve_W(X, H, self.fresh_restarts_max_iter)
        else:
            W, *_ = _multiplicative_update_w(
                X, W, H, self._beta_loss, l1_reg_W, l2_reg_W, self._gamma
            )

        # necessary for stability with beta_loss < 1
        if self._beta_loss < 1:
            raise NotImplementedError
            W[W < np.finfo(np.float64).eps] = 0.0

        batch_cost = (
            _beta_divergence(X, W, H, self._beta_loss)
            + l1_reg_W * W.sum()
            + l1_reg_H * H.sum()
            + l2_reg_W * (W**2).sum()
            + l2_reg_H * (H**2).sum()
        ) / batch_size

        # update H (only at fit or fit_transform)
        if update_H:
            H[:] = _multiplicative_update_h(
                X,
                W,
                H,
                beta_loss=self._beta_loss,
                l1_reg_H=l1_reg_H,
                l2_reg_H=l2_reg_H,
                gamma=self._gamma,
                A=self._components_numerator,
                B=self._components_denominator,
                rho=self._rho,
            )

            # necessary for stability with beta_loss < 1
            if self._beta_loss <= 1:
                raise NotImplementedError
                H[H < np.finfo(np.float64).eps] = 0.0

        return batch_cost

    def _minibatch_convergence(
        self, X, batch_cost, H, H_buffer, n_samples, step, n_steps, pbar=None
    ):
        """Helper function to encapsulate the early stopping logic"""
        batch_size = X.shape[0]

        # counts steps starting from 1 for user friendly verbose mode.
        step = step + 1

        # Ignore first iteration because H is not updated yet.
        if step == 1:
            out_string = f"Minibatch step {step}/{n_steps}: mean batch cost: {batch_cost}"
            if self.verbose and pbar is not None:
                pbar.set_description(out_string)
            elif self.verbose:
                print(out_string)
            return False

        # Compute an Exponentially Weighted Average of the cost function to
        # monitor the convergence while discarding minibatch-local stochastic
        # variability: https://en.wikipedia.org/wiki/Moving_average
        if self._ewa_cost is None:
            self._ewa_cost = batch_cost
        else:
            alpha = batch_size / (n_samples + 1)
            alpha = min(alpha, 1)
            self._ewa_cost = self._ewa_cost * (1 - alpha) + batch_cost * alpha

        # Log progress to be able to monitor convergence
        out_string = f"Minibatch step {step}/{n_steps}: mean batch cost: {batch_cost}, ewa cost: {self._ewa_cost}"
        if self.verbose and pbar is not None:
            pbar.set_description(out_string)
        elif self.verbose:
            print(out_string)

        # Early stopping based on change of H
        # H_diff = linalg.norm(H - H_buffer) / linalg.norm(H)
        H_diff = torch.norm(H - H_buffer) / torch.norm(H)
        if self.tol > 0 and H_diff <= self.tol:
            if self.verbose:
                print(f"Converged (small H change) at step {step}/{n_steps}")
            return True

        # Early stopping heuristic due to lack of improvement on smoothed
        # cost function
        if self._ewa_cost_min is None or self._ewa_cost < self._ewa_cost_min:
            self._no_improvement = 0
            self._ewa_cost_min = self._ewa_cost
        else:
            self._no_improvement += 1

        if (
            self.max_no_improvement is not None
            and self._no_improvement >= self.max_no_improvement
        ):
            if self.verbose:
                print(
                    "Converged (lack of improvement in objective function) "
                    f"at step {step}/{n_steps}"
                )
            return True

        return False

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None, W=None, H=None):
        """Learn a NMF model for the data X and returns the transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed.

        y : Ignored
            Not used, present here for API consistency by convention.

        W : array-like of shape (n_samples, n_components), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `None`, uses the initialisation method specified in `init`.

        H : array-like of shape (n_components, n_features), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `None`, uses the initialisation method specified in `init`.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        X = self._validate_data(
            X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32], cast_to_ndarray=False
        )

        with config_context(assume_finite=True):
            W, H, n_iter, n_steps = self._fit_transform(X, W=W, H=H)

        self.reconstruction_err_ = _beta_divergence(
            X, W, H, self._beta_loss, square_root=True
        )

        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_iter_ = n_iter
        self.n_steps_ = n_steps

        return W

    def _fit_transform(self, X, W=None, H=None, update_H=True):
        """Learn a NMF model for the data X and returns the transformed data.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed.

        W : array-like of shape (n_samples, n_components), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `update_H=False`, it is initialised as an array of zeros, unless
            `solver='mu'`, then it is filled with values calculated by
            `np.sqrt(X.mean() / self._n_components)`.
            If `None`, uses the initialisation method specified in `init`.

        H : array-like of shape (n_components, n_features), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `update_H=False`, it is used as a constant, to solve for W only.
            If `None`, uses the initialisation method specified in `init`.

        update_H : bool, default=True
            If True, both W and H will be estimated from initial guesses,
            this corresponds to a call to the `fit_transform` method.
            If False, only W will be estimated, this corresponds to a call
            to the `transform` method.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.

        H : ndarray of shape (n_components, n_features)
            Factorization matrix, sometimes called 'dictionary'.

        n_iter : int
            Actual number of started iterations over the whole dataset.

        n_steps : int
            Number of mini-batches processed.
        """
        if not self.all_non_negative:
            check_non_negative(X, "MiniBatchNMF (input X)")
        self._check_params(X)

        assert self._beta_loss == 2
        # if X.min() == 0 and self._beta_loss <= 0:
        #     raise ValueError(
        #         "When beta_loss <= 0 and X contains zeros, "
        #         "the solver may diverge. Please add small values "
        #         "to X, or use a positive beta_loss."
        #     )

        n_samples = X.shape[0]

        # initialize or check W and H
        W, H = self._check_w_h(X, W, H, update_H, W_filepath=self.W_filepath)
        if self.W_filepath is None:
            W = torch.from_numpy(W)
        H = torch.from_numpy(H).to(self.device)
        # H_buffer = H.copy()
        H_buffer = H.clone()

        # Initialize auxiliary matrices
        # self._components_numerator = H.copy()
        # self._components_denominator = np.ones(H.shape, dtype=H.dtype)
        self._components_numerator = H.clone()
        self._components_denominator = torch.ones_like(H)

        # Attributes to monitor the convergence
        self._ewa_cost = None
        self._ewa_cost_min = None
        self._no_improvement = 0

        batches = gen_batches(n_samples, self._batch_size)
        batches = itertools.cycle(batches)
        n_steps_per_iter = int(np.ceil(n_samples / self._batch_size))
        n_steps = self.max_iter * n_steps_per_iter


        steps_iterator = trange(n_steps) if self.verbose else range(n_steps)
        for i, batch in zip(steps_iterator, batches):
            # batch_cost = self._minibatch_step(X[batch], W[batch], H, update_H)
            Xbatch = torch.from_numpy(np.array(X[batch])).to(self.device).float()
            if self.W_filepath is not None:
                Wbatch = torch.from_numpy(np.array(W[batch])).to(self.device)
            else:
                Wbatch = W[batch].to(self.device)
            batch_cost = self._minibatch_step(Xbatch, Wbatch, H, update_H)
            if self.W_filepath is not None:
                W[batch] = Wbatch.cpu().numpy()
            else:
                W[batch] = Wbatch.cpu()

            if update_H and self._minibatch_convergence(
                Xbatch, batch_cost, H, H_buffer, n_samples, i, n_steps, pbar=steps_iterator if self.verbose else None,
            ):
                break

            H_buffer[:] = H

        if self.fresh_restarts:
            raise NotImplementedError
            W = self._solve_W(X, H, self._transform_max_iter)

        n_steps = i + 1
        n_iter = int(np.ceil(n_steps / n_steps_per_iter))

        if n_iter == self.max_iter and self.tol > 0:
            warnings.warn(
                (
                    f"Maximum number of iterations {self.max_iter} reached. "
                    "Increase it to improve convergence."
                ),
                ConvergenceWarning,
            )

        # return W, H, n_iter, n_steps
        if self.W_filepath is not None:
            return W, H.cpu().numpy(), n_iter, n_steps
        else:
            return W.cpu().numpy(), H.cpu().numpy(), n_iter, n_steps

    def transform(self, X):
        """Transform the data X according to the fitted MiniBatchNMF model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be transformed by the model.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self)
        X = self._validate_data(
            X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32], reset=False
        )

        W = self._solve_W(X, self.components_, self._transform_max_iter)

        return W

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None, W=None, H=None):
        """Update the model using the data in `X` as a mini-batch.

        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.

        This is especially useful when the whole dataset is too big to fit in
        memory at once (see :ref:`scaling_strategies`).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed.

        y : Ignored
            Not used, present here for API consistency by convention.

        W : array-like of shape (n_samples, n_components), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            Only used for the first call to `partial_fit`.

        H : array-like of shape (n_components, n_features), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            Only used for the first call to `partial_fit`.

        Returns
        -------
        self
            Returns the instance itself.
        """
        has_components = hasattr(self, "components_")

        X = self._validate_data(
            X,
            accept_sparse=("csr", "csc"),
            dtype=[np.float64, np.float32],
            reset=not has_components,
        )

        if not has_components:
            # This instance has not been fitted yet (fit or partial_fit)
            self._check_params(X)
            _, H = self._check_w_h(X, W=W, H=H, update_H=True)

            self._components_numerator = H.copy()
            self._components_denominator = np.ones(H.shape, dtype=H.dtype)
            self.n_steps_ = 0
        else:
            H = self.components_

        self._minibatch_step(X, None, H, update_H=True)

        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_steps_ += 1

        return self
