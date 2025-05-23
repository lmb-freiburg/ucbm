
"""
CRAFT Module for Tensorflow
"""

from abc import ABC, abstractmethod
from math import ceil
from typing import Callable, Optional

import torch
import numpy as np
from sklearn.decomposition import NMF, MiniBatchNMF
from sklearn.exceptions import NotFittedError
import torch.utils
from tqdm import tqdm
import os

from nmf_gpu import MiniBatchNMF as MiniBatchNMF_GPU


def torch_to_numpy(tensor):
  try:
    return tensor.detach().cpu().numpy()
  except:
    return np.array(tensor)


def _batch_inference(model, dataset, batch_size=128, resize=None, device='cuda'):
  nb_batchs = ceil(len(dataset) / batch_size)
  start_ids = [i*batch_size for i in range(nb_batchs)]

  results = []

  with torch.no_grad():
    for i in start_ids:
      x = torch.tensor(dataset[i:i+batch_size])
      x = x.to(device)

      if resize:
        x = torch.nn.functional.interpolate(x, size=resize, mode='bilinear', align_corners=False)

      results.append(model(x).cpu())

  results = torch.cat(results)
  return results


class BaseConceptExtractor(ABC):
    """
    Base class for concept extraction models.

    Parameters
    ----------
    input_to_latent : Callable
        The first part of the model taking an input and returning
        positive activations, g(.) in the original paper.
    latent_to_logit : Callable
        The second part of the model taking activation and returning
        logits, h(.) in the original paper.
    number_of_concepts : int
        The number of concepts to extract.
    batch_size : int, optional
        The batch size to use during training and prediction. Default is 64.

    """

    def __init__(self, input_to_latent : Callable,
                       latent_to_logit : Optional[Callable] = None,
                       number_of_concepts: int = 20,
                       batch_size: int = 64):

        # sanity checks
        assert(number_of_concepts > 0), "number_of_concepts must be greater than 0"
        assert(batch_size > 0), "batch_size must be greater than 0"
        assert(callable(input_to_latent)), "input_to_latent must be a callable function"

        self.input_to_latent = input_to_latent
        self.latent_to_logit = latent_to_logit
        self.number_of_concepts = number_of_concepts
        self.batch_size = batch_size

    @abstractmethod
    def fit(self, inputs):
        """
        Fit the CAVs to the input data.

        Parameters
        ----------
        inputs : array-like
            The input data to fit the model on.

        Returns
        -------
        tuple
            A tuple containing the input data and the matrices (U, W) that factorize the data.

        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, inputs):
        """
        Transform the input data into a concepts embedding.

        Parameters
        ----------
        inputs : array-like
            The input data to transform.

        Returns
        -------
        array-like
            The transformed embedding of the input data.

        """
        raise NotImplementedError


class Craft(BaseConceptExtractor):
    """
    Class Implementing the CRAFT Concept Extraction Mechanism.

    Parameters
    ----------
    input_to_latent : Callable
        The first part of the model taking an input and returning
        positive activations, g(.) in the original paper.
    latent_to_logit : Callable, optional
        The second part of the model taking activation and returning
        logits, h(.) in the original paper.
    number_of_concepts : int
        The number of concepts to extract.
    batch_size : int, optional
        The batch size to use during training and prediction. Default is 64.
    patch_size : int, optional
        The size of the patches to extract from the input data. Default is 64.
    """

    def __init__(self, input_to_latent: Callable,
                       latent_to_logit: Optional[Callable] = None,
                       number_of_concepts: int = 20,
                       batch_size: int = 64,
                       patch_size: int = 64,
                       device : str = 'cuda'):
        super().__init__(input_to_latent, latent_to_logit, number_of_concepts, batch_size)

        self.patch_size = patch_size
        self.activation_shape = None
        self.device = device

    def fit(self, inputs: np.ndarray):
        """
        Fit the Craft model to the input data.

        Parameters
        ----------
        inputs : np.ndarray
            Preprocessed Iinput data of shape (n_samples, channels, height, width).
            (x1, x2, ..., xn) in the paper.

        Returns
        -------
        (X, W, H)
            A tuple containing the crops (X in the paper),
            the concepts values (W) and the concepts basis (H).
        """
        assert len(inputs.shape) == 4, "Input data must be of shape (n_samples, channels, height, width)."
        assert inputs.shape[2] == inputs.shape[3], "Input data must be square."

        image_size = inputs.shape[2]

        # extract patches from the input data, keep patches on cpu
        strides = int(self.patch_size * 0.80)

        patches = torch.nn.functional.unfold(inputs, kernel_size=self.patch_size, stride=strides)
        patches = patches.transpose(1, 2).contiguous().view(-1, 3, self.patch_size, self.patch_size)

        # encode the patches and obtain the activations
        activations = _batch_inference(self.input_to_latent, patches, self.batch_size, image_size, 
                                       device=self.device)

        assert torch.min(activations) >= 0.0, "Activations must be positive."

        # if the activations have shape (n_samples, height, width, n_channels),
        # apply average pooling
        if len(activations.shape) == 4:
            activations = torch.mean(activations, dim=(2, 3))

        # apply NMF to the activations to obtain matrices W and H
        reducer = NMF(n_components=self.number_of_concepts)
        W = reducer.fit_transform(torch_to_numpy(activations))
        H = reducer.components_.astype(np.float32)

        # store the factorizer and H as attributes of the Craft instance
        self.reducer = reducer
        self.H = np.array(H, dtype=np.float32)

        return patches, W, H
    
    @torch.no_grad()
    def fit_all(self, dataset: torch.utils.data.Dataset, save_dir, W_filepath="tmp.mmap", verbose: bool = False):
        """
        Fit the Craft model to the input data.

        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the model on.
        verbose: bool, optional

        Returns
        -------
        (X, U, W)
            A tuple containing the crops (X in the paper),
            the concepts values (U) and the concepts basis (W).
        """
        # extract patches from the input data, keep patches on cpu
        strides = int(self.patch_size * 0.80)

        input_example = dataset.__getitem__(0)[0]
        image_size = input_example.shape[2]
        patches = torch.nn.functional.unfold(input_example.unsqueeze(0), kernel_size=self.patch_size, stride=strides)
        patches = patches.transpose(1, 2).contiguous().view(-1, 3, self.patch_size, self.patch_size)
        nof_patches = len(patches)
        if image_size:
            patches = torch.nn.functional.interpolate(patches, size=image_size, mode='bilinear', align_corners=False)
        test_activations = self.input_to_latent(patches.to(self.device)).mean(dim=(2,3)).cpu()

        if not os.path.isfile(os.path.join(save_dir, "activations.mmap")):
            # encode the patches and obtain the activations
            activations_mmap = np.memmap(os.path.join(save_dir, "activations.mmap"), dtype=np.float32, mode='w+', shape=(nof_patches*len(dataset), test_activations.size(1)))

            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size // nof_patches, shuffle=False, num_workers=16) 
            iterator = tqdm(loader, leave=False, total=len(loader)) if verbose else loader
            lower_idx = 0
            for images, _ in iterator:
                patches = torch.nn.functional.unfold(images, kernel_size=self.patch_size, stride=strides)
                patches = patches.transpose(1, 2).contiguous().view(-1, 3, self.patch_size, self.patch_size)
                assert patches.size(0) == nof_patches*len(images)
                
                patches = patches.to(self.device)
                if image_size:
                    patches = torch.nn.functional.interpolate(patches, size=image_size, mode='bilinear', align_corners=False)
                # activations_mmap[batch_idx*self.batch_size:min(nof_patches*len(dataset), (batch_idx+1)*self.batch_size)] = self.input_to_latent(patches).mean(dim=(2,3)).cpu().numpy()
                activations_mmap[lower_idx:min(nof_patches*len(dataset), lower_idx+patches.size(0))] = self.input_to_latent(patches).mean(dim=(2,3)).cpu().numpy()
                lower_idx += patches.size(0)

        activations_mmap = np.memmap(os.path.join(save_dir, "activations.mmap"), dtype=np.float32, mode='r', shape=(nof_patches*len(dataset), test_activations.size(1)))
        
        # apply NMF to the activations to obtain matrices W and H
        # reducer = NMF(n_components=int(self.number_of_concepts*len(dataset.class_to_idx)), verbose=int(verbose))
        # reducer = MiniBatchNMF(n_components=self.number_of_concepts*len(dataset.class_to_idx), max_no_improvement=int(1e9), batch_size=int(2**13), verbose=int(verbose))
        # ca. 10x speed-up
        # init NNDSVD is not supported -> init="random"
        reducer = MiniBatchNMF_GPU(
           # n_components=self.number_of_concepts*len(dataset.class_to_idx), 
           n_components=self.number_of_concepts,
           max_no_improvement=int(1e9),
           tol=1e-6, 
           batch_size=int(2**13), 
           verbose=int(verbose), 
           device=self.device, 
           all_non_negative=True, 
           init="random", 
           W_filepath=W_filepath
        )
        W = reducer.fit_transform(activations_mmap)
        H = reducer.components_.astype(np.float32)

        # store the factorizer and H as attributes of the Craft instance
        self.reducer = reducer
        self.H = np.array(H, dtype=np.float32)

        # return patches, W, H
        return W, H

    def check_if_fitted(self):
        """Checks if the factorization model has been fitted to input data.

        Raises
        ------
        NotFittedError
            If the factorization model has not been fitted to input data.
        """

        if not hasattr(self, 'reducer'):
            raise NotFittedError("The factorization model has not been fitted to input data yet.")

    def transform(self, inputs: np.ndarray, activations: Optional[np.ndarray] = None):
        self.check_if_fitted()

        if activations is None:
            activations = _batch_inference(self.input_to_latent, inputs, self.batch_size,
                                           device=self.device)

        is_4d = len(activations.shape) == 4

        if is_4d:
            # (N, C, W, H) -> (N * W * H, C)
            activation_size = activations.shape[-1]
            activations = activations.permute(0, 2, 3, 1)
            activations = torch.reshape(activations, (-1, activations.shape[-1]))

        H_dtype = self.reducer.components_.dtype
        W = self.reducer.transform(torch_to_numpy(activations).astype(H_dtype))

        if is_4d:
          # (N * W * H, R) -> (N, W, H, R)
          W = np.reshape(W, (-1, activation_size, activation_size, W.shape[-1]))

        return W