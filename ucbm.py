from typing import Literal, Union, Optional, Callable
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from copy import deepcopy

from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from concept_auxiliaries.concept_auxiliaries import raw_concept_sims, PDataset
from torcheval.metrics.functional import multilabel_accuracy, \
    multiclass_accuracy, multiclass_auprc, multilabel_auprc, binary_auroc, \
    multiclass_auroc, binary_auprc


################################################################################
#                                                                              #
#                                  LeakyReLU                                   #
#         ReLU-similar function, which is not killing the gradient.            #
#                                                                              #
################################################################################
class LeakyReLU(torch.autograd.Function):
    @staticmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.neg = x < 0
        return x.clamp(min=0.0)
    
    @staticmethod
    def backward(self, grad_output: torch.Tensor):
        grad_input = grad_output.clone()
        grad_input[self.neg] *= 0.1
        return grad_input



################################################################################
#                                                                              #
#                                   JumpReLU                                   #
#                 JumpReLU with straight-through estimator.                    #
#                Following: https://arxiv.org/abs/2407.14435                   #
#                                                                              #
################################################################################
class RectangleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input

def rectangle(x: torch.Tensor) -> torch.Tensor:
    return RectangleFunction.apply(x)

class _JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                x: torch.Tensor, 
                threshold: torch.Tensor, 
                bandwidth: float) -> torch.Tensor:
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return x * (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor): # ste
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = (x > threshold).to(x.dtype) * output_grad
        rect_value = rectangle((x - threshold) / bandwidth)
        threshold_grad = -(threshold / bandwidth) * rect_value * output_grad
        return x_grad, threshold_grad, None

class JumpReLU(nn.Module):
    def __init__(self, 
                 num_concepts: int, 
                 threshold_init: Optional[torch.Tensor] = None, 
                 bandwidth: float = 1e-3):
        super(JumpReLU, self).__init__()
        self.log_threshold = nn.Parameter(-10*torch.ones(num_concepts, 
                                                         requires_grad=True))
        if threshold_init is not None:
            assert threshold_init.numel() == num_concepts, \
                f"Init threshold is of dimension {threshold_init.size()}" + \
                f", but should be of dimension {num_concepts}. "
            self.log_threshold = threshold_init
        self.bandwidth = bandwidth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _JumpReLU.apply(
            x, 
            self.log_threshold.exp(),  # exp ensures positive threshold
            self.bandwidth
            )



################################################################################
#                                                                              #
#                                    L0-Loss                                   #
#           L0-similar function which is not killing the gradient.             #
#                                                                              #
################################################################################
class _StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                x: torch.Tensor, 
                threshold: torch.Tensor, 
                bandwidth: float) -> torch.Tensor:
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor): # ste
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = torch.zeros_like(x)
        rect_value = rectangle((x - threshold) / bandwidth)
        threshold_grad = -(1.0 / bandwidth) * rect_value * output_grad
        return x_grad, threshold_grad, None

class StepFunction(nn.Module):
    def __init__(self):
        super(StepFunction, self).__init__()

    def forward(self, 
                x: torch.Tensor, 
                threshold: torch.Tensor, 
                bandwidth: float) -> torch.Tensor:
        return _StepFunction.apply(x, threshold, bandwidth)

step_function = StepFunction()
def l0_loss(x: torch.Tensor, 
            threshold: torch.Tensor, 
            bandwidth: float) -> torch.Tensor:
    out = step_function(x, threshold, bandwidth)
    return torch.sum(out, dim=-1).sum()


def l0_approx(x: torch.Tensor, 
              threshold: torch.Tensor, 
              a: int = 20) -> float:
    out = 1 / (1 + torch.exp(-a*(x - threshold)))
    return out.sum().item()



################################################################################
#                                                                              #
#                                 Elastic-Loss                                 #
#                                                                              #
################################################################################
def elastic_loss(weight_or_act: torch.Tensor, 
                 alpha: float = 0.99) -> torch.Tensor:
    l1 = weight_or_act.norm(p=1)
    l2 = (weight_or_act**2).sum()
    return 0.5 * (1 - alpha) * l2 + alpha * l1

def elastic_loss_weights(weight: torch.Tensor, 
                         alpha: float = 0.99) -> torch.Tensor:
    l1 = weight.norm(p=1)
    l2 = (weight**2).sum()
    return 0.5 * (1 - alpha) * l2 + alpha * l1

def elastic_loss_activations(act: torch.Tensor, 
                             alpha: float = 0.99) -> torch.Tensor:
    l1 = act.norm(p=1, dim=-1)
    l2 = (act**2).sum(dim=-1)
    return torch.sum(0.5 * (1 - alpha) * l2 + alpha * l1)


################################################################################
#                                                                              #
#                                 TopK-Module                                  #
#                  Module that keeps only k largest values.                    #
#      Implementation from  https://github.com/openai/sparse_autoencoder       #
################################################################################
class TopK(nn.Module):
    def __init__(self, 
                 k: int, 
                 postact_fn: Callable = nn.ReLU()):
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        # make all other values 0
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict.update(
            {prefix + "k": self.k, 
             prefix + "postact_fn": self.postact_fn.__class__.__name__})
        return state_dict

    @classmethod
    def from_state_dict(cls, 
                        state_dict: dict[str, torch.Tensor], 
                        strict: bool = True) -> "TopK":
        k = state_dict["k"]
        postact_fn = ACTIVATIONS_CLASSES[state_dict["postact_fn"]]()
        return cls(k=k, postact_fn=postact_fn)

ACTIVATIONS_CLASSES = {
    "ReLU": nn.ReLU,
    "Identity": nn.Identity,
    "TopK": TopK,
}




################################################################################
#                                                                              #
#                                  Classifier                                  #
#             Interpretable classifier from concepts to classes.               #
#                                                                              #
################################################################################
class Classifier(nn.Module):
    def __init__(self, 
                 num_concepts: int, 
                 num_classes: int, 
                 relu: Literal["no", "ReLU", "jumpReLU"] = "ReLU", 
                 scale: Literal["learn", "no"] = "no", 
                 bias: Literal["learn", "no"] = "no", 
                 dropout_p: float = 0, 
                 k: int = -1, 
                 jumpReLU_threshold_init: Optional[torch.Tensor] = None):
        '''
        Interpretable (linear) classifier from concept space to 
        classification space. 

        num_concepts: int
            number of concepts in concept space
        num_classes: int
            number of classes in classification space
        relu: Literal["no", "ReLU", "jumpReLU"] = "ReLU"
            relu function to use after gating
        scale: Literal["learn", "no"] = "no"
            scale to scale the cosine similarities, 
            set "learn" to learn scale and "no" to disable
        bias: Literal["learn", "no"] = "no"
            bias to substract from scaled cosine similarities 
            set "learn" to learn bias and "no" to disable
        dropout_p: float = 0
            dropout rate from concept dropout
        k: int = -1
            If k >= 0 TopK is used instead of gating, else nothing happens
        jumpReLU_threshold_init: Optional[torch.Tensor] = None
            Init value of jumpReLU
        '''
        super().__init__()
        self.relu = relu
        if relu == "jumpReLU":
            self._jumpReLU = JumpReLU(num_concepts, jumpReLU_threshold_init)
        self.scale_method = scale
        self.bias_method = bias
        self.dropout = (dropout_p > 0)
        self.dropout_layer = nn.Dropout(p=dropout_p)
        self.k = k
        if self.k >= 0:
            self.top_k = TopK(self.k)
        if self.scale_method == "learn":
            self.log_scaling = nn.Parameter(
                torch.zeros(num_concepts, requires_grad=True))
        if self.bias_method == "learn":
            self.log_offset = nn.Parameter(
                -10*torch.ones(num_concepts, requires_grad=True))
        self.linear = nn.Linear(num_concepts, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # input-dependent concept selection
        if self.scale_method == "learn":
            x = self.log_scaling.exp() * x  # exp ensures positive scale
        
        if self.bias_method == "learn":
            x = x - self.log_offset.exp()  # exp ensures positive bias
        
        if self.relu == "ReLU":
            gated = F.relu(x)
        elif self.relu == "jumpReLU":
            gated = self._jumpReLU(x)
        elif self.k >= 0:
            gated = self.top_k(x)
        elif self.relu == "no":
            gated = x
        else:
            raise NotImplementedError
        
        # concept dropout
        if self.dropout:
            mask = torch.ones_like(gated)
            mask = self.dropout_layer(mask)
            gated = gated * mask
            x = x * mask
            
        # sparse linear layer
        out = self.linear(gated)
        return out, gated, x


################################################################################
#                                                                              #
#                                     UCBM                                     #
#                    unsupervised concept bottleneck model                     #
#                                                                              #
################################################################################
class UCBM:
    '''
    Class implementing an unsupervised concept bottleneck model. 

    Parameters
    ----------
    backbone
        model backbone to compute the embeddings
    h: torch.Tensor | np.ndarray
        matrix of shape (n, p) containing all n concept activation vectors
    batch_size: int
        batch size to use
    epochs: int
        amount of training epochs
    lam_gate: float
        regularization strength of penalization on gate output
    lam_w: float
        regularization strength of penalization on linear weights
    dropout_p: float
        concept dropout rate
    learning_rate: float
        learning rate
    relu: Literal["no", "ReLU", "jumpReLU"]
        relu to use for gating
    scale_mode: Literal['learn', 'no']
        scale for concept similarities
    bias_mode: Literal['learn', 'no']
        bias for concept similarities
    normalize: bool
        normalize the cosine similarities of each concept? 
    k: int
        k to use for TopK module, if -1 no TopK module is used
    device: Literal['cuda', 'cpu']
        device to use for computations
    '''

    def __init__(self, 
                 backbone, 
                 h: Union[torch.Tensor, np.ndarray], 
                 batch_size: int, 
                 epochs: int,
                 lam_gate: float, 
                 lam_w: float, 
                 dropout_p: float, 
                 learning_rate: float, 
                 relu: Literal["no", "ReLU", "jumpReLU"], 
                 scale_mode: Literal['learn', 'no'], 
                 bias_mode: Literal['learn', 'no'], 
                 normalize: bool, 
                 k: int,
                 device: Literal['cuda', 'cpu']):
        
        self._backbone = backbone
        if not torch.is_tensor(h):
            h = torch.tensor(h)
        self._num_concepts = h.shape[0]
        self._h = h.to(device)
        self._h = self._h / torch.norm(self._h, dim=1, keepdim=True)
        self._batch_size = batch_size
        self._lr = learning_rate
        self._device = device

        self._epochs = epochs
        self._lam_gate = lam_gate
        self._lam_w = lam_w
        self._dropout_p = dropout_p
        self._relu = relu
        self._scale_mode = scale_mode
        self._bias_mode = bias_mode
        self._normalize = normalize
        self._k = k
    
    @torch.no_grad()
    def _get_concept_embeddings(self, 
                                dataset: Dataset, 
                                saved_activation_path: Optional[str] = None, 
                                data_label: Optional[str] = None, 
                                normalize=False, 
                                mean=None, 
                                std=None) \
                                    -> Dataset[torch.Tensor]:
        '''
        Compute and save the concept embeddings using the cosine similarity or 
        load them from the given information. 

        Parameters
        ----------
        dataset: Dataset
            dataset to compute concept embeddings on
        saved_activation_path: Optional[str] = None
            folder where the activations of the current dataset and
            backbone can be/are saved
        data_label: Optional[str] = None
            data_label (train or test) for the dataset to identify the
            correct pre_computed concept similarities
        
        Returns
        -------
        sims: Dataset[torch.Tensor]
            Dataset containing the concept similarities of each image in 
            given dataset (in the same order) 
        '''

        return raw_concept_sims(self._h, 
                                dataset, 
                                self._backbone, 
                                self._batch_size, 
                                self._device, 
                                saved_activation_path, 
                                data_label, 
                                normalize=normalize, 
                                mean=mean, 
                                std=std)
    
    def fit(self, 
            training_set: ImageFolder, 
            saved_activation_path: str, 
            test_set: Optional[ImageFolder] = None, 
            verbose: bool = True, 
            cocostuff_training: bool = False):
        '''
        Function fit UCBM to given dataset. 

        Parameters
        ----------
        training_set: ImageFolder
            data_set to train UCBM on
        saved_activation_path: str
            path to save concept activations
        test_set: Optional[ImageFolder] = None
            If given, test accuracy is printed to stdout each epoch
        verbose: bool = True
            print progress to to stdout? 
        '''

        # Load the concept activations. 
        embeddings = self._get_concept_embeddings(
            training_set, 
            saved_activation_path, 
            "train", 
            normalize=self._normalize, 
            mean=None, 
            std=None)
        self._mean = embeddings.mean if self._normalize else None
        self._std = embeddings.std if self._normalize else None

        if verbose:
            print("Loaded concept activations of training dataset...")

        # Function that returns the indexes of a sequence, that are label
        # with the given class_id. 
        def indices_tensor(targets, class_id):
            return (torch.Tensor(targets) == class_id).nonzero().reshape(-1)

        # Load scale and bias values. 
        self._num_classes = len(training_set.classes)
        self._multilabel = isinstance(training_set.targets[0], list)
        
        # Load the model. 
        self._classifier = Classifier(
            self._num_concepts, 
            self._num_classes, 
            self._relu, 
            self._scale_mode, 
            self._bias_mode, 
            self._dropout_p, 
            self._k)
        self._classifier = self._classifier.to(self._device)

        # Load stuff required for training the model. 
        loss_fn = nn.BCEWithLogitsLoss() if self._multilabel else nn.CrossEntropyLoss()
        optimizer = optim.Adam(self._classifier.parameters(), lr=self._lr)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self._epochs)
        
        # Load data
        dset = PDataset(embeddings, training_set.targets)
        if not cocostuff_training:
            data_loader = DataLoader(dset, self._batch_size, shuffle=True, 
                                    num_workers=8)
            
            # Train the model
            for i in trange(self._epochs, leave=False):
                self._classifier.train()
                corr, n_samples = 0, 0
                if self._relu == "jumpReLU":
                    print(self._classifier._jumpReLU.log_threshold.exp().mean())
                    print(self._classifier._jumpReLU.log_threshold.mean())
                for X_batch, y_batch in tqdm(data_loader, leave=False):
                    y_batch = y_batch.to(self._device)
                    out = X_batch.to(self._device)
                    
                    # Train the model. 
                    optimizer.zero_grad()
                    y_pred, after_gate, before_gate = self._classifier(out)

                    # task-specific loss
                    if self._multilabel:
                        loss = loss_fn(y_pred, y_batch.to(y_pred.dtype))
                    else:
                        loss = loss_fn(y_pred, y_batch)

                    # gate sparsity penalty
                    if self._lam_gate != 0:
                        if self._relu != "jumpReLU":
                            loss += self._lam_gate * elastic_loss_activations(after_gate)
                        else:
                            loss += self._lam_gate * l0_loss(before_gate, self._classifier._jumpReLU.log_threshold.exp(), self._classifier._jumpReLU.bandwidth)
                    
                    # weight sparsity penalty
                    if self._lam_w != 0:
                        loss += self._lam_w * elastic_loss_weights(self._classifier.linear.weight)

                    loss.backward()
                    optimizer.step()

                    # Compute training accuracy
                    if self._multilabel:
                        corr += y_pred.shape[0] * multilabel_accuracy(
                            torch.sigmoid(y_pred), y_batch, criteria="hamming")
                    else:
                        corr += y_pred.shape[0] * multiclass_accuracy(
                            torch.argmax(y_pred, dim=1), y_batch)
                    n_samples += y_pred.shape[0]
                
                lr_scheduler.step()
                if test_set:
                    self._classifier.eval()
                    test_acc = self.get_evaluation_metric(
                        test_set, saved_activation_path=saved_activation_path, data_label="test", metric=["acc"])["acc"]
                if verbose:
                    print(f"Epoch {i+1} / {self._epochs} - " + 
                        f"train acc: {100 * corr / n_samples:.2f}%" + 
                        (f", test acc: {100 * test_acc:.2f}%" if test_set else ""))
        else:
            # Train the model
            for i in trange(self._epochs, leave=False):
                self._classifier.train()
                corr = 0

                target_classes = deepcopy(list(training_set.class_to_idx.values()))
                np.random.shuffle(target_classes)
                all_subsets = []
                for target_cls in target_classes:
                    optimizer.zero_grad()
                    tar = torch.tensor(training_set.targets)[:,target_cls]
                    pos_idcs = indices_tensor(tar, 1).detach().numpy()
                    neg_idcs = indices_tensor(tar, 0).detach().numpy()
                    n = self._batch_size // 2
                    pos_samples = np.random.choice(pos_idcs, n, replace=len(pos_idcs) <= n)
                    neg_samples = np.random.choice(neg_idcs, n, replace=len(neg_idcs) <= n)

                    subset_dset = Subset(dset, pos_samples.tolist() + neg_samples.tolist())
                    assert len(subset_dset) == self._batch_size
                    all_subsets.append(subset_dset)
                data_loader = DataLoader(torch.utils.data.ConcatDataset(all_subsets), batch_size=self._batch_size, shuffle=False, num_workers=8)

                for idx, (X_batch, y_batch) in tqdm(enumerate(data_loader), leave=False, total=len(target_classes)):
                    y_batch = y_batch.to(self._device)
                    out = X_batch.to(self._device)
                    y_pred, gate, _ = self._classifier(out)

                    # loss = loss_fn(y_pred, y_batch.to(y_pred.dtype))
                    loss = loss_fn(y_pred[:, target_classes[idx]], y_batch.to(y_pred.dtype)[:, target_classes[idx]]) # compute loss only on the target class output logit
                    if self._lam_gate != 0:
                        loss = loss + self._lam_gate * elastic_loss_activations(gate)
                    if self._lam_w != 0:
                        loss = loss + self._lam_w * elastic_loss_weights(self._classifier.linear.weight)

                    loss.backward()
                    optimizer.step()

                    corr += multilabel_accuracy(torch.sigmoid(y_pred), y_batch, criteria="hamming")
                
                lr_scheduler.step()
                if test_set:
                    self._classifier.eval()
                    test_acc = self.get_evaluation_metric(
                        test_set, saved_activation_path=saved_activation_path, data_label="test", metric=["acc"])["acc"]
                if verbose:
                    print(f"Epoch {i+1} / {self._epochs} - " + 
                        f"train acc: {100 * corr / len(training_set.classes):.2f}%" + 
                        (f", test acc: {100 * test_acc:.2f}%" if test_set else ""))


    
    @torch.no_grad()
    def predict(self, 
                imgs: torch.Tensor) \
                    -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Function prediction the img classes and concept of imgs. 

        Paramters
        ---------
        imgs: torch.Tensor
            Input images in shape (n, #channel, width, height). 
        
        Returns
        -------
        y_pred: torch.Tensor
            Tensor containing the predicted classes. Shape (n). 
        concept values:
            Tensor of shape (n, #concepts) 
        '''

        assert hasattr(self, "_classifier"), "Model not yet fitted. "

        self._classifier.eval()

        out = self._backbone(imgs.to(self._device))

        if len(out.shape) == 4:
            out = torch.mean(out, dim=(2, 3))
            
        out = out / torch.norm(out, dim=1, keepdim=True)
        out = out.type(self._h.dtype)

        out = torch.matmul(out, self._h.T)

        if self._normalize:
            out = (out - self._mean.to(self._device)) / self._std.to(self._device)

        out, gate, _ = self._classifier(out)

        if self._multilabel:
            out = torch.sigmoid(out)

        out, gate = out.cpu(), gate.cpu()
        
        return out, gate
    
    @torch.no_grad()
    def get_evaluation_metric(self, 
                              dataset: ImageFolder, 
                              metric: list[Literal["acc", "auprc", "auroc", "auprc_pc"]] = ["acc"], 
                              saved_activation_path: Optional[str] = None, 
                              data_label: Optional[str] = None) \
                                -> dict[str, float]:
        '''
        Function that computes the accuracy of this model to the given dataset. 

        Parameters
        ----------
        dataset: ImageFolder
            Dataset for which the accuracy should be computed for. 
        metric: Literal["acc", "auprc", "auroc"]
        saved_activation_path: Optional[str] = None
            If not None, this is the path where the concept similarities 
            can be/are saved. 
        data_label: Optional[str] = None
            Label (train or test) from given dataset. 
        
        Returns
        -------
        accuracy: list[Optional[float]]
        '''

        if next(self._classifier.parameters()).device != self._device:  
            self._classifier.to(self._device)
        self._classifier.eval()

        # Load the concept activations. 
        embeddings = self._get_concept_embeddings(
            dataset, saved_activation_path, data_label, 
            self._normalize, self._mean, self._std)
        
        dset = PDataset(embeddings, dataset.targets)
        data_loader = DataLoader(dset, batch_size=self._batch_size, 
                                 shuffle=False, num_workers=8)
        y_pred = []
        y_true = []
        for X_batch, y_batch in data_loader:
            y_predb, _, _ = self._classifier(X_batch.to(self._device))
            if self._multilabel:
                y_predb = torch.sigmoid(y_predb)
            y_predb = y_predb.cpu()
            y_pred.append(y_predb)
            y_true.append(y_batch)
        
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)

        def indices_tensor(targets, class_id):
            return (torch.Tensor(targets) == class_id).nonzero().reshape(-1)

        metrics = {}
        for me in metric:
            if me == "acc":
                if self._multilabel:
                    metrics[me] = multilabel_accuracy(y_pred, y_true, criteria="hamming").item()
                else:
                    metrics[me] = multiclass_accuracy(y_pred, y_true).item()
            elif me == "auprc" and self._multilabel:
                if self._multilabel:
                    # metrics[me] = multilabel_auprc(y_pred, y_true).item()
                    n = 500
                    auprc_pc = 0
                    for cls in dataset.class_to_idx.values():
                        tar = torch.tensor(dataset.targets)[:,cls]
                        pos_idcs = indices_tensor(tar, 1).detach().numpy()
                        neg_idcs = indices_tensor(tar, 0).detach().numpy()
                        pos_samples = np.random.choice(pos_idcs, n//2, len(pos_idcs) < n//2)
                        neg_samples = np.random.choice(neg_idcs, n//2, len(neg_idcs) < n//2)

                        samples = pos_samples.tolist() + neg_samples.tolist()

                        auprc_pc += binary_auprc(y_pred[samples, cls], y_true[samples, cls]).item()
                    auprc_pc /= len(dataset.class_to_idx)
                    metrics[me] = auprc_pc

                else:
                    metrics[me] = multiclass_auprc(y_pred, y_true).item()
            elif me == "auprc_pc" and self._multilabel:
                n = 500
                auprc_pc = []
                for cls in dataset.class_to_idx.values():
                    tar = torch.tensor(dataset.targets)[:,cls]
                    pos_idcs = indices_tensor(tar, 1).detach().numpy()
                    neg_idcs = indices_tensor(tar, 0).detach().numpy()
                    pos_samples = np.random.choice(pos_idcs, n//2, len(pos_idcs) < n//2)
                    neg_samples = np.random.choice(neg_idcs, n//2, len(neg_idcs) < n//2)

                    samples = pos_samples.tolist() + neg_samples.tolist()

                    auprc_pc.append(binary_auprc(y_pred[samples, cls], y_true[samples, cls]).item())
                
                metrics[me] = auprc_pc
            elif me == "auroc":
                if self._multilabel:
                    auroc = 0
                    n = y_pred.shape[1]
                    for i in range(n):
                        auroc += binary_auroc(y_pred[:,i], y_true[:,i]).item()
                    metrics[me] = auroc / n
                else:
                    if len(y_true.unique()) == 2:
                        # metrics[me] = binary_auroc(torch.argmax(y_pred, dim=1), y_true).item()
                        metrics[me] = roc_auc_score(y_true.numpy(), softmax(y_pred.numpy(), axis=1)[:, 1])
                    else:
                        metrics[me] = multiclass_auroc(y_pred, y_true, num_classes=len(dataset.classes)).item()
        return metrics
    
    @torch.no_grad()
    def compute_concept_similarities(self, 
                                     dataset: Dataset, 
                                     saved_activation_path: str, 
                                     data_label: str) -> torch.Tensor:
        '''
        Get concept simularities for given dataset. 

        Parameters
        ----------
        dataset: Dataset
            The dataset for which the concept similarities should be 
            computed for. 
        saved_activation_path: str
            Path where the concept similarities can be/are saved. 
        data_label: str
            Label (train or test) from given dataset. 

        Returns
        -------
        concept_similarities: torch.Tensor
            The concept similarities in shape (n, #concepts). 
        '''

        self._classifier.eval()

        # Load the concept activations. 
        embeddings = self._get_concept_embeddings(
            dataset, saved_activation_path, data_label, 
            self._normalize, self._mean, self._std)
        
        data_loader = DataLoader(embeddings, batch_size=self._batch_size, 
                                 shuffle=False, num_workers=8)
        sims = []
        for con in data_loader:
            _, sim, _ = self._classifier(con.to(self._device))
            sim = sim.cpu()
            sims.append(sim)
        sims = torch.cat(sims, dim=0)

        return sims

    @torch.no_grad()
    def avg_non_zero_concept_ratio(self, 
                                   dataset: Dataset, 
                                   saved_activation_path: str, 
                                   data_label: str) -> torch.Tensor:
        '''
        Get the average amount of non zero values in the concept bottleneck. 

        Parameters
        ----------
        dataset: Dataset
            The dataset for which the concept similarities should be 
            computed for. 
        saved_activation_path: str
            Path where the concept similarities can be/are saved. 
        data_label: str
            Label (train or test) from given dataset. 

        Returns
        -------
        non_zero_ratio: float
        '''

        self._classifier.eval()

        # Load the concept activations. 
        embeddings = self._get_concept_embeddings(
            dataset, saved_activation_path, data_label, 
            self._normalize, self._mean, self._std)
        
        data_loader = DataLoader(embeddings, batch_size=self._batch_size, 
                                 shuffle=False, num_workers=8)
        sum = 0
        for con in data_loader:
            _, sim, _ = self._classifier(con.to(self._device))
            sum += float(torch.count_nonzero(sim).cpu() / sim.shape[1])

        return sum / len(embeddings)
    
    def save_to_file(self, filepath: str, filename: str):
        '''
        Saves the classifier of the model into a file. 

        Parameters
        ----------
        filepath: str
            Filepath to save the model. 
        filename: str
            Filename for the file. 
        '''

        def get_backbone():
            try:
                return self._backbone.cpu()
            except AttributeError:
                return None

        path = os.path.join(filepath, filename)
        torch.save(
            {
                "model_state_dict": self._classifier.state_dict(),
                "backbone": get_backbone(),
                "epochs": self._epochs,
                "batch_size": self._batch_size,
                "lam_gate": self._lam_gate,
                "lam_w": self._lam_w,
                "dropout_p": self._dropout_p, 
                "num_concepts": self._num_concepts, 
                "num_classes": self._num_classes, 
                "w": self._h.detach().cpu(), 
                "learning_rate": self._lr, 
                "relu": self._relu, 
                "scale_mode": self._scale_mode, 
                "bias_mode": self._bias_mode, 
                "multilabel": self._multilabel, 
                "normalize": self._normalize, 
                "mean": self._mean, 
                "std": self._std, 
                "k": self._k
            }, path)
    
    @classmethod
    def load_from_file(cls, 
                       filepath: str, 
                       filename: str, 
                       device: Literal["cuda", "cpu"] = "cuda", 
                       backbone_p = None):
        '''
        Load the classifier of the model from a file. 

        Parameters
        ----------
        filepath: str
            Filepath to save the model. 
        filename: str
            Filename for the file. 
        device: Literal["cuda", "cpu"]
        backbone_p = None
            The backbone if backbone is function (can't be saved). 
        '''

        path = os.path.join(filepath, filename)
        data: dict = torch.load(path)
        if data["backbone"] is not None:
            backbone = data["backbone"].to(device)
        elif backbone_p is not None:
            backbone = backbone_p
        else:
            raise AttributeError()
        h = data["w"].to(device)
        num_concepts = data["num_concepts"]
        num_classes = data["num_classes"]
        lam_gate = data["lam_gate"]
        lam_w = data["lam_w"]
        batch_size = data["batch_size"]
        learning_rate = data["learning_rate"]
        epochs = data["epochs"]
        relu = data.get("relu", "ReLU")
        scale_mode = data["scale_mode"]
        bias_mode = data["bias_mode"]
        scale = data.get("scale", None)
        bias = data.get("bias", None)
        if torch.is_tensor(scale) and \
            torch.allclose(scale, torch.ones(num_concepts).to(scale.device)):
            scale_mode = "no"
        elif scale is None:
            pass
        else:
            raise NotImplementedError
        if torch.is_tensor(bias) and \
            torch.allclose(bias, torch.zeros(num_concepts).to(bias.device)):
            bias_mode = "no"
        elif bias is None:
            pass
        else:
            raise NotImplementedError
        dropout_p = data["dropout_p"]
        multilabel = data.get("multilabel", False)
        normalize = data.get("normalize", False)
        mean = data.get("mean", None)
        std = data.get("std", None)
        k = data.get("k", -1)
        if scale_mode == "no" and bias_mode == "no" and k == -1 and "relu" not in data:
            relu = "no"

        classifier = Classifier(
            num_concepts, 
            num_classes, 
            relu, 
            scale_mode, 
            bias_mode, 
            dropout_p, 
            k
        )
        if "top_k.k" in data["model_state_dict"]:
            del data["model_state_dict"]["top_k.k"]
        classifier.load_state_dict(data["model_state_dict"])
        classifier = classifier.eval().to(device)

        ucbm = UCBM(
            backbone, 
            h, 
            batch_size, 
            epochs, 
            lam_gate, 
            lam_w, 
            dropout_p, 
            learning_rate, 
            relu, 
            scale_mode, 
            bias_mode, 
            normalize, 
            k, 
            device
        )
        ucbm._classifier = classifier
        ucbm._num_classes = num_classes
        ucbm._multilabel = multilabel
        ucbm._mean = mean
        ucbm._std = std
        return ucbm

    @torch.no_grad()
    def compute_confusion_matrix(self, 
                                 dataset: ImageFolder, 
                                 saved_activation_path: str, 
                                 data_label: str) \
        -> dict[int, dict[int, float]]:
        '''
        Function that computes the confusion matrix for this model. 

        Parameters
        ----------
        dataset: ImageFolder
            The dataset for which the concept similarities should be 
            computed for. 
        saved_activation_path: str
            Path where the concept similarities can be/are saved. 
        data_label: str
            Label (train or test) from given dataset. 
        
        Returns
        -------
        confusion_matrix: dict[int, dict[int, float]]
            class_id: {other_class: percentage of class_id images
                                    mapped on other class}
        '''

        assert not self._multilabel

        self._classifier.eval()

        # Load the concept activations. 
        embeddings = self._get_concept_embeddings(
            dataset, saved_activation_path, data_label, 
            self._normalize, self._mean, self._std)
        
        classes = dataset.class_to_idx.values()
        confusion_matrix = {c1: {c2: 0 for c2 in classes} for c1 in classes}
        
        dset = PDataset(embeddings, dataset.targets)
        data_loader = DataLoader(dset, batch_size=self._batch_size, 
                                 shuffle=False, num_workers=8)
        
        for X_batch, y_batch in data_loader:
            y_pred, _, _ = self._classifier(X_batch.to(self._device))
            y_pred = y_pred.cpu()
            for i in range(y_pred.numel()):
                confusion_matrix[y_batch[i]][y_pred[i]] += 1
        
        all = len(embeddings)
        confusion_matrix = {c: {k: v / all for k, v in cv.items()} 
                              for c, cv in confusion_matrix.items()}
        
        return confusion_matrix
    
    def get_classifier_weights(self) -> torch.Tensor:
        '''
        Get weights of the classifier. 

        Returns
        -------
        weights: torch.Tensor
            Shape (#concepts, #classes)
        '''

        return self._classifier.linear.weight
    
    def get_classifier_bias(self) -> torch.Tensor:
        '''
        Get bias of the classifier. 

        Returns
        -------
        bias: torch.Tensor
            Shape (#classes)
        '''

        return self._classifier.linear.bias
    
    def get_info_dict(self, 
                      training_data: ImageFolder, 
                      test_data: ImageFolder, 
                      saved_activation_path: str,
                      metrics = ["acc", "auprc", "auprc_pc", "auroc"]) -> dict:
        '''
        Get the most important information abou this dict in a dictionary. 

        Parameters
        ----------
        training_data: ImageFolder
        test_data: ImageFolder
            Data used to compute test accuracy, ...
        saved_activation_path: str
        '''

        data = dict()
        data["amount of concepts"] = int(self._h.shape[0])
        data["amount of classes"] = len(test_data.classes)
        train_res = self.get_evaluation_metric(
            training_data, metrics, saved_activation_path, "train")
        test_res = self.get_evaluation_metric(
            test_data, metrics, saved_activation_path, "test")
        if "acc" in metrics and "acc" in train_res:
            data["train acc"] = train_res["acc"]
            data["test acc"] = test_res["acc"]
        if "auprc" in metrics and "auprc" in train_res:
            data["train auprc"] = train_res["auprc"]
            data["test auprc"] = test_res["auprc"]
        if "auprc_pc" in metrics and "auprc_pc" in train_res:
            data["train auprc_pc"] = train_res["auprc_pc"]
            data["test auprc_pc"] = test_res["auprc_pc"] 
        if "auroc" in metrics and "auroc" in train_res:
            data["train auroc"] = train_res["auroc"]
            data["test auroc"] = test_res["auroc"]

        data["avg non zero concept ratio"] = \
            self.avg_non_zero_concept_ratio(
                test_data, saved_activation_path, "test")
        data["learning rate"] = self._lr
        data["lambda gate"] = self._lam_gate
        data["lambda w"] = self._lam_w
        data["epochs"] = self._epochs
        data["dropout p"] = self._dropout_p
        data["scale mode"] = self._scale_mode
        data["bias mode"] = self._bias_mode
        data["multilabel"] = self._multilabel
        data["normalize"] = self._normalize
        data["k"] = self._k
    
        return data