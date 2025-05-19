from typing import Optional, Literal, Union, Callable
from os import path, listdir, makedirs

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import math
import tempfile
import shutil


# Dataset that is loads two datasets in parallel. 
class PDataset(Dataset):
    def __init__(self, *its: list[Union[Dataset, torch.Tensor, list]], list_to_tensor: bool = True):
        assert len(its) > 0, "At least one sequence must be given"
        assert all(self.get_length(its[0]) == self.get_length(it) for it in its), \
            "Provided iterables needs to be from same size. "
        self.its = its
        self.list_to_tensor = list_to_tensor
    
    def get_length(self, it):
        if isinstance(it, Dataset):
            return len(it)
        elif torch.is_tensor(it):
            return it.shape[0]
        else:
            return len(it)
    
    def __len__(self):
        return self.get_length(self.its[0])
    
    def __getitem__(self, idx):
        res = []
        for i in range(len(self.its)):
            it = self.its[i][idx]
            if isinstance(it, list) and self.list_to_tensor:
                it = torch.tensor(it)
            res.append(it)
        return res

# Class implementing a dataset, that is loading single tensors from files
# in a given directory in alphabetic order. 
class MemTensorDataset(Dataset):
    def __init__(self, root: str, normalize=False, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None):
        self.root = root
        self.normalize = normalize
        if root[-1] == "/":
            root = root[:-1]
        last_dir = root.split("/")[-1]
        self.samples = [f for f in listdir(root) if f.endswith(".pth") and "activations" in f]
        def key(x: str) -> int:
            x = x.removeprefix(last_dir + "_")
            x = x.removesuffix(".pth")
            return int(x)
        self.samples.sort(key=key)
        if self.normalize:
            if mean is None or std is None:
                if not path.exists(path.join(root, "mean.pt")) or not path.exists(path.join(root, "std.pt")):
                    batch_size = 64
                    size = torch.load(path.join(self.root, self.samples[0])).size()
                    mean = torch.zeros(size)
                    meansq = torch.zeros(size)
                    for i in trange(math.ceil(len(self) / batch_size), leave=False):
                        start = i * batch_size
                        end = (i+1) * batch_size
                        if end > len(self):
                            end = len(self)
                        batch = torch.stack(
                            [torch.load(path.join(self.root, sam)) 
                            for sam in self.samples[start:end]])
                        mean += torch.sum(batch, dim=0)
                        meansq += torch.sum(batch**2, dim=0)
                    mean = mean / len(self)
                    meansq = meansq / len(self)
                    std = torch.sqrt(meansq - mean**2)
                    if torch.any(torch.isnan(std)):
                        diff = meansq - mean**2
                        std = torch.sqrt(torch.clamp(diff, min=diff[diff>0].min()))
                    assert not torch.any(torch.isnan(mean))
                    torch.save(mean, path.join(root, "mean.pt"))
                    assert not torch.any(torch.isnan(std))
                    torch.save(std, path.join(root, "std.pt"))
                else:
                    mean = torch.load(path.join(path.join(root, "mean.pt")))
                    std = torch.load(path.join(path.join(root, "std.pt")))
            self.mean = mean
            self.std = std
            
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tens = torch.load(path.join(self.root, self.samples[idx]))
        if self.normalize:
            tens = (tens - self.mean)  / self.std
        return tens


@torch.no_grad()
def raw_concept_sims(h: Union[np.ndarray, torch.Tensor], 
                     dataset: Dataset, 
                     backbone: Union[nn.Module, Callable], 
                     batch_size: int, 
                     device: Literal['cuda', 'cpu'], 
                     saved_activation_path: Optional[str] = None, 
                     data_label: Optional[str] = None,
                     normalize=False, 
                     mean=None, 
                     std=None) \
                        -> Dataset[torch.Tensor]:
    '''
    Compute raw concept similarities using cosine similarity. 

    Parameters
    ----------
    h: np.ndarray
        Concept vectors in shape (#concepts, p). 
    dataset: Dataset
        The dataset to compute the concept similarities on. 
    backbone: nn.Module | Callable
        The model backbone. 
    batch_size: int
        The batch_size in which the concept similarities are computed. 
    device: Literal['cuda', 'cpu']
        The device to compute the similarities on. 
    saved_activation_path: Optional[str] = None
        The folder where the activations of the current dataset and
        backbone can be/are saved. 
    data_label: Optional[str] = None
        The data_label (train or test) for the dataset to identify the
        correct pre_computed concept similarities. 
    normalize: bool = False
        Wheter to normalize the concept similarities concept-wise. If mean and
        std is given, these values are used, otherwise mean and std will be
        computed on the given dataset. 
    mean: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None

    Returns
    -------
    sim: Dataset[torch.Tensor]
        Dataset containing the concept similarities of each image in 
        given dataset (in the same order). 
    '''

    assert ((saved_activation_path is None and data_label is None) or
            (saved_activation_path is not None and data_label is not None)), \
            "saved_activation_path and data_label must be both None or not None. "
    
    save = data_label is not None
    if not save:
        saved_activation_path = tempfile.gettempdir()
        data_label = "tmp"

    # Check if file with activations already exists. 
    # If so, read them from the file. 
    largest_index = None
    if save:
        save_name = f"saved_{data_label}_activations"
        saving_dir = path.join(
            saved_activation_path, save_name)
        if path.exists(saving_dir) and len([f for f in listdir(saving_dir) if "activations" in f]) == len(dataset):
            return MemTensorDataset(saving_dir, normalize=normalize, mean=mean, std=std)
        elif not path.exists(saving_dir):
            makedirs(saving_dir, exist_ok=True)
        else:
            if len(listdir(saving_dir)) == 0:
                largest_index = -1
            else:
                largest_index = max(int(file.split(".")[0].removeprefix(f"{save_name}_")) for file in listdir(saving_dir) if "activations" in file)
    else:
        save_name = f"saved_{data_label}_activations"
        saving_dir = path.join(
            saved_activation_path, save_name)
        if path.exists(saving_dir):
            shutil.rmtree(saving_dir)
        makedirs(saving_dir)
        largest_index = -1

    
    # Calculate the concept similarities. 

    # Convert CAVs to normalized tensor. 
    h = h if torch.is_tensor(h) else torch.tensor(h)
    h = h.to(device)
    h /= torch.norm(h, dim=1, keepdim=True)

    # Load the data from dataset. 
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # If activations are not saved to file, collect them in variable. 

    if isinstance(backbone, nn.Module):
        backbone.to(device)
        backbone.eval()
    
    # Compute concept similarities. 
    for i, (X_batch, _) in enumerate(tqdm(data_loader, leave=False)):
        if largest_index is not None and (i + 1) * batch_size - 1 < largest_index:
            continue
        out = backbone(X_batch.to(device))

        if len(out.shape) == 4:
            out = torch.mean(out, dim=(2, 3))
            
        out = out / torch.norm(out, dim=1, keepdim=True)
        out = out.type(h.dtype)

        out = torch.matmul(out, h.T)

        out = out.cpu()
        width = out.shape[0]
        for j in range(width):
            f_name = path.join(
                saving_dir, f"{save_name}_{i*batch_size+j}.pth")
            torch.save(torch.from_numpy(out[j,:].numpy()), f_name)

    return MemTensorDataset(saving_dir, normalize=normalize, mean=mean, std=std)