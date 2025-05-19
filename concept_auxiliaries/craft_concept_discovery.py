from typing import Literal
from os import path

from model_loader import Model
from plotter.plotter import Plotter
from torchvision.datasets import ImageFolder
from sklearn.preprocessing import normalize
import torch
import numpy as np
from Craft.craft.craft_torch import Craft
import os


class CRAFTConceptDiscovery:
    """
    Class implementing a concept discovery using CRAFT. 

    Parameters
    ----------
    dataset: ImageFolder
        The dataset used by CRAFT to compute the 
        concept activation vectors (CAVs). 
    model: Model
        The black-box model to gain the concepts from. 
    concepts_per_class: int
        The number of concepts per class. 
    patch_size: int
        The patch size of image crops used to gain the concepts from. 
    batch_size: int
        The batch size to use. 
    device: Literal['cpu', 'cuda']
        The device on which the computations are happening. 
    """

    def __init__(self, 
                 data_set: ImageFolder, 
                 model: Model, 
                 number_of_concepts: int, 
                 patch_size: int, 
                 batch_size: int, 
                 device: Literal['cuda', 'cpu']):
        # Load the given data set. 
        self._dataset = data_set

        # Load the end-to-end model. 
        self._model = model
        
        # self._concepts_per_class = number_of_concepts // len(data_set.class_to_idx)
        self._patch_size = patch_size
        self._batch_size = batch_size
        self._device = device

        self._con_bank_name = f"concepts_{number_of_concepts}_{self._patch_size}"

        # Initialize CRAFT-object. 
        self._craft = Craft(input_to_latent = self._model.g,
                            latent_to_logit = self._model.h,
                            number_of_concepts = number_of_concepts,
                            patch_size = self._patch_size,
                            batch_size = self._batch_size,
                            device = self._device)
    

    def compute_concepts_for_all(self, 
                         plotter: Plotter, 
                         verbose: bool=True) -> np.ndarray:
        '''
        Discover concepts and their CAVs for all classes from the dataset. 

        Parameters
        ----------
        plotter: Plotter
            Used to print best matching image crops and save and load data. 
        verbose: bool = True
            If True, the progress is printed to stdout.  

        Returns
        -------
        w: np.ndarray
            The concept vectors in shape (#concepts, p). 
        '''

        strides = int(self._craft.patch_size * 0.80)

        input_example = self._dataset.__getitem__(0)[0]
        patches = torch.nn.functional.unfold(input_example.unsqueeze(0), kernel_size=self._craft.patch_size, stride=strides)
        patches = patches.transpose(1, 2).contiguous().view(-1, 3, self._craft.patch_size, self._craft.patch_size)
        nof_patches = len(patches)

        # Compute concept activation vectors using CRAFT.
        W_filepath = None
        con_data_path = path.join(plotter.get_concept_data_path(), self._con_bank_name)
        print(con_data_path)
        if not path.exists(con_data_path):
            os.mkdir(con_data_path)
        crops_w, h = self._craft.fit_all(self._dataset, plotter.get_concept_data_path(), W_filepath=W_filepath, verbose=verbose)
        self._h = normalize(h, axis=1)
        np.save(path.join(con_data_path, f"h.npy"), self._h)

        plotter.print_concepts_to_file_new(self._dataset, crops_w, patch_size=self._craft.patch_size, n_patches=nof_patches, strides=strides, 
                                           save_dir=con_data_path, verbose=verbose)
        
        del crops_w
        if W_filepath is not None:
            os.remove(W_filepath)

        return self._h
    
    def get_importance_array(self) -> np.ndarray: 
        '''
        Get the importance array. 

        Returns
        -------
        importances: np.ndarray
            Two-dimensional matrix containing the importances. 
        '''

        # Convert dictionary to numpy array. 
        classes = self._dataset.class_to_idx.values()
        imp = [self._importances[k] for k in classes]
        
        # Convert list to numpy array. 
        imp = np.array(imp)

        return imp