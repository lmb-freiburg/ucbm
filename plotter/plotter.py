from typing import Literal
import sys
sys.path.append('..') # append parent directory to import

from typing import Optional
from os import path, makedirs
from matplotlib import pyplot as plt
import numpy as np
from math import ceil
from matplotlib import font_manager
from mpl_toolkits.axes_grid1 import ImageGrid
import json
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from ucbm import UCBM
import torch
import seaborn as sns
import pandas as pd
from random import randrange
import math
from tqdm import trange, tqdm


def indices_tensor(targets, class_id: int, equal: bool):
    if equal:
        return (torch.Tensor(targets) == class_id).nonzero().reshape(-1)
    else:
        return (torch.Tensor(targets) != class_id).nonzero().reshape(-1)


class Plotter: 
    '''
    Class that is implementing all plotting tasks for CRAFT-CBMs. 

    Parameter
    ---------
    dpath: str
        The path all results should be saved. 
    '''

    def __init__(self, dpath: str):
        # Check if the given path exists. 
        makedirs(dpath, exist_ok=True)

        result_path = path.join(dpath, "results")
        makedirs(result_path, exist_ok=True)
        
        concept_data_path = path.join(dpath, "concept_data")
        makedirs(concept_data_path, exist_ok=True)

        concept_bank_path = path.join(dpath, "concept_banks")
        makedirs(concept_bank_path, exist_ok=True)
        
        classifier_path = path.join(dpath, "classifier")
        makedirs(classifier_path, exist_ok=True)
        
        self._result_path = result_path
        self._concept_data_path = concept_data_path
        self._concept_bank_path = concept_bank_path
        self._classifier_path = classifier_path

        font_dir = ['plotter/']
        for font in font_manager.findSystemFonts(font_dir):
            font_manager.fontManager.addfont(font)
    
    def get_result_path(self) -> str:
        return self._result_path
    
    def get_concept_data_path(self) -> str:
        return self._concept_data_path
    
    def get_concept_bank_path(self) -> str:
        return self._concept_bank_path
    
    def get_classifier_path(self) -> str:
        return self._classifier_path
    
    def print_concepts_to_file(self, concept_discovery, crops, crops_u, imp, 
                               class_name: str, concepts_to_print: int = -1, 
                               start_concept = 0, print_importances = False):
        '''
        Print the concepts including their importances to a file. 

        Parameter
        ---------
        concept_discovery: CRAFTConceptDiscovery
            Model which was used to compute concepts. 
        crops: np.ndarray
            Image crops the concepts were learned from. 
        crops_u: np.ndarray
            The image crops in the concept basis. 
        imp: list[float]
            The importances of each concept for the class. 
        all_classes: list[int]
            The data classes the concepts were calculated from. 
        class_name: str
            The name used to save the images. 
        concepts_to_print: int, optional
            Amount of concepts that are going to be print in a file. 
        start_concept: int, optional
            Index to start the concept enumeration with. 
        print_importances: bool, optional
            If True, the importances of the given vectors for the current
            class will be printed to a file. 
        '''

        # Load default value (all concepts). 
        if concepts_to_print == -1:
            concepts_to_print = concept_discovery._concepts_per_class
        
        # Check if amount of concepts to print is valid value. 
        if concepts_to_print <= 0 or \
            concepts_to_print > concept_discovery._concepts_per_class:
            raise AttributeError('The parameter concepts_to_print is smaler ' + 
                                 'or equal to zero or larger than the amount ' + 
                                 'concepts calculated by CRAFT. ')
        
        # Plot the importances. 
        if print_importances:
            plt.clf()
            plt.bar(range(len(imp)), imp)
            plt.xticks(range(len(imp)))
            plt.title("Concept Importance")
            plt.savefig(path.join(self._concept_data_path, 
                                  f"Class_{class_name}_Concept_Importance.jpg"))
            plt.close()

        # Sort the concept by importances. 
        most_important_concepts = \
            np.argsort(imp)[::-1][:concepts_to_print]

        # Function to show an image. 
        def show(img, **kwargs):
            img = np.array(img)
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)

            img -= img.min();img /= img.max()
            plt.imshow(img, **kwargs); plt.axis('off')
        
        # Print each concept into an image. 
        nb_crops = 9
        for c_id in most_important_concepts:
            best_crops_ids = np.argsort(crops_u[:, c_id])[::-1][:nb_crops]
            best_crops = crops[best_crops_ids]

            plt.clf()
            plt.figure(figsize=(5, 5))
            plt.axis('off')
            # plt.title("Concept " + str(c_id + start_concept))
            # plt.title("Concept " + str(c_id + start_concept) + 
            #           " has an importance value of " + str(importances[c_id]))
            width = int(nb_crops ** 0.5)
            height = ceil(nb_crops / width)
            for i in range(nb_crops):
                plt.subplot(width, height, i+1)
                show(best_crops[i])
            plt.savefig(path.join(self._concept_data_path, 
                                  f"Concept_{(c_id + start_concept):03}_-_Class_{class_name}.jpg"))
            plt.close()

    def print_concepts_to_file_new(self, dataset, crops_u, patch_size: int, n_patches: int, strides: float, save_dir: str, nb_crops: int = 9, nb_crops_json: int = 100, verbose: bool = False):
        # Function to show an image. 
        def show(ax, img, **kwargs):
            img = np.array(img)
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)

            img -= img.min();img /= img.max()
            ax.imshow(img, **kwargs); ax.axis('off')
        
        # Print each concept into an image. 
        assert nb_crops > 0
        sqrt_nb_crops = int(math.sqrt(nb_crops))
        iterator = trange(crops_u.shape[1], leave=False) if verbose else range(crops_u.shape[1])
        fill = ceil(math.log10(crops_u.shape[1]))
        crops_ids_res = {}
        for c_id in iterator:
            fig = plt.figure()
            grid = ImageGrid(fig, 111, nrows_ncols=(ceil(nb_crops/sqrt_nb_crops), sqrt_nb_crops), axes_pad=0.1)

            argsorted_crops_ids = np.argsort(crops_u[:, c_id])[::-1]
            crops_ids_res[c_id] = list(map(int, argsorted_crops_ids[:min(nb_crops_json, len(argsorted_crops_ids))]))
            best_crops_ids = argsorted_crops_ids[:nb_crops]
            img_ids = best_crops_ids // n_patches
            patch_ids = best_crops_ids % n_patches + np.array(list(range(nb_crops)))*n_patches
            all_images = torch.stack([dataset[img_id][0] for img_id in img_ids], dim=0)
            patches = torch.nn.functional.unfold(all_images, kernel_size=patch_size, stride=strides)
            patches = patches.transpose(1, 2).contiguous().view(-1, 3, patch_size, patch_size)
            best_crops = patches[patch_ids]

            for ax, i in zip(grid, range(nb_crops)):
                show(ax, best_crops[i])
            
            plt.savefig(path.join(save_dir, f"Concept_{str(c_id).zfill(fill)}.jpg"))
            plt.close()

        with open(path.join(save_dir, "crops_ids.json"), "w") as f:
            json.dump(crops_ids_res, f, indent=2)
    
    def plot_images_with_concept_similarities_to_file(
            self, pictures: np.ndarray, similarities: np.ndarray, 
            names: Optional[list[str]] = None, concepts_to_show: int = 10, 
            concept_labels: Optional[list[str]] = None):
        '''
        Plot images with the correponding similarities to file. 

        Parameter
        ---------
        pictures: np.ndarray
            Pictures in shape (N, C, W, H). 
        similarities: np.ndarray
            Similarities in shape (N, #concepts)
        names: list[str], optional
            Names corresponding to pictures. 
        concepts_to_show: int, optional
            Amount of best concepts to show. 
        concept_labels: list[str], optional
            A list of the concept labels. 
        '''
        
        # Check given inputs. 
        assert names is None or len(names) == pictures.shape[0], \
            'Name list and amount of pictures has to be equal. '
        assert similarities.shape[0] == pictures.shape[0], \
            'Amount of pictures in pictures and similarities must be equal. '
        
        if names is None:
            names = [f'Image_{i}' for i in range(pictures.shape[0])]

        # Function to show an image. 
        def show(img, ax, **kwargs):
            img = np.array(img)
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)

            img -= img.min();img /= img.max()
            ax.imshow(img, **kwargs); ax.axis('off')

        # print images. 
        n = len(names)
        for i in range(n):
            # Set up plot. 
            plt.clf()
            plt.rcParams.update({'font.size': 16})
            plt.rcParams.update({'font.family': 'CMU Serif'})
            fig, ax = plt.subplots(1, 3, width_ratios=[1, 2, 2])
            fig.set_figwidth(12)
            fig.set_figheight(6)
            # fig.suptitle(names[i])
            ax[1].axis('off')

            # Extract simularites for image i and sort them by simularity. 
            sim = similarities[i,:]
            sim_sorted = np.sort(sim)[::-1]
            sim_argsorted = np.argsort(sim)[::-1]

            # Show the image itself. 
            show(pictures[i,:,:,:], ax[0])
            
            # Plot the bars shwoing the concept simularities. 
            y_pos = np.arange(concepts_to_show)
            labels = [(f'Concept {i}' 
                      if concept_labels is None else 
                      f'{concept_labels[i]} ({i})')# + f': {sim[i]:.2f}%'
                      for i in sim_argsorted[:concepts_to_show]]
            bar_con = ax[-1].barh(y_pos, sim_sorted[:concepts_to_show], 
                                 color="#00376d", align="center")
            ax[-1].bar_label(bar_con, 
                             labels=[str(round(sim, 4)) 
                                     for sim in sim_sorted[:concepts_to_show]], 
                             label_type='center', color='white')
            ax[-1].set_yticks(y_pos, labels=labels)
            ax[-1].set_xlabel('concept simularity')
            ax[-1].invert_yaxis()

            # Save the plot to file. 
            plt.savefig(path.join(self._result_path, 
                                  f"Concept_Sim_{names[i]}.jpg"), dpi=300)
            plt.close()
    
    def plot_concept_violin_plot(self, 
                                 cons: list[int], 
                                 class_id: int, 
                                 class_label: str, 
                                 similarities: np.ndarray, 
                                 targets: np.ndarray, 
                                 concept_labels: Optional[list[str]] = None,
                                 importances: Optional[list[str]] = None,
                                 tmp: str = ""):
        '''
        Plot violin plots of the given concept numbers for the class class_id. 

        Parameters
        ----------
        cons: list[int]
            List of the concept numbers the violin plot should be done for. 
        class_id: int
            The class each concept is corresponding to. 
        class_label: str
        similarities: np.ndarray
            The concept similarities of n images in a matrix of shape 
            (n, #concepts). 
        targets: np.ndarray
            The targets of the images in a vector of shape (n). 
        concept_labels = Optional[list[str]] = None
            The concept labels. 
        importances: Optional[list[str]] = None
            The importance score of each concept
        tmp: str = ""
            String that is also inputted into the file name. 
        '''

        similarities = torch.stack(
            [similarities[i] for i in range(len(similarities))])

        # Function that returns the indexes of a sequence, that are label
        # with the given class_id. 
        def indices_tensor(targets, class_id, eq=True):
            if eq:
                return (torch.Tensor(targets) == class_id).nonzero().reshape(-1)
            else:
                return (torch.Tensor(targets) != class_id).nonzero().reshape(-1)

        sim_class = dict()
        sim_non_class = dict()
        for con in cons:
            sim_class[con] = \
                similarities[indices_tensor(targets, class_id), con].tolist()
            sim_non_class[con] = \
                similarities[indices_tensor(targets, class_id, eq=False), 
                             con].tolist()
        
        imp_dict = {con: "" for con in cons}
        if importances is not None:
            imp_dict = {con: f"{importances[i]:.4f}" 
                        for i, con in enumerate(cons)}

        df = pd.DataFrame({
            "concept simularity": sum((sim_class[con] + sim_non_class[con] 
                                      for con in cons), []), 
            "belonging": sum((["class data"] * len(sim_class[con]) + 
                                ["non data class"] * len(sim_non_class[con]) 
                                for con in cons), []), 
            "concept": sum(([
                (f"Concept {con}" 
                if concept_labels is None 
                else f"{concept_labels[con]} ({con})") + f" {imp_dict[con]}"
            ] * (len(sim_class[con]) + len(sim_non_class[con])) 
                           for con in cons), [])})
        
        plt.clf()
        fig, ax = plt.subplots(1, 2)
        ax[0].axis('off')
        sns.violinplot(data=df, x="concept simularity", y="concept", hue="belonging", 
                       split=True, inner="quart", cut=0, ax=ax[-1])
        
        fig.suptitle(f"class {class_label} - concept similarity")
        plt.savefig(path.join(self._result_path, 
                              f"class_{class_id}_{tmp}_concept_violin_plot.jpg"), 
                    dpi=300)
        plt.close()

    def plot_classifier_weights(self, 
                                coeff: np.ndarray, 
                                best_coeff_to_plot: int, 
                                class_to_idx: dict[str, int], 
                                concept_labels: Optional[list[str]] = None, 
                                tmp: str = ""):
        '''
        Plot a blot bar which shows the coeff of the linear classifier for the
        given class. 

        Parameters
        ----------
        coeff: np.ndarray
            The coefficients for the each class for every concept. 
            Shape (#concepts, #classes). 
        best_coeff_to_plot: int
            That much best coefficients can be seen as a single bar in the plot. 
            The others are combined to rest. 
        class_to_idx: dict[str, int]
            Provides a dictionary with all class names and class ids
        concept_labels: Optional[list[str]] = None
            The labels of each concept. 
        tmp: str = ""
            String that is also inputted into the file name. 
        '''

        for cls_name, cls_id in class_to_idx.items():
            abs_coeff = np.abs(coeff[:,cls_id])
            co_argsorted = np.argsort(abs_coeff)[::-1]
            co_argsorted_used = co_argsorted[:best_coeff_to_plot]
            co_argsorted_unused = co_argsorted[best_coeff_to_plot:]
            co_sorted_used = abs_coeff[co_argsorted_used]
            co_inv = coeff[:,cls_id] < 0
            co_sorted_unused = abs_coeff[co_argsorted_unused]
            co_sorted = np.append(co_sorted_used, np.sum(co_sorted_unused))
            co_argsorted = [f"{'NOT ' if co_inv[i] else ''}Concept {i}" 
                            if concept_labels is None else 
                            f"{'NOT ' if co_inv[i] else ''}{concept_labels[i]} ({i})"
                            for i in co_argsorted_used] + ["others"]
        
            plt.clf()
            fig, ax = plt.subplots(1, 2)
            ax[0].axis('off')
            y_pos = np.arange(best_coeff_to_plot+1)
            bar_con = ax[-1].barh(y_pos, co_sorted, color="#00376d", align="center")
            ax[-1].bar_label(bar_con, 
                                labels=[str(round(sim, 4)) 
                                        for sim in co_sorted], 
                                label_type='center', color='white')
            ax[-1].set_yticks(y_pos, labels=co_argsorted)
            ax[-1].set_xlabel('weight')
            ax[-1].invert_yaxis()
            fig.suptitle(f"Classifier weights of class {cls_name}")
            plt.savefig(path.join(self._result_path, 
                                f"class_{cls_id}_{tmp}_weights.jpg"), 
                        dpi=300)
            plt.close()
    
    @torch.no_grad()
    def plot_example_pictures(self, 
                              dataset: ImageFolder, 
                              ph_cbm: UCBM, 
                              img_per_class: int, 
                              concepts_to_show: int, 
                              tmp: str = ""):
        
        # Function to show an image. 
        def show(img, ax, **kwargs):
            img = np.array(img)
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)

            img -= img.min();img /= img.max()
            ax.imshow(img, **kwargs); ax.axis('off')
        
        concept_labels = ph_cbm._concept_labels
        weights = ph_cbm.get_classifier_weights().cpu()
        for j, cls_id in tqdm(enumerate(dataset.class_to_idx.values()), leave=False, total=len(dataset.class_to_idx)):
            indices = []
            idx_to_cls = {v: k for k, v in dataset.class_to_idx.items()}
            for _ in trange(img_per_class, leave=False):
                r = randrange(len(dataset))
                while dataset.targets[r] != cls_id or r in indices:
                    r = randrange(len(dataset))
                indices.append(r)
        
            subset = Subset(dataset, indices)
            subset = torch.stack([subset[i][0] for i in range(len(subset))], dim=0)

            preds, con_acts = ph_cbm.predict(subset)
            for i in range(subset.shape[0]):
                con_acts[i,:] = con_acts[i,:] * weights[torch.argmax(preds[i]).item(),:]
        
            for i in range(len(indices)):
                plt.clf()
                pred = preds[i]
                con_act = con_acts[i,:]
                con_act_asorted = torch.argsort(torch.abs(con_act), descending=True)
                con_act_sorted = torch.abs(con_act[con_act_asorted])

                plt.rcParams.update({'font.size': 16})
                plt.rcParams.update({'font.family': 'CMU Serif'})
                fig, ax = plt.subplots(1, 3, width_ratios=[1, 1, 2])
                fig.set_figwidth(12)
                fig.set_figheight(6)
                pred_s = torch.argmax(pred).item()
                fac = torch.nn.functional.softmax(pred, dim=0)[pred_s].item()
                fig.suptitle(f"{idx_to_cls[cls_id]}; \n"
                            f"Pred: {idx_to_cls[pred_s]}, {100*fac:.2f}%")
                ax[1].axis('off')

                # Show the image itself. 
                show(subset[i,:,:,:], ax[0])
                
                # Plot the bars shwoing the concept simularities. 
                y_pos = np.arange(concepts_to_show+1)
                labels = [(("NOT " if con_act[j] < 0 else "") + 
                        (f'Concept {j}' 
                        if concept_labels is None else 
                        f'{concept_labels[j]} ({j})'))
                        for j in con_act_asorted[:concepts_to_show]] + \
                            [f"{con_act_asorted.numel() - concepts_to_show} others"]
                cons = torch.cat((con_act_sorted[:concepts_to_show], 
                                torch.sum(con_act_sorted[concepts_to_show:]).unsqueeze(0)), 
                                dim=0)

                bar_con = ax[-1].barh(y_pos, cons, color="#00376d", align="center")
                ax[-1].bar_label(bar_con, 
                                labels=[str(round(float(sim.item()), 4)) for sim in cons], 
                                label_type='center', color='white')
                ax[-1].set_yticks(y_pos, labels=labels)
                ax[-1].set_xlabel('concept simularity * class-concept weight')
                ax[-1].invert_yaxis()

                # Save the plot to file. 
                plt.savefig(path.join(self._result_path, 
                                    f"Example_{tmp}_{i+j*img_per_class}_{idx_to_cls[cls_id]}.jpg"), dpi=300)
                plt.close()