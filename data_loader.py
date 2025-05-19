from typing import Optional, Callable
from constants import DATA_DIR

from torchvision.datasets import ImageFolder, Places365
from datasets.imagenet import ImageNet
from datasets.cub import CUB2011


def load_data(dataset_name: str, 
              train: bool, 
              transform: Optional[Callable] = None, 
              target_transform: Optional[Callable] = None, 
              reorder_idcs: bool = False,
              for_concept_extraction: bool = False) -> ImageFolder:
    '''
    Load a data set. 

    Parameters
    ----------
    dataset_name: str
    train: bool
    transform: Optional[Callable] = None
    target_transform: Optional[Callable] = None
    reorder_idcs: bool = False
        If imagenette or imagenet100 is load the class ids are shiftet to 0 - n. 
    '''

    if dataset_name == 'imagenet':
        return ImageNet(DATA_DIR['imagenet'], train, transform=transform, 
                        target_transform=target_transform)
    

    elif dataset_name == 'imagenette':
        return ImageNet(DATA_DIR['imagenet'], train, transform=transform, 
                        target_transform=target_transform, 
                        class_idcs=[0, 217, 482, 491, 497, 
                                    566, 569, 571, 574, 701], 
                        reorder_idcs=reorder_idcs)
    

    elif dataset_name == 'imagenet10':
        return ImageNet(DATA_DIR['imagenet'], train, transform=transform, 
                        target_transform=target_transform, 
                        class_idcs=list(range(10)))
    

    elif dataset_name == 'imagenet20':
        return ImageNet(DATA_DIR['imagenet'], train, transform=transform, 
                        target_transform=target_transform, 
                        class_idcs=list(range(20)))
    

    elif dataset_name == 'imagenet100':
        return ImageNet(DATA_DIR['imagenet'], train, transform=transform, 
                        target_transform=target_transform, 
                        class_idcs=[0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 
                                    14, 16, 18, 19, 20, 21, 22, 24, 26, 28, 
                                    29, 32, 33, 34, 35, 36, 38, 39, 41, 42, 
                                    46, 48, 50, 52, 54, 55, 56, 57, 59, 60, 
                                    61, 64, 65, 66, 67, 68, 70, 71, 72, 73, 
                                    74, 75, 76, 77, 78, 80, 81, 83, 84, 88, 
                                    89, 90, 91, 92, 93, 94, 96, 97, 99, 100, 
                                    104, 106, 107, 108, 110, 
                                    111, 112, 113, 115, 116, 
                                    117, 118, 119, 123, 124, 
                                    125, 127, 129, 130, 133, 
                                    135, 137, 138, 140, 141, 
                                    143, 144, 146, 150, 517], 
                        reorder_idcs=reorder_idcs)
    

    elif dataset_name == 'cub':
        return CUB2011(DATA_DIR['cub'], train, transform=transform, 
                       target_transform=target_transform)
    
    elif dataset_name == 'places365':
        try:
            data = Places365(DATA_DIR['places365'], 
                            'train-standard' if train else 'val', 
                            small=True, download=True, transform=transform, 
                            target_transform=target_transform)
        except RuntimeError:
            data =  Places365(DATA_DIR['places365'], 
                            'train-standard' if train else 'val', 
                            small=True, download=False, transform=transform, 
                            target_transform=target_transform)
        
        # Convert the labels, such that leading /x/ is removed and '/'
        # are replaced by '-'. 
        def conv_label(x: str):
            x = x[3:].replace('/', '-')
            return x

        data.classes = [conv_label(c) for c in data.classes]
        data.class_to_idx = {
            conv_label(c):i for c, i in data.class_to_idx.items()}

        return data
    
    
    raise AttributeError(f"Dataset {dataset_name} is not provided. ")