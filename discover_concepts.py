import torch
from os import path

from data_loader import load_data
from model_loader import get_model
from concept_auxiliaries.craft_concept_discovery import CRAFTConceptDiscovery
from plotter.plotter import Plotter
import argparse
from constants import DATA_SETS, MODELS, RESULT_PATH
import numpy as np
import json
import random
from tqdm import trange, tqdm
from concept_auxiliaries.concept_auxiliaries import *


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load models.
    model = get_model(args.backbone, args.device)
    print(f"Model {args.backbone} loaded successfully...")

    training_data = load_data(args.dataset, True, model.transform, for_concept_extraction=True)

    # subsample places & imagenet
    if args.dataset == "imagenet":
        # subsample imagenet to 10% per class
        if path.isfile("utils/imagenet_sample_ids.json"):
            all_sample_ids = json.load(open("utils/imagenet_sample_ids.json"))
            subsampled_samples = [training_data.samples[i] for i in tqdm(all_sample_ids, leave=False)]
        else:
            subsample_ratio = 0.1

            class_to_count = [0]*len(training_data.classes)
            cur_target = 0
            for _, target in tqdm(training_data.samples, leave=False):
                assert cur_target <= target
                class_to_count[target] += 1
                cur_target = target
            
            all_sample_ids, subsampled_samples = [], []
            for cls_idx in trange(len(training_data.classes)):
                prev_samples = 0 if cls_idx == 0 else np.cumsum(class_to_count[:cls_idx])[-1]
                for idx in random.sample(list(range(class_to_count[cls_idx])), k=int(subsample_ratio*class_to_count[cls_idx])):
                    sample_idx = prev_samples + idx
                    subsampled_samples.append(training_data.samples[sample_idx])

            with open("utils/imagenet_sample_ids.json", "w") as f:
                json.dump(all_sample_ids, f, indent=2)
        
        training_data.samples = subsampled_samples
        training_data.targets = [s[1] for s in subsampled_samples]
    elif args.dataset == "places365":
        # subsample places365 to 10% per class
        if path.isfile("utils/places365_sample_ids.json"):
            all_sample_ids = json.load(open("utils/places365_sample_ids.json"))
            subsampled_samples = [training_data.imgs[i] for i in tqdm(all_sample_ids, leave=False)]
        else:
            subsample_ratio = 0.1

            class_to_count = [0]*len(training_data.classes)
            cur_target = 0
            for _, target in tqdm(training_data.imgs, leave=False):
                assert cur_target <= target
                class_to_count[target] += 1
                cur_target = target
            
            all_sample_ids, subsampled_samples = [], []
            for cls_idx in trange(len(training_data.classes)):
                prev_samples = 0 if cls_idx == 0 else np.cumsum(class_to_count[:cls_idx])[-1]
                for idx in random.sample(list(range(class_to_count[cls_idx])), k=int(subsample_ratio*class_to_count[cls_idx])):
                    sample_idx = prev_samples + idx
                    subsampled_samples.append(training_data.imgs[sample_idx])
                    all_sample_ids.append(int(sample_idx))

            with open("utils/places365_sample_ids.json", "w") as f:
                json.dump(all_sample_ids, f, indent=2)
        
        training_data.imgs = subsampled_samples
        training_data.targets = [s[1] for s in subsampled_samples]

    print(f"Dataset {args.dataset} loaded successfully with {len(training_data)} images...")

    # Load plotter object.
    plotter = Plotter(path.join(RESULT_PATH, f'{args.dataset}-{args.backbone}/'))
    con_data_path = plotter.get_concept_data_path()
    print("Plotter loaded successfully...")

    n_concept = int(args.con_am*len(training_data.class_to_idx))
    # Load concept discovery.
    concept_dis = CRAFTConceptDiscovery(
        training_data, model, 
        # args.con_am, 
        n_concept,
        args.patch_size, 
        args.batch_size,
        args.device)
    print("CRAFT Concept Discovery loaded successfully...")

    # Get all CRAFT concepts.
    if args.regenerate or not path.exists(path.join(con_data_path, concept_dis._con_bank_name, f"h.npy")):
        h = concept_dis.compute_concepts_for_all(plotter)
    print("Computed concepts successfully...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Settings for unsupervised concept generation'
        )

    parser.add_argument("-d", "--dataset", type=str, default="imagenet",
                        help="The dataset to load",
                        choices=DATA_SETS)
    parser.add_argument("-b", "--backbone", type=str, default="resnet50_v2",
                        help="Which pretrained model backbone to use",
                        choices=MODELS)


    parser.add_argument("--device", type=str,
                        default=("cuda" if torch.cuda.is_available() else "cpu"),
                        help="Which device to use", choices=["cpu", "cuda"])


    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size used by concept discovery")
    parser.add_argument("--patch_size", type=int, default=64,
                        help="Patch size used by concept discovery")
    parser.add_argument("--con_am", type=float, default=10,
                        help="Amount of computed concepts per class")


    parser.add_argument("--regenerate", action="store_true", 
                        help="Regeneration of concept vectors " +
                            "(even if already computed)")

    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    main(args)