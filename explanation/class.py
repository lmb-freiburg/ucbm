import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import seaborn as sns
import torch
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm, trange
from tueplots import bundles

from data_loader import load_data
from model_loader import get_model
from ucbm import UCBM
from constants import MODELS, DATA_SETS, RESULT_PATH
import json
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as pchs

# Function to show an image. 
def prepare_img(img):
    img = np.array(img)
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    img -= img.min();img /= img.max()
    return img

parser = argparse.ArgumentParser(description='Plot example classifications of individual examples')
parser.add_argument("-d", "--dataset", type=str, default="imagenet",
                    help="The dataset to load",
                    choices=DATA_SETS)
parser.add_argument("-b", "--backbone", type=str, default="resnet_v2",
                    help="Which pretrained model to use as backbone",
                    choices=MODELS)
parser.add_argument("-c", "--concept_bank", type=str, default="", 
                    help="Name of the concept bank to use")
parser.add_argument("--cbm_name", type=str,
                    help="Name of the post-hoc concept bottleneck model " + 
                    "that should be loaded. ")
parser.add_argument("--device", type=str,
                    default=("cuda" if torch.cuda.is_available() else "cpu"),
                    help="Which device to use", choices=["cpu", "cuda"])
parser.add_argument("--concept_amount", type=int, default=3)
parser.add_argument("--class_idcs", type=int, nargs="*")
parser.add_argument("--save_folder_name", type=str, default="")
parser.add_argument("--folder_name", type=str, default="")
args = parser.parse_args()

# Load models.
model = get_model(args.backbone, args.device)
print(f"Model {args.backbone} loaded successfully...")
training_data = load_data(args.dataset, True, model.transform)
training_d = load_data(args.dataset, True, model.transform)
dataset = load_data(args.dataset, False, 
                         model.transform, reorder_idcs=True)
if args.dataset == "imagenet":
    # subsample imagenet to 10% per class
    if os.path.isfile("utils/imagenet_sample_ids.json"):
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
        
        all_sample_ids = []
        subsampled_samples = []
        for cls_idx in trange(len(training_data.classes)):
            prev_samples = 0 if cls_idx == 0 else np.cumsum(class_to_count[:cls_idx])[-1]
            for idx in random.sample(list(range(class_to_count[cls_idx])), k=int(subsample_ratio*class_to_count[cls_idx])):
                sample_idx = prev_samples + idx
                subsampled_samples.append(training_data.samples[sample_idx])
                all_sample_ids.append(int(sample_idx))

        with open("utils/imagenet_sample_ids.json", "w") as f:
            json.dump(all_sample_ids, f, indent=2)
    
    training_data.samples = subsampled_samples
    training_data.targets = [s[1] for s in subsampled_samples]
elif args.dataset == "places365":
    # subsample places365 to 10% per class
    if os.path.isfile("utils/places365_sample_ids.json"):
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

        with open("utils/places365_sample_ids2.json", "w") as f:
            json.dump(all_sample_ids, f, indent=2)
    
    training_data.imgs = subsampled_samples
    training_data.targets = [s[1] for s in subsampled_samples]

classifier_path = os.path.join(
    RESULT_PATH,
    f'{args.dataset}-{args.backbone}' if args.folder_name == "" else args.folder_name,
    "classifier",
    args.concept_bank,
    args.cbm_name,
)
assert os.path.isdir(classifier_path)

concept_data_path = os.path.join(
    RESULT_PATH,
    f'{args.dataset}-{args.backbone}' if args.folder_name == "" else args.folder_name,
    "concept_data",
    args.concept_bank
)
assert os.path.isdir(concept_data_path)

ph_cbm = UCBM.load_from_file(
    classifier_path, 
    "classifier.pth", 
    args.device, 
    model.g
)

plot_folder = os.path.join(classifier_path, "per_example_plots" if args.save_folder_name == "" else args.save_folder_name)
os.makedirs(plot_folder, exist_ok=True)
zfill_examples = math.ceil(math.log10(len(dataset.targets)))
zfill_cls_id = math.ceil(math.log10(len(set(dataset.targets))))
zfill_con_id = math.ceil(math.log10(ph_cbm._num_concepts))

weights = ph_cbm.get_classifier_weights().cpu()
bias = ph_cbm.get_classifier_bias().cpu()
d_classes = dataset.class_to_idx.values() if args.class_idcs is None or args.class_idcs == [] else args.class_idcs

idx_to_class = {v: k for k, v in dataset.class_to_idx.items() if v in d_classes}

json_path = os.path.join(RESULT_PATH, f"{args.dataset}-{args.backbone}", "concept_data", args.concept_bank, "crops_ids.json")
assert os.path.isfile(json_path)
with open(json_path) as f:
    data = json.load(f)
    data = {int(a): b for a, b in data.items()}
print(f"Loaded crops_id data successfully...")

patch_size = int(args.concept_bank.split("_")[-1])
strides = int(patch_size * 0.80)
input_example = training_data.__getitem__(0)[0]
patches = torch.nn.functional.unfold(input_example.unsqueeze(0), kernel_size=patch_size, stride=strides)
patches = patches.transpose(1, 2).contiguous().view(-1, 3, patch_size, patch_size)
n_patches = len(patches)

for j, cls_id in tqdm(enumerate(d_classes), leave=False, total=len(d_classes)):
    def indices_tensor(targets, class_id):
        return (torch.Tensor(targets) == class_id).nonzero().reshape(-1)

    con_sim = torch.zeros(ph_cbm._num_concepts)
    cls_subset = Subset(training_d, indices=indices_tensor(training_d.targets, cls_id))
    loader = DataLoader(cls_subset, batch_size=128, shuffle=False, num_workers=2)
    for X_batch, _ in loader:
        _, con_sim_tmp = ph_cbm.predict(X_batch)
        con_sim += torch.sum(con_sim_tmp, dim=0)
    con_sim = con_sim / len(training_d)
    con_sim = con_sim.cpu()


    cls_weights = weights[cls_id,:] * con_sim
    cls_weights_asorted = torch.argsort(torch.abs(cls_weights), descending=True)
    cls_sorted = cls_weights[cls_weights_asorted]
    cls_sorted_per = torch.abs(cls_sorted) / torch.sum(torch.abs(cls_sorted)) * 100
    most_important_concepts = cls_weights_asorted[:args.concept_amount]

    plt.rcParams.update(bundles.iclr2024(rel_width=.5, nrows=2, ncols=3))
    fig, (ax2, ax3) = plt.subplots(1, 2, width_ratios=[1.5, 1], gridspec_kw={'wspace': 0})

    # Plot the bars shwoing the concept simularities. 
    y_pos = np.arange(args.concept_amount+1)
    labels = [(("NOT " if cls_weights[j] < 0 else "") + f'{j}' )
              for j in most_important_concepts] + [f"others"]
    cons = torch.cat((cls_sorted_per[:args.concept_amount], 
                    torch.sum(cls_sorted_per[args.concept_amount:]).unsqueeze(0)), 
                    dim=0).detach().numpy()

    color_pallete = sns.color_palette("deep", n_colors=(args.concept_amount+1))

    bars = sns.barplot(
        x=cons, 
        y=labels, 
        ax=ax2, 
        palette=color_pallete
    )
    ax2.set_ylabel('Concept id')
    ax2.set_xlabel('Avg concept contribution in \%')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    x = 0
    for c_id in most_important_concepts:
        crops_ids = data[c_id.item()]
        best_crops_ids = np.array(crops_ids[:5])
        img_ids = best_crops_ids // n_patches
        patch_ids = best_crops_ids % n_patches + np.array(list(range(len(best_crops_ids))))*n_patches
        all_images = torch.stack([training_data[img_id][0] for img_id in img_ids], dim=0)
        patches = torch.nn.functional.unfold(all_images, kernel_size=patch_size, stride=strides)
        patches = patches.transpose(1, 2).contiguous().view(-1, 3, patch_size, patch_size)
        best_crops = patches[patch_ids]
        y = 0
        for crop in best_crops:
            size = 0.22
            ax_inset = inset_axes(ax3, 
                                width=size, 
                                height=size, 
                                bbox_to_anchor=(y / 3.5, 1 - (x + 1) / 2.8, size, size), 
                                bbox_transform=ax3.transAxes,
                                borderpad=0)
            if y == 0:
                triangle = pchs.Polygon(np.array([[0, 1], [0, 0], [1, 0.5]]), facecolor=color_pallete[x])
                ax_inset.add_patch(triangle)
                ax_inset.axis('off')
                y += 1
                continue
            ax_inset.imshow(prepare_img(crop))
            ax_inset.axis('off')
            y += 1
        x += 1
    ax3.axis('off')

    # Save the figure
    fig.savefig(os.path.join(plot_folder, f"{cls_id}_{idx_to_class[cls_id].replace('/', '-').replace(' ', '_')}.pdf"), dpi=300)
    plt.close()