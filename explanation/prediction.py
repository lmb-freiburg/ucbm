import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import seaborn as sns
import torch
from torch.utils.data import Subset
from tqdm import tqdm, trange
from tueplots import bundles

from data_loader import load_data
from model_loader import get_model
from ucbm import UCBM
from concept_auxiliaries.concept_auxiliaries import raw_concept_sims
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
parser.add_argument("-b", "--backbone", type=str, default="resnet50_v2",
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
parser.add_argument("--img_per_class", type=int,
                    default=5,
                    help="How many images per class to show")
parser.add_argument("--class_idcs", type=int, nargs="*")
parser.add_argument("--concepts_to_show", type=int,
                    default=5,
                    help="How many concepts to show")
parser.add_argument("--save_folder_name", type=str, default="")
parser.add_argument("--only_missclassified", action="store_true")
parser.add_argument("--gt", type=int, default=-1)
parser.add_argument("--folder_name", type=str, default="")
args = parser.parse_args()

# Load models.
model = get_model(args.backbone, args.device)
print(f"Model {args.backbone} loaded successfully...")
training_data = load_data(args.dataset, True, model.transform)
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
    indices = []
    idx_to_cls = {v: k for k, v in dataset.class_to_idx.items()}

    indices = [i for i, target in enumerate(dataset.targets) if target == cls_id]
    if args.img_per_class != -1:
        random.shuffle(indices)
        indices = indices[:args.img_per_class]

    subset = Subset(dataset, indices)
    con_sims_raw = raw_concept_sims(ph_cbm._h, subset, model.g, 64, device="cpu")
    subset = torch.stack([subset[i][0] for i in range(len(subset))], dim=0)

    preds, con_acts_raw = ph_cbm.predict(subset)
    con_acts = con_acts_raw.detach().clone()
    for i in range(subset.shape[0]):
        if args.gt == -1:
            j = torch.argmax(preds[i]).item()
        elif args.gt == -2:
            j = cls_id
        else:
            j = args.gt
        con_acts[i,:] = con_acts[i,:] * weights[j,:]

    for i in trange(len(indices), leave=False):
        pred = preds[i]
        con_act = con_acts[i,:]
        con_act_asorted = torch.argsort(torch.abs(con_act), descending=True)
        con_act_sorted = torch.abs(con_act[con_act_asorted])

        pred_s = torch.argmax(pred).item()
        if pred_s == cls_id and args.only_missclassified:
            continue
        conf = torch.nn.functional.softmax(pred, dim=0)[pred_s].item()
    
        plt.rcParams.update(bundles.iclr2024(rel_width=.5, nrows=3, ncols=4))
        fig, (ax1, ax_tmp, ax2, ax3) = plt.subplots(1, 4, width_ratios=[0.7, 1.0, 1.2, 1.4], gridspec_kw={'wspace': 0})
        ax_tmp.axis('off')
        
        ax1.imshow(prepare_img(subset[i,:,:,:]))
        ax1.axis('off')

        # Plot the bars shwoing the concept simularities. 
        y_pos = np.arange(args.concepts_to_show+1)
        labels = [(("NOT " if con_act[j] < 0 else "") + f'{j}' )
                  for j in con_act_asorted[:args.concepts_to_show]] + [f"others"]
                    # [f"{con_act_asorted.numel() - args.concepts_to_show} others"]
        cons = torch.cat((con_act_sorted[:args.concepts_to_show], 
                        torch.sum(con_act_sorted[args.concepts_to_show:]).unsqueeze(0)), 
                        dim=0).detach().numpy()

        color_pallete = sns.color_palette("deep", n_colors=(args.concepts_to_show+1))

        bars = sns.barplot(
            x=cons, 
            y=labels, 
            ax=ax2, 
            palette=color_pallete
        )
        ax2.set_ylabel('Concept id')
        ax2.set_xlabel('Concept contribution')

        x = 0
        for c_id in con_act_asorted[:args.concepts_to_show]:
            crops_ids = data[c_id.item()]
            best_crops_ids = np.array(crops_ids[:5])
            img_ids = best_crops_ids // n_patches
            patch_ids = best_crops_ids % n_patches + np.array(list(range(len(best_crops_ids))))*n_patches
            all_images = torch.stack([training_data[img_id][0] for img_id in img_ids], dim=0)
            patches = torch.nn.functional.unfold(all_images, kernel_size=patch_size, stride=strides)
            patches = patches.transpose(1, 2).contiguous().view(-1, 3, patch_size, patch_size)
            best_crops = patches[patch_ids]
            for y in range(5):
                size = 0.17
                ax_inset = inset_axes(ax3, 
                                      width=size, 
                                      height=size, 
                                      bbox_to_anchor=(.2 + y * 1.3 * size, 1 - (x + 1) * 1.6 * size + 0.1, size, size), 
                                      bbox_transform=ax3.transAxes,
                                      borderpad=0)
                if y == 0:
                    triangle = pchs.Polygon(np.array([[0, 1], [0, 0], [1, 0.5]]), facecolor=color_pallete[x])
                    ax_inset.add_patch(triangle)
                    ax_inset.axis('off')
                    continue
                ax_inset.imshow(prepare_img(best_crops[y-1]))
                ax_inset.axis('off')
            x += 1
        ax3.axis('off')

        # Save the plot to file.
        sns.despine(fig) 
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, f"{str(cls_id).zfill(zfill_cls_id)}_{str(indices[i]).zfill(zfill_examples)}_gt_{idx_to_cls[cls_id].replace('/', '-').replace(' ', '_')}_pred_{idx_to_cls[pred_s].replace('/', '-').replace(' ', '_')}_conf_{int(10000*conf)}.pdf"), dpi=300)
        plt.close()