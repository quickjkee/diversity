import sys
sys.path.append('/home/quickjkee/diversity/models/src')


import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader, TensorDataset

from utilss.parser import Parser
from dataset import DiversityDataset
from models.baseline_clip import preprocess, ClipBase
#from models.baseline_blip import preprocess, BlipBase

def samples_metric(true, pred, n_boots=30):
    labels = set(true)
    class_weights = {}
    for label in labels:
        class_weights[label] = 0
        for el in true:
            if el == label:
                class_weights[label] += 1
        class_weights[label] = 1 / class_weights[label]

    sample_weights = [class_weights[item] for item in true]
    idx = list(range(len(true)))
    dataset = TensorDataset(torch.tensor(idx))
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(true), replacement=True)
    accs = []
    for _ in range(n_boots):
        dl = DataLoader(dataset, sampler=sampler, batch_size=len(true))
        for item in dl:
            item = item[0]
            true_labels = [true[it.item()] for it in item]
            pred_labels = [pred[it.item()] for it in item]
            acc = accuracy_score(true_labels, pred_labels)
            accs.append(acc)

    return np.mean(accs), np.std(accs)

parser = Parser()
paths = ['files/0_500_pickscore_coco',
         'files/diverse_coco_pick_3_per_prompt_500_1000.out',
         'files/diverse_coco_pick_3_per_prompt_1000_1500',
         'files/diverse_coco_pick_3_per_prompt_1500_2000',
         'files/diverse_coco_pick_3_per_prompt_2000_2500']
df = parser.raw_to_df(paths, do_overlap=True, keep_no_info=False)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)
dataset_train = DiversityDataset(train_df,
                                 local_path='/extra_disk_1/quickjkee/diversity_images',
                                 preprocess=preprocess)
dataset_test = DiversityDataset(test_df,
                                local_path='/extra_disk_1/quickjkee/diversity_images',
                                preprocess=preprocess)
clip_baseline = ClipBase()

factor = 'main_object'
pred, true = clip_baseline(dataset_train, factor)
pred = np.array(pred)
true = np.array(true)
idx = true != -1
true = list(true[idx])
pred = pred[idx]

accs = []
threshs = np.linspace(min(pred), max(pred), 10)
for thresh in threshs:
    curr_pred = (np.array(pred) > thresh) * 1
    curr_pred = list(curr_pred.astype(int))
    accs.append(samples_metric(true, curr_pred)[0])

stupid = [0] * len(true)
ac_stupid = samples_metric(true, stupid)[0]
max_value = max(accs)

found_thresh = threshs[accs.index(max_value)]
pred_val, true_val = clip_baseline(dataset_test, factor)
pred_val = (np.array(pred_val) > found_thresh) * 1
pred_val = list(pred_val.astype(int))
val_acc, val_std = samples_metric(true_val, pred_val)
print(f'acc {val_acc}, std {val_std}')

fig, axes = plt.subplots(1, 1,  figsize=(10, 5))
metrics = ['accuracy']
values = [accs]
stupid = [ac_stupid]
for j in range(1):
    axes.set_title(metrics[j])
    axes.plot(threshs, values[j])
    axes.axhline(stupid[j], color='r', linestyle='-', label='majority class')
    axes.axhline(val_acc, color='g', linestyle='-', label='validation')

plt.legend()
plt.suptitle(f'{factor}')
plt.savefig(f'{factor}.png')
