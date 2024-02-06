import torch
import numpy as np

from sklearn.metrics import accuracy_score
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader, TensorDataset

from utils.parser import Parser
from dataset import DiversityDataset
from models.baseline_clip import preprocess


parser = Parser()
paths = ['files/diverse_coco_pick_3_per_prompt_1000_1500',
         'files/0_500_pickscore_coco',
         'files/diverse_coco_pick_3_per_prompt_500_1000.out']
df = parser.raw_to_df(paths, do_overlap=True)[:30]
dataset = DiversityDataset(df, preprocess=preprocess)
train_loader = DataLoader(dataset, batch_size=5)

for step, batch_data_package in enumerate(train_loader):
    print(batch_data_package['image_1'].shape)

"""
def samples_metric(true, pred, n_boots=30, type='accuracy'):
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
            if type == 'accuracy':
                acc = accuracy_score(true_labels, pred_labels) 
            elif type == 'similarity':
                true_labels = np.array(true_labels)
                pred_labels = np.array(pred_labels)
                acc = np.dot(pred_labels, true_labels) / (np.linalg.norm(pred_labels) * np.linalg.norm(true_labels))
            accs.append(acc)

    return np.mean(accs)

def sim_fn(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    sim = np.dot(pred, true) / (np.linalg.norm(pred) * np.linalg.norm(true))
    return sim

parser = Parser()
paths = ['files/diverse_coco_pick_3_per_prompt_1000_1500',
         'files/0_500_pickscore_coco',
         'files/diverse_coco_pick_3_per_prompt_500_1000.out']
df = parser.raw_to_df(paths, do_overlap=True)
dataset = DiversityDataset(df, preprocess=preprocess)

clip_baseline = ClipBase()
factor = 'background'
pred, true = clip_baseline(dataset, factor)
pred = np.array(pred)
true = np.array(true)
idx = true != -1
true = list(true[idx])
pred = pred[idx]

accs = []
means = []
threshs = np.linspace(min(pred), max(pred), 10)
for thresh in threshs:
    curr_pred = (np.array(pred) > thresh) * 1
    curr_pred = list(curr_pred.astype(int))
    sim = sim_fn(true, curr_pred)
    means.append(sim)
    accs.append(samples_metric(true, curr_pred))

stupid = [0] * len(true)
ac_stupid = samples_metric(true, stupid)

true = np.array([not item for item in true]) * 1
true_mean = sim_fn(true, pred)
stupid_mean = 0

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
metrics = ['accuracy', 'means']
values = [accs, means]
stupid = [ac_stupid, stupid_mean]
for j in range(2):
    axes[j].set_title(metrics[j])
    axes[j].plot(threshs, values[j])
    axes[j].axhline(stupid[j], color='r', linestyle='-')
    if metrics[j] == 'means':
        axes[j].axhline(true_mean, color='g', linestyle='-')

plt.suptitle(f'{factor}')
plt.savefig(f'{factor}.png')
"""