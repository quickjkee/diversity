import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader, TensorDataset

from utils.parser import Parser
from dataset import DiversityDataset
from models.baseline_clip import preprocess, ClipBase

def samples_accuracy(true, pred):
    labels = set(true)
    class_weights = {}
    for label in labels:
        class_weights[label] = 0
        for el in true:
            if el == label:
                class_weights[label] += 1
        class_weights[label] = 1 / class_weights[label]

    sample_weights = [class_weights[item] for item in true]
    dataset = TensorDataset([[tr, pr] for tr, pr in zip(true, pred)])
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(true), replacement=True)
    dl = DataLoader(dataset, sampler=sampler, batch_size=len(true))
    for item in dl:
        print(item)

parser = Parser()
paths = ['../files/diverse_coco_pick_3_per_prompt_1000_1500',
         '../files/0_500_pickscore_coco',
         '../files/diverse_coco_pick_3_per_prompt_500_1000.out']
df = parser.raw_to_df(paths, do_overlap=True)
dataset = DiversityDataset(df, preprocess=preprocess)

clip_baseline = ClipBase()
factor = 'angle'
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
    sim = np.dot(curr_pred, true) / (np.linalg.norm(curr_pred) * np.linalg.norm(true))
    means.append((sim + 1) / 2)
    accs.append(samples_accuracy(true, curr_pred))
    break

stupid = [1] * len(true)
ac_stupid = accuracy_score(true, stupid)

true = np.array([not item for item in true]) * 1
true_mean = np.dot(pred, true) / (np.linalg.norm(pred) * np.linalg.norm(true))

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(30, 5))
metrics = ['accuracy', 'means']
values = [accs, means]
stupid = [ac_stupid, true_mean]
for j in range(2):
    axes[j].set_title(metrics[j])
    axes[j].plot(threshs, values[j])
    axes[j].axhline(stupid[j], color='r', linestyle='-')

plt.suptitle(f'{factor}')
plt.savefig(f'{factor}.png')
