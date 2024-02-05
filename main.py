import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from utils.parser import Parser
from dataset import DiversityDataset
from models.baseline_clip import preprocess, ClipBase


parser = Parser()
df = parser.raw_to_df(['files/0_500_pickscore_coco'], do_overlap=True)
dataset = DiversityDataset(df, preprocess=preprocess)

clip_baseline = ClipBase()
factor = 'angle'
pred, true = clip_baseline(dataset, factor)
pred = np.array(pred)
true = np.array(true)
idx = true != -1
true = list(true[idx])
pred = pred[idx]
print(true)

prs = []
rs = []
f1s = []
accs = []
means = []
threshs = np.linspace(min(pred), max(pred), 10)
for thresh in threshs:
    curr_pred = np.array(pred) > thresh
    curr_pred = np.array([not item for item in curr_pred]) * 1
    curr_pred = list(curr_pred.astype(int))
    means.append(np.sum(curr_pred * np.array(true)) / np.linalg.norm(curr_pred * np.array(true)) )
    accs.append(accuracy_score(true, curr_pred))
    f1s.append(f1_score(true, curr_pred))
    rs.append(recall_score(true, curr_pred))
    prs.append(precision_score(true, curr_pred))

stupid = [1] * len(true)
f1_stupid =f1_score(true, stupid)
r_stupid = recall_score(true, stupid)
ac_stupid = accuracy_score(true, stupid)
pr_stupid = precision_score(true, stupid)

fake_taxi = np.array([not item for item in true]) * 1
true_mean = np.sum(pred * fake_taxi) / np.linalg.norm(pred * fake_taxi)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 5, figsize=(30, 5))
metrics = ['precision', 'recall', 'f1', 'accuracy', 'means']
values = [prs, rs, f1s, accs, means]
stupid = [pr_stupid, r_stupid, f1_stupid, ac_stupid, true_mean]
for j in range(5):
    axes[j].set_title(metrics[j])
    axes[j].plot(threshs, values[j])
    axes[j].axhline(stupid[j], color='r', linestyle='-')

plt.suptitle(f'{factor}')
plt.savefig(f'{factor}.png')
