import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from utils.parser import Parser
from dataset import DiversityDataset
from models.baseline_clip import preprocess, ClipBase


parser = Parser()
df = parser.raw_to_df(['files/0_500_pickscore_coco'], do_overlap=True).head(10)
dataset = DiversityDataset(df, preprocess=preprocess)

clip_baseline = ClipBase()
factor = 'main_object'
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
threshs = np.linspace(min(pred), max(pred), 10)
for thresh in threshs:
    curr_pred = np.array(pred) > thresh
    curr_pred = np.array([not item for item in curr_pred]) * 1
    curr_pred = list(curr_pred.astype(int))
    accs.append(accuracy_score(true, curr_pred))
    f1s.append(f1_score(true, curr_pred))
    rs.append(recall_score(true, curr_pred))
    prs.append(precision_score(true, curr_pred))


import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
metrics = ['precision', 'recall', 'f1', 'accuracy']
values = [prs, rs, f1s, accs]
for j in range(4):
    axes[j].set_title(metrics[j])
    axes[j].plot(threshs, values[j])

plt.suptitle(f'{factor}')
plt.savefig(f'metrics.png')