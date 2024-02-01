import matplotlib.pyplot as plt
import numpy as np
from utils.parser import Parser

parser = Parser()
sbs = parser.raw_to_df(['files/0_500_pickscore_coco'], do_overlap=True)

models = ['addxl', 'lcmxl', 'sd21', 'sdxl']

for model in models:
    fig, axes = plt.subplots(1, 5, sharex=True, sharey=True)
    for i, factor in enumerate(['angle', 'style', 'similar', 'background', 'main_object']):
        dict_ = {}
        values = sbs[factor]
        names = sbs['image_1']
        new_values = [value for i, value in enumerate(values) if model in names[i]]
        for value in new_values:
            try:
                dict_[value] += 1
            except KeyError:
                dict_[value] = 0
                dict_[value] += 1
        axes[i].set_title(factor)
        axes[i].bar(dict_.keys(), dict_.values())

    plt.suptitle(f'{model}')
    plt.savefig(f'{model}.png')