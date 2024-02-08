from sklearn.model_selection import train_test_split

from utils.parser import Parser
from dataset import DiversityDataset
from models.src.DivReward import DivReward
from train import run_train

from models.baseline_blip import preprocess, model

# Prepare dataset and models
parser = Parser()
paths = ['files/0_500_pickscore_coco',
         'files/diverse_coco_pick_3_per_prompt_500_1000.out',
         'files/diverse_coco_pick_3_per_prompt_1000_1500',
         'files/diverse_coco_pick_3_per_prompt_1500_2000',
         'files/diverse_coco_pick_3_per_prompt_2000_2500']
df = parser.raw_to_df(paths, do_overlap=True, keep_no_info=False)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)
train_dataset = DiversityDataset(train_df,
                                 local_path='/extra_disk_1/quickjkee/diversity_images',
                                 preprocess=preprocess)
valid_dataset = DiversityDataset(test_df,
                                 local_path='/extra_disk_1/quickjkee/diversity_images',
                                 preprocess=preprocess)
main_model = DivReward()

run_train(train_dataset=train_dataset,
          valid_dataset=valid_dataset,
          model=main_model)
