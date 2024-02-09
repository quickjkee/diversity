import sys
sys.path.append('/home/quickjkee/diversity/models/src')
sys.path.append(f'/home/quickjkee/diversity/models/src/config')
sys.path.append(f'/home/quickjkee/diversity/models')

from models.src.config.options import *
from models.src.config.utils import *
from models.lora_wrapper import ViTConfig, ViTModel
from peft import PeftModel, LoraConfig, get_peft_model

os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
opts.BatchSize = opts.batch_size * opts.accumulation_steps * opts.gpu_num

from sklearn.model_selection import train_test_split

from utilss.parser import Parser
from dataset import DiversityDataset
from models.src.DivReward import DivReward
from train import run_train
from models.baseline_blip import blip_pretrain


#from models.baseline_clip import preprocess, model
from models.baseline_blip import blip_pretrain, preprocess

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

backbone = blip_pretrain(pretrained=config['blip_path'], image_size=config['BLIP']['image_size'],
                         vit=config['BLIP']['vit'])


######################
use_lora = True

# FOR DREAMSIMSIM
#backbone.visual_encoder = model.extractor_list[0].model

# FOR BLIP
model = backbone.visual_encoder

# FOR OPENCLIP
#model = model.visual
#model.pool_type = 'none'

if use_lora:
    lora_config =  {
            "r": 16,
            "lora_alpha": 0.5,
            "lora_dropout": 0.3,
            "bias": "none",
            "target_modules": ['qkv']
             }
    lora_config = LoraConfig(**lora_config)
    model = get_peft_model(ViTModel(model, ViTConfig()), lora_config)
##########################

backbone.visual_encoder = model
backbone.visual_encoder.requires_grad_(True)

######### TEXT
#text_encoder = backbone.text_encoder
#for name, parms in text_encoder.named_parameters():
#    print(name)
#lora_config =  {
#            "r": 32,
#            "lora_alpha": 5,
#            "lora_dropout": 0.1,
#            "bias": "none",
#            "target_modules": ['query', 'value', 'key']
#             }
#lora_config = LoraConfig(**lora_config)
#text_encoder = get_peft_model(text_encoder, lora_config)
#backbone.text_encoder = text_encoder

main_model = DivReward(backbone, img_lora=use_lora)

run_train(train_dataset=train_dataset,
          valid_dataset=valid_dataset,
          model=main_model)
