'''
@File       :   DivReward.py
@Time       :   2023/02/28 19:53:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   ImageReward Reward model for reward model.
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from config.options import *
from config.utils import *
from models.blip_pretrain import blip_pretrain
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class HingeReward(nn.Module):
    def __init__(self, backbone, img_lora=False):
        super().__init__()
        self.device = 'cpu'

        self.visual_encoder = backbone

        if img_lora:
            self.visual_encoder.requires_grad_(False)
            for name, parms in self.blip.visual_encoder.named_parameters():
                if 'lora' in name:
                    parms.requires_grad_(True)

        if not img_lora:
            # fix certain ratio of layers
            self.image_layer_num = 24 if config['BLIP']['vit'] == 'large' else 12
            if opts.fix_rate > 0:
                image_fix_num = "blocks.{}".format(int(self.image_layer_num * opts.fix_rate))
                for name, parms in self.blip.visual_encoder.named_parameters():
                    parms.requires_grad_(False)
                    if image_fix_num in name:
                        break

        all = 0
        trainable = 0
        for name, parms in self.blip.visual_encoder.named_parameters():
            all += 1
            if parms.requires_grad:
                trainable += 1
        print(f'Visual trainable layers {trainable}/{all}')


    def forward(self, batch_data):
        # parse data
        batch_data = self.encode_pair(batch_data)
        emb_img_1, emb_img_2 = batch_data['emb_img_1'], batch_data['emb_img_2']

        return emb_img_1, emb_img_2

    def encode_pair(self, batch_data):
        img_1, img_2 = batch_data['image_1'], batch_data['image_2']
        img_1 = img_1.to(self.device)  # [batch_size, C, H, W]
        img_2 = img_2.to(self.device)  # [batch_size, C, H, W]

        # img
        image_embeds_1 = self.blip.visual_encoder(img_1)
        image_embeds_2 = self.blip.visual_encoder(img_2)

        # get batch data
        batch_data = {'emb_img_1': image_embeds_1[:, 0, :].float(),
                      'emb_img_2': image_embeds_2[:, 0, :].float()}

        return batch_data
