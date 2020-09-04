#!/usr/bin/env python 

#models
from models.attention import decoder, encoder, AttentionMatching
from models.superglueflow import SuperGlueFlow

import code
from utils.utils import get_configs
from pathlib import Path
import torch
import numpy as np 
from tqdm import tqdm

#datasets
from TrianFlow.core.dataset.kitti_odo import KITTI_Odo
from utils.superpnp_dataset import KITTI_Dataset


model_cfg, cfg = get_configs('../../configs/kitti/superglueflow.yaml', mode='superglueflow')    

#get dataset and process
kitti_raw_dataset = KITTI_Odo(cfg["kitti"]["vo_path"],  cfg["kitti"]["vo_gts"])
vo_sequences_processed = Path(cfg["kitti"]["procressed_data_path"])
kitti_raw_dataset.prepare_data_mp(vo_sequences_processed, stride=1)

dataset = KITTI_Dataset(vo_sequences_processed, num_scales=model_cfg['trianflow'].num_scales, img_hw=cfg['img_hw'], num_iterations=(cfg['num_iterations'] - cfg['iter_start']) * cfg['batch_size'], stride=1)

#create dataloader
dataloader = torch.utils.data.DataLoader(dataset,  batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], drop_last=False)

#create the model
matcher = SuperGlueFlow(model_cfg, cfg)
encoder = encoder()
decoder = decoder()
model = AttentionMatching(matcher, encoder, decoder).cuda().eval()

for iteration, inputs in enumerate(tqdm(dataloader)):
    print(f'iteration {iteration}')
    images, K_batch, Kinv_batch, gt = inputs
    h = int(images.shape[2]/2)
    image1_batch, image2_batch = images[:, :, :h, :], images[:, :, h:, :]
#     preds = model(model.get_matches(image1_batch, image2_batch, K_batch, Kinv_batch))
    
    break