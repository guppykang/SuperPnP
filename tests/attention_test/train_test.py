#!/usr/bin/env python 

#models
from models.attention import decoder, encoder, AttentionMatching, AttentionLoss
from models.superglueflow import SuperGlueFlow

import code
from utils.utils import get_configs
from pathlib import Path
import torch
import torch.optim as optim
import numpy as np 
from tqdm import tqdm
from utils.train_vo import train_vo

#datasets
from TrianFlow.core.dataset.kitti_odo import KITTI_Odo
from utils.superpnp_dataset import KITTI_Dataset


model_cfg, cfg = get_configs('../../configs/kitti/superglueflow.yaml', mode='attention')    

#get dataset and process
kitti_raw_dataset = KITTI_Odo(cfg["kitti"]["vo_path"],  cfg["kitti"]["vo_gts"])
vo_sequences_processed = Path(cfg["kitti"]["procressed_data_path"])
kitti_raw_dataset.prepare_data_mp(vo_sequences_processed, stride=1)
dataset = KITTI_Dataset(vo_sequences_processed, num_scales=model_cfg['trianflow'].num_scales, img_hw=cfg['img_hw'], num_iterations=(cfg['num_iterations'] - cfg['iter_start']) * cfg['batch_size'], stride=1)

#create dataloader
dataloader = torch.utils.data.DataLoader(dataset,  batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], drop_last=False)

#inference vo 
train_vo = train_vo()

#create the model
matcher = SuperGlueFlow(model_cfg, cfg)
encoder = encoder()
decoder = decoder()
model = AttentionMatching(matcher, encoder, decoder).cuda().eval()
loss = AttentionLoss()
encoder_optimizer = optim.Adam(model.encoder.parameters(), lr=float(model_cfg['attention']['encoder_lr']), betas=(0.5, 0.999) )
decoder_optimizer = optim.Adam(model.decoder.parameters(), lr=float(model_cfg['attention']['decoder_lr']), betas=(0.5, 0.999) )

for iteration, inputs in enumerate(tqdm(dataloader)):
    print(f'iteration {iteration}')
    images, images_gray, K_batch, K_inv_batch, gt = inputs
    
    #For now we will try it on the left image only
    h = int(images.shape[2]/2)
    attention_preds = model(images[:, :, :h, :])
    
    batch_poses, batch_loss_scale = train_vo.process_video(images, images_gray, K_batch, K_inv_batch, model)
    print(f'loss scale : {batch_loss_scale}\n')
    
    break
    
    
    
    
    
    
    