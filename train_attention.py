#!/usr/bin/env python 

#models
from models.attention import decoder, encoder, AttentionMatching
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
import argparse

from superpoint.utils.var_dim import toNumpy

arg_parser = argparse.ArgumentParser(
    description="Training on attention module of pipeline."
)
#TODO : Experiment with training on bigger strides for kitti
arg_parser.add_argument('--stride', type=int, default='1', help='Stride between images')
args = arg_parser.parse_args()


model_cfg, cfg = get_configs('./configs/kitti/superglueflow.yaml', mode='attention')    

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
#TODO : refactor this so that it can do it for other models as well
matcher = SuperGlueFlow(model_cfg, cfg)
encoder = encoder()
decoder = decoder()
model = AttentionMatching(matcher, encoder, decoder).cuda().eval()
encoder_optimizer = optim.Adam(model.encoder.parameters(), lr=float(model_cfg['attention']['encoder_lr']), betas=(0.5, 0.999) )
decoder_optimizer = optim.Adam(model.decoder.parameters(), lr=float(model_cfg['attention']['decoder_lr']), betas=(0.5, 0.999) )

for iteration, inputs in enumerate(tqdm(dataloader)):
    print(f'iteration {iteration}')
    images, images_gray, K_batch, K_inv_batch, gt = inputs
    
    #For now we will try it on the left image only
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    h = int(images.shape[2]/2)
    attention_out = model(images[:, :, :h, :])

    code.interact(local=locals())
    
    #TODO : choose matches in the regions of interest
    
    batch_poses, batch_loss_scale, gt_attention = train_vo.process_video(images, images_gray, K_batch, K_inv_batch, model)
    
    loss = torch.mean(-attention_out * gt_attention) * batch_loss_scale
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    
    #TODO : back prop
    
    
    
    print(f'loss scale : {batch_loss_scale}\n')
    
    
    
    
    
    
    
    