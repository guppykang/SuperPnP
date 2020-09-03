#!/usr/bin/env python 

from models.attention import decoder, encoder, AttentionMatching
import code
from utils.utils import get_configs
from pathlib import Path
import torch
import numpy as np 
from TrianFlow.core.dataset.kitti_odo import KITTI_Odo
from utils.superpnp_dataset import KITTI_Dataset
from tqdm import tqdm


model_cfg, cfg = get_configs('../../configs/kitti/superglueflow.yaml')    

#get dataset and process
kitti_raw_dataset = KITTI_Odo(cfg["kitti"]["vo_path"])
vo_sequences_processed = Path(cfg["kitti"]["procressed_data_path"])
kitti_raw_dataset.prepare_data_mp(vo_sequences_processed, stride=1)

dataset = KITTI_Dataset(vo_sequences_processed, num_scales=model_cfg['trianflow'].num_scales, img_hw=cfg['img_hw'], num_iterations=(cfg['num_iterations'] - cfg['iter_start']) * cfg['batch_size'])

#create dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], drop_last=False)

for iteration, inputs in enumerate(tqdm(dataloader)):
    print(inputs.shape)
    break