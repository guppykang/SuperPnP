#!/usr/bin/env python 

from models.superflow import SuperFlow

import argparse
import cv2
import os
import yaml
import code
from pathlib import Path
import random
import numpy as np
import torch
import pickle

#utils stuff
from utils.utils import load_image_pair, load_camera_intrinsics, pObject, get_configs, get_random_sequence

#dataset stuff
from TrianFlow.core.dataset.kitti_odo import KITTI_Odo
from utils.superpnp_dataset import KITTI_Dataset

#bad juju probably should leave this commented in the worst case that stuff breaks
# import warnings
# warnings.filterwarnings("ignore")

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Unit Tests for Correspondence_model")
    arg_parser.add_argument('-c', '--config_file', default='./../../configs/superflow.yaml', help='config file.')
    args = arg_parser.parse_args()

    #Require GPU
    if not torch.cuda.is_available():
        assert RuntimeError('Must have access to a GPU to run this script')

    #do config stuff
    model_cfg, cfg = get_configs(args.config_file)    

    #create the model
    model = SuperFlow(model_cfg, cfg)
    model.load_modules(model_cfg)
    model.cuda()
    # model.eval()

    sequence = get_random_sequence()

    #load a pair of images
    vo_sequences_root = Path(cfg["kitti"]["vo_path"])
    images = load_image_pair(vo_sequences_root, sequence)
    
    #load camera intrinsics
    K = load_camera_intrinsics(vo_sequences_root, sequence, cfg['raw_hw'], cfg['img_hw'])
    K_inv = np.linalg.inv(K)
    #TODO : assert that the K has the right dims that we expect to have 3x3

    #inference
    outs = model.inference(images[0], images[1], K, K_inv, cfg['img_hw'])
    torch.save(outs, 'inference_outs.pth')
    
    
    #get dataset and process
    kitti_raw_dataset = KITTI_Odo(vo_sequences_root)
    vo_sequences_processed = Path(cfg["kitti"]["procressed_data_path"])
    kitti_raw_dataset.prepare_data_mp(vo_sequences_processed, stride=1)

    dataset = KITTI_Dataset(vo_sequences_processed, num_scales=model_cfg['trianflow'].num_scales, img_hw=cfg['img_hw'], num_iterations=(cfg['num_iterations'] - cfg['iter_start']) * cfg['batch_size'])

    #create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], drop_last=False)

    #test foward pass
    

