#!/usr/bin/env python 

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.siftflow import SiftFlow


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

import warnings
warnings.filterwarnings("ignore")

from infer_deepF import deepF_frontend

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Unit Tests for Correspondence_model")
    arg_parser.add_argument('-c', '--config_file', default='./configs/siftflow_deepF.yaml', help='config file.')
    args = arg_parser.parse_args()

    #Require GPU
    if not torch.cuda.is_available():
        assert RuntimeError('Must have access to a GPU to run this script')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #do config stuff
    model_cfg, cfg = get_configs(args.config_file, mode='siftflow')

    deepF_fe = deepF_frontend(cfg["models"]["deepF"], device=device)
    # load model
    deepF_fe.load_model()
    deepF_fe.prepare_model()

    ### siftflow

    #do config stuff
    model_cfg, cfg = get_configs(args.config_file, mode='siftflow')

    #create the model
    model = SiftFlow(model_cfg, cfg)
    model.load_modules(model_cfg)
    model.cuda()
    # model.eval()


    sequence = "10"

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