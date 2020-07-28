#!/usr/bin/env python 

from models.corresondence_model import SuperFlow

import argparse
import cv2
import os
import yaml
import code
from pathlib import Path
import random
import numpy as np
import torch

#utils
from utils.utils import load_image_pair, load_camera_intrinsics, pObject

#bad juju probably should leave this commented in the worst case that stuff breaks
# import warnings
# warnings.filterwarnings("ignore")

def test_inference(model, image1, image2, K, K_inv, match_num):
    outs = model.inference(image1, image2, K, K_inv, match_num)
    return outs

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Unit Tests for Correspondence_model")
    arg_parser.add_argument('-c', '--config_file', default='./../../configs/train.yaml', help='config file.')
    arg_parser.add_argument('-g', '--gpu', type=str, default=0, help='gpu id.')
    args = arg_parser.parse_args()

    #do config stuff
    with open(args.config_file, "r") as f:
        cfg = yaml.load(f)

    trianflow_cfg = pObject()
    for attr in list(cfg["models"]["trianflow"].keys()):
        setattr(trianflow_cfg, attr, cfg["models"]["trianflow"][attr])

    model_cfg = { 'trianflow' : trianflow_cfg }

    #create the model
    model = SuperFlow(model_cfg)
    model.load_modules(model_cfg)
    model.cuda()
    model.eval()

    #load a pair of images
    vo_sequences_root = Path(cfg["kitti"]["vo_path"]) / 'sequences'
    images = load_image_pair(vo_sequences_root, '09', cfg['img_hw'][0], cfg['img_hw'][1])
    
    #load camera intrinsics
    K = load_camera_intrinsics(vo_sequences_root, '09', cfg['raw_hw'], cfg['img_hw'])
    K_inv = np.linalg.inv(K)
    
    #inference
    outs = test_inference(model, images[0], images[1], K, K_inv, model_cfg['trianflow'].match_num)

    print('pass')
