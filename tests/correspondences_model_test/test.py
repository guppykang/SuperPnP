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

#bad juju probably should leave this commented in the worst case that stuff breaks
# import warnings
# warnings.filterwarnings("ignore")

def test_inference(model, image1, image2, K, K_inv, match_num):
    outs = model.inference(image1, image2, K, K_inv, match_num)
    return outs


class pObject(object):
    def __init__(self):
        pass

def load_image_pair(image_path, sequence, h, w):
    """
    loads random pair of subsequent images for correspondence testing in the given sequence of vo data
    """
    seq_dir = Path(image_path) / sequence
    image_dir = seq_dir / 'image_2'
    num = len(os.listdir(image_dir))
    random_frame_t = random.randint(0, num-1)

    images = []
    for i in [random_frame_t, random_frame_t+1]:
        image_path = str(image_dir / ('%.6d.png'%i))
        image = cv2.imread(image_path)
        image = cv2.resize(image, (h, w))
        images.append(image)
    return images

def load_camera_intrinsics(image_path, sequence, raw_hw, img_hw):
    """
    loads the camera intrinsics for the given sequence
    """
    calib_path = Path(image_path) / sequence / 'calib.txt'
    
    with open(calib_path, 'r') as f:
        lines = f.readlines()
            
    data = lines[-1].strip('\n').split(' ')[1:]
    data = [float(k) for k in data]
    data = np.array(data).reshape(3,4)
    
    cam_intrinsics = data[:3,:3]
    cam_intrinsics[0,:] = cam_intrinsics[0,:] * img_hw[1] / raw_hw[1]
    cam_intrinsics[1,:] = cam_intrinsics[1,:] * img_hw[0] / raw_hw[0]
    return cam_intrinsics

    

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
