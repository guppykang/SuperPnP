#!/usr/bin/env python 
from models.corresondence_model import SuperFlow

import argparse
import cv2
import os
import yaml
import code
from pathlib import Path
import random

def test_inference(cfg):
    pass


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
        image = cv2.imread(str(image_dir / i / '.png'))
        image = cv2.resize(image, (h, w))
        images.append(image)
    return images

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

    #load a pair of images
    vo_sequences_root = Path(cfg["kitti"]["vo_path"]) / 'sequences'
    images = load_image_pair(vo_sequences_root, '09', cfg['img_hw'][0], cfg['img_hw'][1])

    #inference

    print('pass')