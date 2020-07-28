import argparse
import cv2
import os
import yaml
import code
from pathlib import Path
import random
import numpy as np
import torch


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

class pObject(object):
    def __init__(self):
        pass