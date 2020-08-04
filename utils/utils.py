import argparse
import cv2
import os
import yaml
import code
from pathlib import Path
import random
import numpy as np
import torch

from superpoint.utils.var_dim import toNumpy, squeezeToNumpy
from superpoint.models.model_utils import SuperPointNet_process

def get_superpoint_2d_matches(descriptor_matches, image1_keypoints, image2_keypoints, num_matches):
#     sort_index = np.argsort(descriptor_matches[:, 2])
#     image1_keypoints = toNumpy(image1_keypoints)
#     image2_keypoints = toNumpy(image2_keypoints)

#     if (len(sort_index) > num_matches):
#         sort_index = sort_index[-num_matches:]
        
#     match_pts = np.zeros((len(sort_index), 4))
    
#     for idx, match in enumerate(sort_index):
#         match_indices = descriptor_matches[match]

#         match_pts[idx][:2] = image1_keypoints[int(match_indices[0])].astype(int)
#         match_pts[idx][2:] = image2_keypoints[int(match_indices[1])].astype(int)
        
        

#     return match_pts

    image1_keypoints = toNumpy(image1_keypoints)
    image2_keypoints = toNumpy(image2_keypoints)

    code.interact(local=locals())

    match_pts = np.zeros((len(descriptor_matches), 4))

    for idx, match in enumerate(descriptor_matches):
        match_pts[idx][:2] = image1_keypoints[int(match[0])].astype(int)
        match_pts[idx][2:] = image2_keypoints[int(match[1])].astype(int)
        
        

    return match_pts



def prep_superpoint_image(image, new_hw):
    resized_image = cv2.resize(image, (new_hw[1], new_hw[0])) #Why does the hw ordering convention change every three days..
    return torch.from_numpy(cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)/ 255.0).cuda().float().unsqueeze(0).unsqueeze(0)

def prep_trianflow_image(image, new_hw):
    resized_image = cv2.resize(image, (new_hw[1], new_hw[0])) #God knows why this is wh not hw
    return torch.from_numpy(np.transpose(resized_image/ 255.0, [2,0,1])).cuda().float().unsqueeze(0), resized_image


def desc_to_sparseDesc(outs):
    """
    Gets the sparse descriptors given a sparse set of keypoints from magicpoint
    
    Parameters: 
        outs : dict with keys : {'semi', 'desc', 'pts'}
    
    Returns : 
        desc : np [N,D]
    """
    return SuperPointNet_process.sample_desc_from_points(outs['desc'], outs['pts'])
     

def get_configs(path): 
    """
    Returns the configs for the model and general hyperparameters

    Returns: 
        cfg : the remaining hyperparameter configs
    """
    with open(path, "r") as f:
        cfg = yaml.load(f)

    #trianflow configs
    trianflow_cfg = pObject()
    for attr in list(cfg["models"]["trianflow"].keys()):
        setattr(trianflow_cfg, attr, cfg["models"]["trianflow"][attr])

    model_cfg = { 'trianflow' : trianflow_cfg, 'superpoint' : cfg['models']['superpoint']}

    return model_cfg, cfg


def load_image_pair(image_path, sequence):
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
        images.append(image)
    return images

def load_camera_intrinsics(image_path, sequence, raw_hw, img_hw):
    """
    loads the camera intrinsics for the given sequence
    
    Parameters : 
        raw_hw : the h and w of the raw data
        img_hw : the h and w that we process the images on
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