import argparse
import cv2
import copy
import os
import yaml
import code
from pathlib import Path
import random
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from datetime import datetime



from superpoint.utils.var_dim import toNumpy, squeezeToNumpy
from superpoint.models.model_utils import SuperPointNet_process

def vehicle_to_world(pose_t, vehicle_3d_points):
    """
    Do an inverse Transform on the 3d vehicle frame points to be in world coordinates
    
    pose_t : (3, 4) R, T 
    vehicle_3d_points : (n x 3)
    """
    R = pose_t[:3, :3]
    T = pose_t[:3, 3:]
    inverse_pose_t = np.concatenate((np.linalg.pinv(R), -T), axis=1)
    world_points = inverse_pose_t @ np.vstack((vehicle_3d_points.T, np.ones(vehicle_3d_points.shape[0])))
    
#decide what to do about the 1/3 decimal issue
#     for row_idx, row in enumerate(world_points):
#         for col_idx, value in enumerate(row):
#             if value == 2.220446049250313e-16:
#                 inverse_pose[row_idx][col_idx] = 0                
    return world_points

def sample_random_k(data, num_sample, num_total):
    indices = np.random.choice(num_total, num_sample, replace=False)
    return data[indices]

def matches_attention(attention_map, matches, k):
    """
    Parameters : 
        attention_map : HxWx1 of proabilities of a good keypoint location
    Returns : 
        Chosen flownet matches
    """
    code.interact(local=locals())
    hw = attention_map.shape
    attention_map_flat = attention_map.flatten()
    top_k_ind = np.argpartition(attention_map_flat, -k)[-k:]
    top_k_yx = np.array([np.array([int(ind/w), ind%w]) for ind in top_k_ind])
    assert RuntimeError('Not finished implementing this yet')
    
    pass

def dense_sparse_hybrid_correspondences(image1_keypoints, image2_keypoints, flownet_matches, superpoint_matches, num_matches, attention_map=None, superpoint_ratio=0.5):
    """
    Finds the flownet matches that lie on the keypoints from given keypoints. Fills in remaining num_matches with split given ratio of dense and sparse correspondences
    
    image1_keypoints : keypoints from image1 (n x 2)
    image2_keypoints : keypoints from image2 (n x 2)
    flownet_matches : 2d-2d matches between image1 and image2, respectively(n x 4) 
    superpoint_matches : 2d-2d matches between image1 and image2, respectively(n x 4) 
    num_matches : number of 2d-2d matches to output
    flownet_ratio : ratio of remaining matches to fill in using the flownet dense matches
    """
    print(f'keypoints : {image1_keypoints.shape[0] + image2_keypoints.shape[0]}, superpoint matches : {superpoint_matches.shape[0]}, flownet : {flownet_matches.shape[0]}')
        
    matches = []
    
    current_start_index = 0
        
    
#     #image1 keypoints
#     common_matches_image1, flownet_matches = get_flownet_matches_from_superpoint_keypoints(image1_keypoints, flownet_matches, image_keypoints=1)
#     if common_matches_image1.shape[0] > 0:
#         matches.extend(common_matches_image1)
    
#     #image2 keypoints 
#     common_matches_image2, flownet_matches = get_flownet_matches_from_superpoint_keypoints(image2_keypoints, flownet_matches, image_keypoints=2)
#     if common_matches_image2.shape[0] > 0:
#         matches.extend(common_matches_image2)
    
    
#     print(f'number of hybrid matches : {len(matches)}')
    
    
    #Fill with random choices from remaining flownet and superoints matches
    temp_num_matches = int((num_matches-len(matches)) * superpoint_ratio)
    #get superpoint correspondences
    if superpoint_matches.shape[0] <= temp_num_matches:
        matches.extend(superpoint_matches)
        print(f'Number of superpoint matches used : {superpoint_matches.shape[0]}')
    else:
        #I recognize here that I still possibly have the superpoint matches that I chose in "common_matches"
        superpoint_indices = np.random.choice(superpoint_matches.shape[0], temp_num_matches, replace=False)
        matches.extend(superpoint_matches[superpoint_indices])
        print(f'Number of superpoint matches used : {temp_num_matches}')

    
    #get half flownet correspondences
    temp_num_matches = num_matches - len(matches)
    
    
    if flownet_matches.shape[0] <= temp_num_matches:
        matches.extend(flownet_matches)
        print(f'Number of flownet matches used : {flownet_matches.shape[0]}')
    else:
        if attention_map is not None: 
            matches.extend(matches_attention(attention_map, flownet_matches, temp_num_matches)
        else:
            flownet_indices = np.random.choice(flownet_matches.shape[0], temp_num_matches, replace=False)
            matches.extend(flownet_matches[flownet_indices])
            print(f'Number of flownet matches used : {temp_num_matches}')
    return np.array(matches)


def get_random_sequence():
    """
    For the kitti vo dataset
    """
    sequences = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
    return sequences[random.randint(0, len(sequences)-1)]
        

def get_flownet_matches_from_superpoint_keypoints(keypoints, matches, image_keypoints=1):
    """
    Given keypoints from superpoint, grab the correspondences that exist in those pixel locations
    
    Parameters : 
        keypoints : (N x 2) Superpoint Keypoints from image1 or image2
        matches : (N x 4) Correspondences from trianflow flownet 
        image_keypoints : which image we are getting the keypoints from
    Returns : 
        N x 4
    """
    
    start_time = datetime.utcnow()

    remaining_matches = copy.deepcopy(matches)
    chosen_indices = []
    
    match_points = []
    
    if image_keypoints == 1: 
        offset = 0
    else : 
        offset = 2
    
    for keypoint_idx, keypoint in enumerate(keypoints):
        for match_idx, match in enumerate(matches):
            if int(match[0 + offset]) == int(keypoint[0]) and int(match[1 + offset]) == int(keypoint[1]):
                match_points.append(match)
                chosen_indices.append(match_idx)
                break
                
    remaining_matches = np.delete(remaining_matches, chosen_indices, axis=0)
        
    end_time = datetime.utcnow()
    print(f'{end_time - start_time} to find hybrid matches in image {image_keypoints}')
    
    return np.array(match_points), remaining_matches


def get_2d_matches(descriptor_matches, image1_keypoints, image2_keypoints, num_matches=None):
    """
    Given a set of descriptor matches, finds the top num_matcheas with the highest score and returns those 2d-2d correspondences. 
    
    Parameters : 
        descriptor_matches : (N, 3) where each entry is a tuple of (image1_index, image2_index, score)
        image1_keypoints, image2_keypoints : (N, 2). 2d pixel wise keypoints
        num_matches : the number of matches that we want to choose from, if it is less than descriptor_matches.shape[0]
        
    Returns : 
        M x 4 : (image1_x, image1_y, image2_x, image2_y) where M is num_matches or len(descriptor_matches)
    """
    image1_keypoints = toNumpy(image1_keypoints)
    image2_keypoints = toNumpy(image2_keypoints)


    if num_matches:
        sort_index = np.argsort(descriptor_matches[:, 2])
        

        if (len(sort_index) > num_matches):
            sort_index = sort_index[-num_matches:]
            
        match_pts = np.zeros((len(sort_index), 4))

        for idx, match in enumerate(sort_index):
            match_indices = descriptor_matches[match]

            match_pts[idx][:2] = image1_keypoints[int(match_indices[0])].astype(int)
            match_pts[idx][2:] = image2_keypoints[int(match_indices[1])].astype(int)
    else:
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
     

def get_configs(path, mode='superflow'): 
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

    if mode == 'superflow':
        model_cfg = { 'trianflow' : trianflow_cfg, 'superpoint' : cfg['models']['superpoint']}
    elif mode == 'siftflow':
        model_cfg = { 'trianflow' : trianflow_cfg}
    elif mode == 'superglueflow': 
        model_cfg = { 'trianflow' : trianflow_cfg, 'superpoint' : cfg['models']['superpoint'], 'superglue' : cfg['models']['superglue']}
    elif mode == 'attention':
        model_cfg = { 'trianflow' : trianflow_cfg, 'superpoint' : cfg['models']['superpoint'], 'superglue' : cfg['models']['superglue'], 'attention' : cfg['models']['attention']}
        

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