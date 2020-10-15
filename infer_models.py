#!/usr/bin/env python 
""" infer_models for kitti and tum dataset

"""

import os, sys
from TrianFlow.core.visualize.visualizer import *
from TrianFlow.core.visualize.profiler import Profiler
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from sklearn import linear_model
import yaml
import warnings
import code
from tqdm import tqdm
import copy
from pathlib import Path
import time

from collections import OrderedDict

from utils.utils import get_configs, vehicle_to_world

from infer_vo import infer_vo

warnings.filterwarnings("ignore")
    
class infer_vo_kitti(infer_vo):
    def __init__(self, seq_id, sequences_root_dir, if_pnp=True, if_deepF=False):
        super().__init__(seq_id, sequences_root_dir, if_pnp, if_deepF)
        self.raw_img_h = 370.0#320
        self.raw_img_w = 1226.0#1024
        self.new_img_h = 256#320
        self.new_img_w = 832#1024
        self.cam_intrinsics = self.read_rescale_camera_intrinsics(os.path.join(self.img_dir, seq_id) + '/calib.txt')

    
class infer_vo_tum(infer_vo):
    def __init__(self, seq_id, sequences_root_dir, if_pnp=True, if_deepF=False):
        super().__init__(seq_id, sequences_root_dir, if_pnp, if_deepF)
        self.img_dir = sequences_root_dir
        #self.img_dir = '/home4/zhaow/data/kitti_odometry/sampled_s4_sequences/'
        #self.seq_id = seq_id
        self.raw_img_h = 480.0 #320
        self.raw_img_w = 640.0 #1024
        self.new_img_h = 384 #320
        self.new_img_w = 512 #1024
        #self.max_depth = 50.0
        #self.min_depth = 0.0
        self.cam_intrinsics = self.rescale_camera_intrinsics(self.read_calib_file())

        self.train_sets = [ # only process train_set
            "rgbd_dataset_freiburg3_long_office_household",
            "rgbd_dataset_freiburg3_long_office_household_validation",
            "rgbd_dataset_freiburg3_sitting_xyz",
            "rgbd_dataset_freiburg3_structure_texture_far",
            "rgbd_dataset_freiburg3_structure_texture_near",
            "rgbd_dataset_freiburg3_teddy",
            ]
        self.test_sets = [
            "rgbd_dataset_freiburg3_walking_xyz",
            "rgbd_dataset_freiburg3_large_cabinet_validation",
            ]
    
    
    def read_calib_file(self):
        """ # directly from the website
        https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect

        """
        calib = np.identity(3)
        fu, fv, cu, cv =  535.4, 539.2, 320.1, 247.6
        calib = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]])
        # D = np.array([0,0,0,0,0])
        # height, width, calib, D = self.load_intrinsics(calib_data)
        # calib = proj_c2p[0:3, 0:3]
        # intrinsics_original = calib + 0
        # calib[0,:] *=  zoom_x
        # calib[1,:] *=  zoom_y
        # print(f"calib: {calib}, intrinsics_original: {intrinsics_original}")
        return calib
    
    # @staticmethod
    def read_images_files_from_folder(self, path_to_sequence):
        rgb_filenames = []
        timestamps = []
        # path_to_sequence = f"{dataset_dir}/{sequence}"
        with open(f"{path_to_sequence}/rgb.txt") as times_file:
            for line in times_file:
                if len(line) > 0 and not line.startswith('#'):
                    t, rgb = line.rstrip().split(' ')[0:2]
                    rgb_filenames.append(f"{path_to_sequence}/{rgb}")
                    timestamps.append(float(t))
        test_files = rgb_filenames
        timestamps = np.array(timestamps)
        return test_files, timestamps
    
    def load_images(self, max_length=-1):
        print(f'Loading images from sequence {self.seq_id}')
        path = self.img_dir
        seq = self.seq_id
        new_img_h = self.new_img_h
        new_img_w = self.new_img_w
        test_files, timestamps = self.read_images_files_from_folder(f"{path}/{seq}")
        self.timestamps = timestamps
        # seq_dir = os.path.join(path, seq)
        # image_dir = os.path.join(seq_dir, 'image_2')
        num = len(test_files)
        if max_length > 0:
            num = min(int(max_length)+1, num)
        
        images = []
        for i in tqdm(range(num)):
            image = cv2.imread(test_files[i])
            image = cv2.resize(image, (new_img_w, new_img_h))
            images.append(image)

        print('Loaded Images')
        return images
    
    @staticmethod
    def mat2quat(mat):
        assert mat.shape == (3,4) or mat.shape == (4,4)
        rotation = mat[:3,:3]
        trans = mat[:3,3]
        from scipy.spatial.transform import Rotation as R
        qua = R.from_matrix(rotation)
        vect = np.concatenate((trans, qua.as_quat() ), axis=0)
        return vect
        
    
    def save_traj(self, traj_save_dir, poses, save_time, model):
        if self.timestamps is not None:
            time_stamps = self.timestamps
            time_stamps = np.array(time_stamps).flatten()
            time_stamps = time_stamps[:len(poses)].reshape(-1,1)
            
            poses_wTime = np.concatenate((time_stamps, poses), axis=1)
        else:
            poses_wTime = poses

        traj_dir = self.save_traj_kitti(traj_save_dir, poses, save_time, model)

        ## save tum txt
        filename = Path(f"{traj_dir}/preds_{save_time}.tum")
        pose_qua = np.array([infer_vo_tum.mat2quat(m.reshape(3,4)) for m in poses])
        poses_qua_wTime = np.concatenate((time_stamps, pose_qua), axis=1)
        np.savetxt(filename, poses_qua_wTime, delimiter=" ", fmt="%.4f")
        # copy tum txt
        filename = Path(f"{traj_dir}/preds.tum")
        np.savetxt(filename, poses_qua_wTime, delimiter=" ", fmt="%.4f")
        print(f'Predicted (TUM) Trajectory saved at : {filename}')
        pass

