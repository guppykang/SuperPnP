import os, sys
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
from superpoint.utils.var_dim import toNumpy
from TrianFlow.core.visualize.visualizer import *
from TrianFlow.core.visualize.profiler import Profiler

from infer_kitti import projection, unprojection, cv_triangulation

warnings.filterwarnings("ignore")

class train_vo():
    def __init__(self):
        self.raw_img_h = 370.0#320
        self.raw_img_w = 1226.0#1024
        self.new_img_h = 256#320
        self.new_img_w = 832#1024
        self.max_depth = 50.0
        self.min_depth = 0.0
        self.flow_pose_ransac_thre = 0.1 #0.2
        self.flow_pose_ransac_times = 10 #5
        self.flow_pose_min_flow = 5
        self.align_ransac_min_samples = 3
        self.align_ransac_max_trials = 100
        self.align_ransac_stop_prob = 0.99
        self.align_ransac_thre = 1.0
        self.PnP_ransac_iter = 1000
        self.PnP_ransac_thre = 1
        self.PnP_ransac_times = 5
    
    
    def get_prediction(self, images, images_gray, model, K, K_inv):
        outs = model.get_match_outs(images, images_gray, K, K_inv)[0]
        
        depth1 = outs['image1_depth'] # H, W
        depth2 = outs['image2_depth'] # H, W
        filt_depth_match = outs['matches'] # N x 4
        
        return filt_depth_match, depth1, depth2

    
    def process_video(self, images, images_gray, Ks, K_invs, model):
        '''
        Done in relative pose estimation fashion
        Process a sequence to get scale consistent trajectory results. 
        Register according to depth net predictions. Here we assume depth predictions have consistent scale.
        If not, pleas use process_video_tri which only use triangulated depth to get self-consistent scaled pose.
        '''
        poses = []
        absolute_pose_t = np.zeros((3, 4))
        global_pose = np.eye(4)
        # The first one global pose is origin.
        poses.append(copy.deepcopy(global_pose))
        seq_len = len(images)

        total_loss_scale = 0
        for i in range(seq_len-1):
            K = Ks[i]
            self.cam_intrinsics = toNumpy(K[0])
            K_inv = K_invs[i]
            h = int(images[i].shape[1]/2)
            w = int(images[i].shape[2])

                        
            depth_match, depth1, depth2 = self.get_prediction(images[i].unsqueeze(0), images_gray[i].unsqueeze(0), model, K.unsqueeze(0), K_inv.unsqueeze(0))
            
            rel_pose = np.eye(4)
            flow_pose, loss_scale, inliers = self.solve_pose_flow(depth_match[:,:2], depth_match[:,2:])
            gt_attention = self.get_gt_attention(inliers, depth_match, 'five_point', (h, w))
            rel_pose[:3,:3] = copy.deepcopy(flow_pose[:3,:3])
            if np.linalg.norm(flow_pose[:3,3:]) != 0:
                scale = self.align_to_depth(depth_match[:,:2], depth_match[:,2:], flow_pose, depth2)
                rel_pose[:3,3:] = flow_pose[:3,3:] * scale
            
            if np.linalg.norm(flow_pose[:3,3:]) == 0 or scale == -1 or inliers is not None:
                print('PnP '+str(i))
                pnp_pose, loss_scale, inliers = self.solve_relative_pose_pnp(depth_match[:,:2], depth_match[:,2:], depth1)
                gt_attention = self.get_gt_attention(inliers, depth_match, 'pnp', (h, w))
                rel_pose = pnp_pose
                
            global_pose[:3,3:] = np.matmul(global_pose[:3,:3], rel_pose[:3,3:]) + global_pose[:3,3:]
            global_pose[:3,:3] = np.matmul(global_pose[:3,:3], rel_pose[:3,:3])
            poses.append(copy.deepcopy(global_pose))
            total_loss_scale += loss_scale
            print(f'pose : {i}')
            
        return poses, total_loss_scale/seq_len, gt_attention

    def get_gt_attention(self, inliers, matches, mode, hw):
        """
        matches : In train_attention.py attention is predicted based on the first image
        """
        out = np.zeros(hw)
        if mode == 'pnp':
            for i in inliers:
                out[int(matches[i[0]][1])][int(matches[i[0]][0])] = 1
        elif mode == 'five_point':
            for idx, i in enumerate(inliers):
                if i[0] == 1:
                    out[int(matches[idx][1])][int(matches[idx][0])] = 1 #because of flownet subpixel accuracy
        
        return out
    
    def normalize_coord(self, xy, K):
        xy_norm = copy.deepcopy(xy)
        xy_norm[:,0] = (xy[:,0] - K[0,2]) / K[0,0]
        xy_norm[:,1] = (xy[:,1] - K[1,2]) / K[1,1]

        return xy_norm
    
    def align_to_depth(self, xy1, xy2, pose, depth2):
        # Align the translation scale according to triangulation depth
        # xy1, xy2: [N, 2] pose: [4, 4] depth2: [H, W]
        
        # Triangulation
        img_h, img_w = np.shape(depth2)[0], np.shape(depth2)[1]
        pose_inv = np.linalg.inv(pose)

        xy1_norm = self.normalize_coord(xy1, self.cam_intrinsics)
        xy2_norm = self.normalize_coord(xy2, self.cam_intrinsics)

        points1_tri, points2_tri = cv_triangulation(np.concatenate([xy1_norm, xy2_norm], axis=1), pose_inv)
        
        depth2_tri = projection(xy2, points2_tri, img_h, img_w)
        depth2_tri[depth2_tri < 0] = 0
        
        # Remove negative depths
        valid_mask = (depth2 > 0) * (depth2_tri > 0)
        depth_pred_valid = depth2[valid_mask]
        depth_tri_valid = depth2_tri[valid_mask]
        
        if np.sum(valid_mask) > 100:
            scale_reg = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(fit_intercept=False), min_samples=self.align_ransac_min_samples, \
                        max_trials=self.align_ransac_max_trials, stop_probability=self.align_ransac_stop_prob, residual_threshold=self.align_ransac_thre)
            scale_reg.fit(depth_tri_valid.reshape(-1, 1), depth_pred_valid.reshape(-1, 1))
            scale = scale_reg.estimator_.coef_[0, 0]
        else:
            scale = -1

        return scale
    
    def solve_relative_pose_pnp(self, xy1, xy2, depth1):
        # Use pnp to solve relative poses.
        # xy1, xy2: [N, 2] depth1: [H, W]

        img_h, img_w = np.shape(depth1)[0], np.shape(depth1)[1]
        
        # Ensure all the correspondences are inside the image.
        x_idx = (xy2[:, 0] >= 0) * (xy2[:, 0] < img_w)
        y_idx = (xy2[:, 1] >= 0) * (xy2[:, 1] < img_h)
        idx = y_idx * x_idx
        xy1 = xy1[idx]
        xy2 = xy2[idx]

        xy1_int = xy1.astype(np.int)
        sample_depth = depth1[xy1_int[:,1], xy1_int[:,0]]
        valid_depth_mask = (sample_depth < self.max_depth) * (sample_depth > self.min_depth)

        xy1 = xy1[valid_depth_mask]
        xy2 = xy2[valid_depth_mask]

        # Unproject to 3d space
        points1 = unprojection(xy1, sample_depth[valid_depth_mask], self.cam_intrinsics)

        # ransac
        best_rt = []
        max_inlier_num = 0
        max_ransac_iter = self.PnP_ransac_times
        loss_scale = 0
        best_inliers = None
        for i in range(max_ransac_iter):
            if xy2.shape[0] > 4:
                flag, r, t, inlier = cv2.solvePnPRansac(objectPoints=points1, imagePoints=xy2, cameraMatrix=self.cam_intrinsics, distCoeffs=None, iterationsCount=self.PnP_ransac_iter, reprojectionError=self.PnP_ransac_thre)
                if flag and inlier.shape[0] > max_inlier_num:
                    best_rt = [r, t]
                    max_inlier_num = inlier.shape[0]
                    best_inliers = inlier
        pose = np.eye(4)
        loss_scale = max_inlier_num/xy1.shape[0]

        if len(best_rt) != 0:
            r, t = best_rt
            pose[:3,:3] = cv2.Rodrigues(r)[0]
            pose[:3,3:] = t
        pose = np.linalg.inv(pose)
        return pose, loss_scale, best_inliers
    
    
    def solve_pose_flow(self, xy1, xy2):
        # Solve essential matrix to find relative pose from flow.

        # ransac
        best_rt = []
        max_inlier_num = 0
        max_ransac_iter = self.flow_pose_ransac_times
        best_inliers = None
        pp = (self.cam_intrinsics[0,2], self.cam_intrinsics[1,2])
        
        # flow magnitude
        avg_flow = np.mean(np.linalg.norm(xy1 - xy2, axis=1))
        if avg_flow > self.flow_pose_min_flow:
            for i in range(max_ransac_iter):
                E, inliers = cv2.findEssentialMat(xy2, xy1, focal=self.cam_intrinsics[0,0], pp=pp, method=cv2.RANSAC, prob=0.99, threshold=self.flow_pose_ransac_thre)
                cheirality_cnt, R, t, _ = cv2.recoverPose(E, xy2, xy1, focal=self.cam_intrinsics[0,0], pp=pp)
                if inliers.sum() > max_inlier_num and cheirality_cnt > 50:
                    best_rt = [R, t]
                    max_inlier_num = inliers.sum()
                    best_inliers = inliers
            if len(best_rt) == 0:
                R = np.eye(3)
                t = np.zeros((3,1))
                best_rt = [R, t]
                
            loss_scale = max_inlier_num/inliers.shape[0]

        else:
            R = np.eye(3)
            t = np.zeros((3,1))
            best_rt = [R, t]
            loss_scale = 0
            
        R, t = best_rt
        pose = np.eye(4)
        pose[:3,:3] = R
        pose[:3,3:] = t
        
        
        return pose, loss_scale, best_inliers
