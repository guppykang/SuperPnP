#!/usr/bin/env python 

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
import logging
import code

from collections import OrderedDict

from utils.utils import get_configs, vehicle_to_world
from utils.logging import *

from infer_kitti import infer_vo, save_traj

warnings.filterwarnings("ignore")
    
class infer_deepF(infer_vo): # edited from infer_tum
    def __init__(self, seq_id, sequences_root_dir):
        super().__init__(seq_id, sequences_root_dir)
        # self.img_dir = sequences_root_dir
        # #self.img_dir = '/home4/zhaow/data/kitti_odometry/sampled_s4_sequences/'
        # self.seq_id = seq_id
        # self.raw_img_h = 480.0 #320
        # self.raw_img_w = 640.0 #1024
        # self.new_img_h = 384 #320
        # self.new_img_w = 512 #1024
        # self.max_depth = 50.0
        # self.min_depth = 0.0
        # self.cam_intrinsics = self.rescale_camera_intrinsics(self.read_calib_file())
        # self.flow_pose_ransac_thre = 0.1 #0.2
        # self.flow_pose_ransac_times = 10 #5
        # self.flow_pose_min_flow = 5
        # self.align_ransac_min_samples = 3
        # self.align_ransac_max_trials = 100
        # self.align_ransac_stop_prob = 0.99
        # self.align_ransac_thre = 1.0
        # self.PnP_ransac_iter = 1000
        # self.PnP_ransac_thre = 1
        # self.PnP_ransac_times = 5
        # self.train_sets = [ # only process train_set
        #     "rgbd_dataset_freiburg3_long_office_household",
        #     "rgbd_dataset_freiburg3_long_office_household_validation",
        #     "rgbd_dataset_freiburg3_sitting_xyz",
        #     "rgbd_dataset_freiburg3_structure_texture_far",
        #     "rgbd_dataset_freiburg3_structure_texture_near",
        #     "rgbd_dataset_freiburg3_teddy",
        #     ]
        # self.test_sets = [
        #     "rgbd_dataset_freiburg3_walking_xyz",
        #     "rgbd_dataset_freiburg3_large_cabinet_validation",
        #     ]
    
    
    # def read_calib_file(self):
    #     """ # directly from the website
    #     https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect

    #     """
    #     calib = np.identity(3)
    #     fu, fv, cu, cv =  535.4, 539.2, 320.1, 247.6
    #     calib = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]])
    #     # D = np.array([0,0,0,0,0])
    #     # height, width, calib, D = self.load_intrinsics(calib_data)
    #     # calib = proj_c2p[0:3, 0:3]
    #     # intrinsics_original = calib + 0
    #     # calib[0,:] *=  zoom_x
    #     # calib[1,:] *=  zoom_y
    #     # print(f"calib: {calib}, intrinsics_original: {intrinsics_original}")
    #     return calib
    
    # # @staticmethod
    # def read_images_files_from_folder(self, path_to_sequence):
    #     rgb_filenames = []
    #     timestamps = []
    #     # path_to_sequence = f"{dataset_dir}/{sequence}"
    #     with open(f"{path_to_sequence}/rgb.txt") as times_file:
    #         for line in times_file:
    #             if len(line) > 0 and not line.startswith('#'):
    #                 t, rgb = line.rstrip().split(' ')[0:2]
    #                 rgb_filenames.append(f"{path_to_sequence}/{rgb}")
    #                 timestamps.append(float(t))
    #     test_files = rgb_filenames
    #     timestamps = np.array(timestamps)
    #     return test_files, timestamps
    
    # def load_images(self, max_length=-1):
    #     print(f'Loading images from sequence {self.seq_id}')
    #     path = self.img_dir
    #     seq = self.seq_id
    #     new_img_h = self.new_img_h
    #     new_img_w = self.new_img_w
    #     test_files, timestamps = self.read_images_files_from_folder(f"{path}/{seq}")
    #     self.timestamps = timestamps
    #     # seq_dir = os.path.join(path, seq)
    #     # image_dir = os.path.join(seq_dir, 'image_2')
    #     num = len(test_files)
    #     if max_length > 0:
    #         num = min(int(max_length)+1, num)
        
    #     images = []
    #     for i in tqdm(range(num)):
    #         image = cv2.imread(test_files[i])
    #         image = cv2.resize(image, (new_img_w, new_img_h))
    #         images.append(image)

    #     print('Loaded Images')
    #     return images
    
    @staticmethod
    def mat2quat(mat):
        assert mat.shape == (3,4) or mat.shape == (4,4)
        rotation = mat[:3,:3]
        trans = mat[:3,3]
        from scipy.spatial.transform import Rotation as R
        qua = R.from_matrix(rotation)
        vect = np.concatenate((trans, qua.as_quat() ), axis=0)
        return vect
        
    @property
    def deepF_fe(self):
        print("get deepF")
        return self._deepF_fe

    @deepF_fe.setter
    def deepF_fe(self, fe):
        print("set deepF frontend")
        self._deepF_fe = fe

    def solve_pose_deepF(self, xy1, xy2):
        """ call deepF front end for pose estimation
        """
        # assert model is ready
        assert self.deepF_fe is not None
        # get K, K_inv
        b_K = torch.tensor(self.K_np).float().unsqueeze(0)
        b_K_inv = torch.tensor(self.K_inv_np).float().unsqueeze(0)
        b_xy1 = torch.tensor(xy1).float().unsqueeze(0)
        b_xy2 = torch.tensor(xy2).float().unsqueeze(0)
        # inference
        # SVD for pose
        poses = self.deepF_fe.run(b_xy1, b_xy2, b_K, b_K_inv)
        pose = poses.squeeze().to('cpu').numpy()
        row = np.array([[0,0,0,1]]).astype(np.float32)
        pose = np.concatenate((pose, row), axis=0)
        return pose

    def solve_pose_flow(self, xy1, xy2):
        return self.solve_pose_deepF(xy1, xy2)
    
    def save_traj(self, traj_txt, poses, save_time, model):
        time_stamps = self.timestamps
        time_stamps = np.array(time_stamps).flatten()
        time_stamps = time_stamps[:len(poses)].reshape(-1,1)
        
        poses_wTime = np.concatenate((time_stamps, poses), axis=1)
        # dir
        traj_dir = Path(f"{traj_txt}")
        traj_dir = traj_dir/f"{self.seq_id}"/f"{model}"
        traj_dir.mkdir(exist_ok=True, parents=True)
        
        # save txt
        filename = Path(f"{traj_dir}/preds_{save_time}.txt")
        np.savetxt(filename, poses, delimiter=" ", fmt="%.4f")
        filename = Path(f"{traj_dir}/preds_{save_time}_t.txt")
        np.savetxt(filename, poses_wTime, delimiter=" ", fmt="%.4f")
        ## save tum txt
        filename = Path(f"{traj_dir}/preds_{save_time}.tum")
        pose_qua = np.array([infer_vo_tum.mat2quat(m.reshape(3,4)) for m in poses])
        poses_qua_wTime = np.concatenate((time_stamps, pose_qua), axis=1)
        np.savetxt(filename, poses_qua_wTime, delimiter=" ", fmt="%.4f")
        # copy tum txt
        filename = Path(f"{traj_dir}/preds.tum")
        np.savetxt(filename, poses_qua_wTime, delimiter=" ", fmt="%.4f")
        pass

##### deepF frontend
from deepFEPE.utils.loader import (
    dataLoader,
    modelLoader,
    pretrainedLoader_net,
    pretrainedLoader_opt,
)

class deepF_frontend(object):
    def __init__(self, config, device='cpu'):
        self.device = device
        self.config = config
        img_zoom_xy = (
            config["data"]["preprocessing"]["resize"][1]
            / config["data"]["image"]["size"][1],
            config["data"]["preprocessing"]["resize"][0]
            / config["data"]["image"]["size"][0],
        )
        self.model_params = {
            "depth": config["model"]["depth"],
            "img_zoom_xy": img_zoom_xy,
            "image_size": config["data"]["image"]["size"],
            "quality_size": config["model"]["quality_size"],
            "if_quality": config["model"]["if_quality"],
            "if_img_des_to_pointnet": config["model"]["if_img_des_to_pointnet"],
            "if_goodCorresArch": config["model"]["if_goodCorresArch"],
            "if_img_feat": config["model"]["if_img_feat"],
            "if_cpu_svd": config["model"]["if_cpu_svd"],
            "if_learn_offsets": config["model"]["if_learn_offsets"],
            "if_tri_depth": config["model"]["if_tri_depth"],
            "if_sample_loss": config["model"]["if_sample_loss"],
        }

        pass

    def load_model(self):
        self.net = modelLoader(self.config["model"]["name"], **self.model_params)
        print(f"deepF net: {self.net}")
        pass

    def prepare_model(self):
        from deepFEPE.train_good import prepare_model
        n_iter = 0
        n_iter_val = 0 + n_iter
        ## load pretrained and create optimizer
        net, optimizer, n_iter, n_iter_val = prepare_model(
            self.config, self.net, self.device, n_iter, n_iter_val, net_postfix=""
        )
        self.net, self.optimizer, self.n_iter, self.n_iter_val = net, optimizer, n_iter, n_iter_val
        pass

    
    '''
    deepFEPE: 
        need a function to do pure inference w/o GT
        assume we won't have GT
    superpoint:
        need to install the package

    '''

    def process_output(self, outs):
        F_est_normalized, T1, T2, out_layers, residual_layers, weights_layers = (
            outs["F_est"],
            outs["T1"],
            outs["T2"],
            outs["out_layers"],
            outs["residual_layers"],
            outs["weights_layers"],
        )
        F_ests = (
            T2.permute(0, 2, 1) @ F_est_normalized @ T1
        )  # If use norm_K, then the output F_est is esentially E_ests, and the next line basically transforms it back: E_ests == Ks.transpose(1, 2) @ {T2.permute(0,2,1).bmm(F_est.bmm(T1))} @ Ks
        E_ests = Ks.transpose(1, 2) @ F_ests @ Ks


    def run(self, b_xy1, b_xy2, Ks, K_invs, train=False):
        from deepFEPE.train_good_utils import get_E_ests_deepF, mat_E_to_pose
        from deepFEPE.dsac_tools.utils_F import _get_M2s, _E_to_M_train
        # Make data batch
        matches_use_ori = torch.cat((b_xy1, b_xy2), 2).cuda()

        data_batch = {
            # "matches_xy": matches_use_normalizedK,
            "matches_xy_ori": matches_use_ori,
            "quality": None,
            # "x1_normalizedK": x1_normalizedK,
            # "x2_normalizedK": x2_normalizedK,
            "Ks": Ks,
            "K_invs": K_invs,
            "des1": None,
            "des2": None,
            "matches_good_unique_nums": b_xy1.shape[1], ## 
            # "t_scene_scale": t_scene_scale,
            # "frame_ids": sample["frame_ids"],
        }
        idx = 0 # first element in the batch
        with torch.no_grad():
            outs = self.net(data_batch)
            # get essential matrix
            E_ests_layers = get_E_ests_deepF(outs, Ks.to(self.device), K_invs.to(self.device)) # [D, B, 3, 3]
            # get R, t
            # results = mat_E_to_pose(E_ests_layers, idx=-1, device=self.device)
            # # R12s_batch_cam -> [[B,3,3], [B,3,3] ], t12s_batch_cam -> [[B,3,1], [B,3,1] ]
            # R12s_batch_cam, t12s_batch_cam = results[0], results[1] 
            # b_pose = torch.cat((R12s_batch_cam[1], t12s_batch_cam[0]), dim=2)
            # return b_pose

            # done Cheirality check            
            # R2s, t2s, M2s = _get_M2s(E_ests_layers[-1][0]) # double check layer
            Ks_np = Ks.numpy()
            b_xy1_np = b_xy1.numpy()
            b_xy2_np = b_xy2.numpy()
            M2_list, error_Rt, Rt_cam = _E_to_M_train(E_ests_layers[-1][idx], Ks_np[idx], b_xy1_np[idx], 
                        b_xy2_np[idx], show_result=True)
            print(f"Rt_cam: {Rt_cam}")
            return Rt_cam

            # pick one of the pose ...
            # import code; code.interact(local=locals())









if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(
        description="Inferencing on TUM pipeline."
    )
    arg_parser.add_argument('--model', type=str, default='superglueflow', help='(choose from : siftflow, superglueflow, superflow, superflow2)')
    arg_parser.add_argument('--traj_save_dir', type=str, default='/jbk001-data1/datasets/tum/vo_pred', help='directory for saving results')
    arg_parser.add_argument('--sequences_root_dir', type=str, default='/jbk001-data1/datasets/tum', help='Root directory for all datasets')
    arg_parser.add_argument('--sequence', type=str, default='rgbd_dataset_freiburg2_desk', help='Test sequence id.')
    arg_parser.add_argument('--iters', type=int, default='-1', help='Limited iterations for debugging')
    args = arg_parser.parse_args()
    
   

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("train on device: %s", device)

    #import the model
    print(f'Using the {args.model} model')
    if args.model == 'superflow':
        config_file = './configs/superflow.yaml'
        model_cfg, cfg = get_configs(config_file, mode='superflow')    
        from models.superflow import SuperFlow as Model
    elif args.model == 'superflow2':
        config_file = './configs/tum/superflow2.yaml'
        model_cfg, cfg = get_configs(config_file, mode='superflow')    
        from models.superflow2 import SuperFlow as Model
    elif args.model == 'siftflow':
        config_file = './configs/siftflow_deepF.yaml'
        model_cfg, cfg = get_configs(config_file, mode='siftflow')    
        from models.siftflow import SiftFlow as Model
    elif args.model == 'siftflow_scsfm':
        config_file = './configs/siftflow_scsfm.yaml'
        model_cfg, cfg = get_configs(config_file)    
        from models.siftflow_scsfm import SiftFlow_scsfm as Model
    elif args.model == 'superglueflow_scsfm':
        config_file = './configs/superglueflow_scsfm.yaml'
        model_cfg, cfg = get_configs(config_file, mode='superglueflow')    
        from models.superglueflow_scsfm import SuperGlueFlow_scsfm as Model
    elif args.model == 'superglueflow':
        config_file = './configs/tum/superglueflow.yaml'
        model_cfg, cfg = get_configs(config_file, mode=args.model)    
        from models.superglueflow import SuperGlueFlow as Model
    elif args.model == 'trianflow':
        config_file = './configs/superflow.yaml'
        model_cfg, cfg = get_configs(config_file, mode='superflow')    
        from models.trianflow import TrianFlow as Model
    
    print(f"config model: {list(cfg['models'])}")

    #initialize the model
    model = Model(model_cfg, cfg)
    model.load_modules(model_cfg)
    model.cuda()
    model.eval()
    print('Model Loaded.')

    #dataset
    vo_test = infer_deepF(args.sequence, cfg["data"]["vo_path"])

    # load deepF model
    deepF_fe = deepF_frontend(cfg["models"]["deepF"], device=device)
    deepF_fe.load_model()
    deepF_fe.prepare_model()
    vo_test.deepF_fe = deepF_fe

    #load and inference
    images = vo_test.load_images(max_length=args.iters)
    print('Images Loaded. Total ' + str(len(images)) + ' images found.')
    print('Testing VO.')
    poses = np.array(vo_test.process_video_relative(images, model, args.model))
    del images
    print('Test completed.')

    save_time = time.strftime("%Y%m%d-%H%M%S")
    poses = poses[:,:3,:4].reshape(-1, 12)
    print(f'Shape of poses : {poses.shape}')
    vo_test.save_traj_kitti(args.traj_save_dir, poses, save_time, args.model)

    # save_time = time.strftime("%Y%m%d-%H%M%S")
    # poses = poses[:,:3,:4].reshape(-1, 12)
    # print(f'Shape of poses : {poses.shape}')
    # vo_test.save_traj(args.traj_save_dir, poses, save_time, args.model)
    # print(f'Predicted Trajectory saved at : {args.traj_save_dir}/{args.sequence}/{args.model}/preds_{save_time}.txt')


  