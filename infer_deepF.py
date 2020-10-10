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

from infer_vo import infer_vo, save_traj
from infer_models import infer_vo_kitti, infer_vo_tum

warnings.filterwarnings("ignore")

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
        logging.debug(f"deepF net: {self.net}")
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
            logging.debug(f"Rt_cam: {Rt_cam}")
            return Rt_cam


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(
        description="Inferencing on TUM pipeline."
    )
    arg_parser.add_argument('--model', type=str, default='superglueflow', help='(choose from : siftflow, superglueflow, superflow, superflow2)')
    arg_parser.add_argument(
        "-d", "--dataset", type=str, default="kitti", help="[kitti |  euroc | tum ... ]"
    )
    arg_parser.add_argument('--traj_save_dir', type=str, default='/jbk001-data1/datasets/tum/vo_pred', help='directory for saving results')
    arg_parser.add_argument('--sequences_root_dir', type=str, default='/jbk001-data1/datasets/tum', help='Root directory for all datasets')
    arg_parser.add_argument('--sequence', type=str, default='rgbd_dataset_freiburg2_desk', help='Test sequence id.')
    arg_parser.add_argument('--iters', type=int, default='-1', help='Limited iterations for debugging')
    # arg_parser.add_argument("--deepF", action="store_true", help="Use DeepF pipeline")

    args = arg_parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("train on device: %s", device)

    #import the model
    print(f'Using the {args.model} model')
    if args.model == 'superflow':
        config_file = './configs/superflow.yaml'
        model_cfg, cfg = get_configs(config_file, mode='superflow')    
        from models.superflow import SuperFlow as Model
    elif args.model == 'superflow2':
        raise "config to be checked!!!"
        #config_file = './configs/tum/superflow2.yaml'
        config_file = './configs/superflow2.yaml'
        model_cfg, cfg = get_configs(config_file, mode='superflow')    
        from models.superflow2 import SuperFlow as Model
    elif args.model == 'siftflow' or args.model == 'siftflow_deepF' :
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
        #config_file = './configs/tum/superglueflow.yaml'
        config_file = './configs/superglueflow.yaml'
        model_cfg, cfg = get_configs(config_file, mode=args.model)    
        from models.superglueflow import SuperGlueFlow as Model
    elif args.model == 'trianflow':
        config_file = './configs/superflow.yaml'
        model_cfg, cfg = get_configs(config_file, mode='superflow')    
        from models.trianflow import TrianFlow as Model
    
    print(f"configs: {cfg}")
    print(f"config model: {list(cfg['models'])}")

    #initialize the model
    model = Model(model_cfg, cfg)
    model.load_modules(model_cfg)
    model.cuda()
    model.eval()
    print('Model Loaded.')

    #dataset
    if_pnp = cfg['models'].get("if_pnp", True)
    if_deepF = cfg['models'].get("if_deepF", False)

    if args.dataset == 'kitti':
        infer_vo = infer_vo_kitti
    elif args.dataset == 'tum':
        infer_vo = infer_vo_tum
    else:
        raise "dataset not defined"
    vo_test = infer_vo(args.sequence, cfg["data"][f"vo_path_{args.dataset}"], if_pnp, if_deepF)
    # load deepF model
    if if_deepF:
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
    vo_test.save_traj(args.traj_save_dir, poses, save_time, args.model)


  