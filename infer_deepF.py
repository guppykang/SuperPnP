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

# deepFEPE
from deepFEPE.train_good_utils import get_E_ests_deepF, mat_E_to_pose
from deepFEPE.dsac_tools.utils_F import _get_M2s, _E_to_M_train

warnings.filterwarnings("ignore")

##### deepF frontend
from deepFEPE.utils.loader import (
    dataLoader,
    modelLoader,
    pretrainedLoader_net,
    pretrainedLoader_opt,
)

class deepF_frontend(torch.nn.Module):
    def __init__(self, config, device='cpu'):
        super(deepF_frontend, self).__init__()
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
        self.load_model()
        self.prepare_model()
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
        """
        """
        # Make data batch
        matches_use_ori = torch.cat((b_xy1, b_xy2), 2).cuda()

        data_batch = {
            "matches_xy_ori": matches_use_ori,
            "quality": None,
            "Ks": Ks,
            "K_invs": K_invs,
            "des1": None,
            "des2": None,
            "matches_good_unique_nums": b_xy1.shape[1], ## 
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

    def compute_epipolar_loss(self, fmat, match, mask=None):
        # fmat: [b, 3, 3] match: [b, 4, h*w] mask: [b,1,h*w]
        num_batch = match.shape[0]
        match_num = match.shape[-1]

        points1 = match[:,:2,:]
        points2 = match[:,2:,:]
        ones = torch.ones(num_batch, 1, match_num).to(points1.get_device())
        points1 = torch.cat([points1, ones], 1) # [b,3,n]
        points2 = torch.cat([points2, ones], 1).transpose(1,2) # [b,n,3]

        # compute fundamental matrix loss
        fmat = fmat.unsqueeze(1)
        fmat_tiles = fmat.view([-1,3,3])
        epi_lines = fmat_tiles.bmm(points1) #[b,3,n]  [b*n, 3, 1]
        dist_p2l = torch.abs((epi_lines.permute([0, 2, 1]) * points2).sum(-1, keepdim=True)) # [b,n,1]
        a = epi_lines[:,0,:].unsqueeze(1).transpose(1,2) # [b,n,1]
        b = epi_lines[:,1,:].unsqueeze(1).transpose(1,2) # [b,n,1]
        dist_div = torch.sqrt(a*a + b*b) + 1e-6
        dist_map = dist_p2l / dist_div # [B, n, 1]
        
        if mask is None:
            loss = dist_map.mean([1,2])
        else:
            loss = (dist_map * mask.transpose(1,2)).mean([1,2]) / mask.mean([1,2])
        return loss, dist_map
        
    def compute_reprojection_loss(self, b_xyz1, b_xyz2, Ks, K_invs, Rt_cam, mask=None, clamp=5):
        """ 
        params:
            b_xyz1, b_xyz2: [b, N, 3]
            Ks: [b, 3, 3]
            K_invs: [b, 3, 3]
            Rt_cam: tensor [b, 3, 4]
            mask: [b,N,1]
        """
        # normalize b_xyz
        # [xyz] to homogeneous [x,y,z,1] -> [b, N, 4]
        #code.interact(local = locals())
        b_xyz1_norm = K_invs.bmm(b_xyz1.transpose(1,2)).transpose(1,2)  # [b, N, 3]
        b_xyz2_norm = K_invs.bmm(b_xyz2.transpose(1,2)).transpose(1,2)

        # Rt_cam to 4x4 matrix (P)
        # loss = | P*X1 - X2 |
        # transform = tgm.ConvertPointsToHomogeneous()
        # b_homo1 = transform(b_xyz1_norm) # [b, N, 4]
        def ConvertPointsToHomogeneous(points):
            """ [b, N, ch] -> [b, N, ch+1]
            """
            num_batch, match_num = points.shape[0], points.shape[1]
            ones = torch.ones(num_batch, match_num, 1).to(points.get_device())
            points = torch.cat([points, ones], 2) # [b,n,ch+1]
            return points
        b_homo1 = ConvertPointsToHomogeneous(b_xyz1_norm)
        b_warp1 = Rt_cam.bmm(b_homo1.transpose(1,2)).transpose(1,2) # [b, N, 3]
        #b_homo2 = ConvertPointsToHomogeneous(b_xyz2_norm)
        #b_warp2 = Rt_cam.bmm(b_homo2.transpose(1,2)).transpose(1,2) # [b, N, 3]
        b_proj1 = Ks.bmm(b_warp1.transpose(1,2)).transpose(1,2)

        dist_map = torch.abs(b_warp1 - b_xyz2_norm)
        dist_map = torch.clamp(dist_map, max=clamp) # change that to config
        if mask is None:
            loss = dist_map.mean([1,2])
        else:
            loss = (dist_map * mask).mean([1,2]) / mask.mean([1,2])
        #code.interact(local = locals())
        
        return loss, {'dist_map': dist_map, 'b_proj1': b_proj1}
        
    def forward(self, x):
        """
        params:
            matches: [b, N, 4]
            K: [b, ch, 3, 3]
            matches_depth: [b, N, 2]
        """
        (matches, Ks, K_invs, matches_depth) = (x[0], x[1], x[2], x[3])
        batch_size = matches.shape[0]
        b_xy1, b_xy2 = matches[...,:2], matches[...,2:]
        b_z1, b_z2 = matches_depth[...,:1], matches_depth[...,1:]
        b_xyz1 = torch.cat([b_xy1, b_z1], dim=2)
        b_xyz2 = torch.cat([b_xy2, b_z2], dim=2)
        from deepFEPE.train_good_utils import get_E_ests_deepF
        data_batch = {
            "matches_xy_ori": matches,
            "quality": None,
            "Ks": Ks,
            "K_invs": K_invs,
            "des1": None,
            "des2": None,
            "matches_good_unique_nums": matches.shape[1], ## 
        }
        loss = 0.
        outs = self.net(data_batch)
        
        #""" # for reprojection loss
        # solve for E, poses
        E_ests_layers = get_E_ests_deepF(outs, Ks.to(self.device), K_invs.to(self.device)) # [D, B, 3, 3]
        outs['E_ests_layers'] = E_ests_layers
        Ks_np = Ks.to('cpu').numpy()
        b_xy1_np = b_xy1.detach().to('cpu').numpy()
        b_xy2_np = b_xy2.detach().to('cpu').numpy()
        b_Rt_cam = []
        for idx in range(batch_size):
            M2_list, error_Rt, Rt_cam = _E_to_M_train(E_ests_layers[-1][idx], Ks_np[idx], b_xy1_np[idx], 
                    b_xy2_np[idx], show_result=True)
            b_Rt_cam.append(Rt_cam)
            #break
        b_Rt_cam = torch.stack(b_Rt_cam, dim=0)

        # reprojection loss
        #code.interact(local = locals())
        loss, obj_by_name = self.compute_reprojection_loss(b_xyz1, b_xyz2, Ks, K_invs, b_Rt_cam,
                                                     mask=outs['weights'].transpose(1,2))
        #"""
        
        # F loss
        #loss, dist_map = self.compute_epipolar_loss(outs["F_est"], matches.transpose(1,2), 
        #                                            mask=outs['weights'])
        outs.update(obj_by_name)
        outs['pose'] = Rt_cam
        return outs, loss

    
    
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


  