"""
Sub Module for finding correspondences, keypoints, and descriptors using SIFT and/or Flownet
"""
#general
import numpy as np
import code
import cv2
from datetime import datetime
import os
import logging


#torch imports
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_

#Superpoint imports
from superpoint.Train_model_heatmap import Train_model_heatmap
from superpoint.utils.var_dim import toNumpy, squeezeToNumpy
from superpoint.utils.utils import flattenDetection
from superpoint.models.model_utils import SuperPointNet_process
from superpoint.models.SuperPointNet_gauss2 import get_matches as get_descriptor_matches
from superpoint.utils.utils import flattenDetection
from superpoint.models.classical_detectors_descriptors import SIFT_det as classical_detector_descriptor

from deepFEPE.dsac_tools.utils_opencv import KNN_match

#TrianFlow imports
from TrianFlow.core.networks.model_depth_pose import Model_depth_pose 

#My Utils
from utils.utils import desc_to_sparseDesc, prep_superpoint_image, prep_trianflow_image, get_2d_matches, dense_sparse_hybrid_correspondences, sample_random_k

# relative imports
from .siftflow import SiftFlow
from infer_deepF import deepF_frontend

class SiftFlow_deepF(torch.nn.Module):
    def __init__(self, model_cfg, general_cfg, device='cpu'):
        super(SiftFlow_deepF, self).__init__()
        self.siftflow = SiftFlow(model_cfg, general_cfg)
        self.deepF_fe = deepF_frontend(general_cfg["models"]["deepF"])
        self.tb_scalar_by_name = {}
        self.tb_image_by_name = {}
        self.optimizers = {}
        self.optimizers.update({'deepF': self.deepF_fe.optimizer})
        self.nets = {'deepF': self.deepF_fe.net}
        
    def load_modules(self, cfg):
        pass
    
    def inference(self, image1, image2, K, K_inv, hw):
        pass
    
    def save_model(self, iter_, model_dir):
        for i, en in enumerate(self.nets):
            model = self.nets[en]
            optimizer = self.optimizers[en]
            filename = f'{en}_{iter_}.pth.tar'
            file = os.path.join(model_dir, filename)
            logging.info(f"save model to: {file}")
            torch.save({"n_iter": iter_, 'n_iter_val': iter_,
                        "model_state_dict": model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(),
                       }, file)
            
        
    
    def plot_tb(self, writer, task='train', n_iter=0):
        for element in list(self.tb_scalar_by_name):
            writer.add_scalar(task + "-" + element, 
                              self.tb_scalar_by_name[element], n_iter)
        
        pass
    
    def forward(self, x):
        """
        """
        loss = {}
        (images, images_gray, K, K_inv) = (x[0], x[1], x[2], x[3]) #flownet input pair, superpoint input pair, K, K_inv
        img_h, img_w = int(images.shape[2] / 2), images.shape[3] 
        image1, image2 = images[:,:,:img_h,:], images[:,:,img_h:,:]
        image1_gray, image2_gray = images_gray[:,:,:img_h,:], images_gray[:,:,img_h:,:]
        
        # feed into siftflow, get correspondences [B, N, 4]
        outs_stg1, loss_stg1 = self.siftflow((image1, image2, K, K_inv))
        
        # feed into deepF, get essential matrix [B, 3, 3]
        outs_stg2, loss_stg2 = self.deepF_fe((outs_stg1['matches'], K, K_inv))
        
        # compute loss from essential matrix
        #code.interact(local = locals())
        outs = outs_stg1.update(outs_stg2)
        loss['all'] = (loss_stg1 + loss_stg2).mean()
        self.tb_scalar_by_name.update(loss)
        return outs, loss
    