"""
Sub Module for finding correspondences, keypoints, and descriptors using SIFT and/or Flownet
"""
#general
import numpy as np
import code
import cv2
from datetime import datetime

#torch imports
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_

#Superpoint imports
from superpoint.utils.var_dim import toNumpy, squeezeToNumpy

#superglue imports 
from superglue.models.matching import Matching

#TrianFlow imports
from TrianFlow.core.networks.model_depth_pose import Model_depth_pose 

#My Utils
from utils.utils import desc_to_sparseDesc, prep_superpoint_image, prep_trianflow_image, prep_scsfm_image, get_2d_matches, dense_sparse_hybrid_correspondences, sample_random_k

import models
from models.superglueflow import SuperGlueFlow

    
class SuperGlueFlow_scsfm(torch.nn.Module):
    def __init__(self, model_cfg, general_cfg):
        """
        Model consists of two modules for correspondences:
            TrianFlow : https://github.com/B1ueber2y/TrianFlow
            SuperPoint : https://github.com/eric-yyjau/pytorch-superpoint
        """
        super(SuperGlueFlow_scsfm, self).__init__()
        
        self.device = 'cuda:0'
        self.num_matches = 6000
        self.ransac_num_matches = general_cfg["ransac_num_matches"]

        #TrianFlow
        self.trianFlow = Model_depth_pose(model_cfg["trianflow"])
        
        # scsfm
        self.disp_net = getattr(models, general_cfg["models"]["scsfm"]["net"])().to(self.device)
        self.load_pretrained_net(general_cfg["models"]["scsfm"], self.disp_net)

        #superglue
        self.superglue_matcher = Matching(model_cfg)
        

    def load_pretrained_net(self, cfg_net, net):
        """
        Loads specific modules that were pretrained into the pipeline, rather than the entire model
        """
        #load trian flow
        weights = torch.load(cfg_net["pretrained"])
        net.load_state_dict(weights['state_dict'])
        pass
    
    def load_modules(self, cfg):
        """
        Loads specific modules that were pretrained into the pipeline, rather than the entire model
        """
        #load trian flow
        weights = torch.load(cfg["trianflow"].pretrained)
        self.trianFlow.load_state_dict(weights['model_state_dict'])

        pass

    def load_model(self):
        """
        Loads the entire model for this class as one, rather than separate modules from which it consists of
        """
        pass

    def dataParallel(self):
        """
        put network and optimizer to multiple gpus
        :return:
        """
        print("=== Let's use", torch.cuda.device_count(), "GPUs!")
        self.net = nn.DataParallel(self.net)
        self.optimizer = self.adamOptim(
            self.net, lr=self.config["model"]["learning_rate"]
        )

    def inference(self, image1, image2, K, K_inv, hw):
        """ Forward pass computes keypoints, descriptors, and 3d-2d correspondences.
        Input
            image1, image2: input pair images
            K, K_inverse : intrinsic matrix, and its inverse
        Output
            outs: {
                   flownet_correspondences, 
                   sift_correspondences,
                   inputs : 
                       image1, 
                       image2
                   keypoints, 
                   image1_superpoint_out:
                       pts_desc, 
                       pts_int, 
                       pts_offset, 
                       semi, 
                       desc
                   image2_superpoint_out:
                       pts_desc, 
                       pts_int, 
                       pts_offset, 
                       semi, 
                       desc
                   image1_depth, 
                   image2_depth, 
                   siftflow_correspondences
                   }
        """
        outs = {}
                   
        start_time = datetime.utcnow()
       
        ########## SUPERGLUE ##########
        image1_t = prep_superpoint_image(image1, hw)
        image2_t = prep_superpoint_image(image2, hw)
        pred = self.superglue_matcher({'image0' : image1_t, 'image1' : image2_t})
        pred = {k: toNumpy(v[0]) for k, v in pred.items()}
        outs['keypoints'] = [pred['keypoints0'], pred['keypoints1']]
        matches, conf = pred['matches0'], pred['matching_scores0']
        valid = matches > -1  # Keep the matching keypoints.
        outs['superglue_correspondences'] = np.concatenate((outs['keypoints'][0][valid], outs['keypoints'][1][matches[valid]]), axis=1)
        outs['superglue_scores'] = conf[valid]
        
        
        ########## TrianFlow ##########
        image1_t, image1_resized = prep_trianflow_image(image1, hw)
        image2_t, image2_resized = prep_trianflow_image(image2, hw)
        outs['inputs'] = { 'image1' : image1_resized , 'image2' : image2_resized }
        K = torch.from_numpy(K).cuda().float().unsqueeze(0)
        K_inverse = torch.from_numpy(K_inv).cuda().float().unsqueeze(0)
        correspondences, image1_depth_map, image2_depth_map = self.trianFlow.infer_vo(image1_t, image2_t, K, K_inverse, self.num_matches)
        
        # scsfm - depthNet
        image1_sc, image1_resized = prep_scsfm_image(image1, hw)
        image2_sc, image2_resized = prep_scsfm_image(image2, hw)
        image1_depth_map = 1/(self.disp_net(image1_sc)[0][0, 0] ) # overwrite depth_map
        image2_depth_map = 1/(self.disp_net(image2_sc)[0][0, 0] ) # overwrite depth_map

        #post process
        outs['flownet_correspondences'] = squeezeToNumpy(correspondences.T)
        outs['image1_depth'] = squeezeToNumpy(image1_depth_map)
        outs['image2_depth'] = squeezeToNumpy(image2_depth_map)

        mid_time = datetime.utcnow()
        print(f'Took {mid_time - start_time} to run')
        
        
        ########## superglueflow ##########
        outs['matches'] = dense_sparse_hybrid_correspondences(outs['keypoints'][0], outs['keypoints'][1], outs['flownet_correspondences'], outs['superglue_correspondences'], self.ransac_num_matches)
        print(f'num matches for ransac : {self.ransac_num_matches}')

        return outs


    def forward(self, x):
        """ Forward pass computes keypoints, descriptors, and 3d-2d correspondences.
        Input
            x: Batch size B's of images : B x (2H) x W
        Output
            output: Losses 
        """

        #superpoint
        #out
        #nms (val fastnms or process_output())
        #pts
        #desc to sparse
        pass

   