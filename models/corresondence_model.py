"""
Sub Module for finding correspondences, keypoints, and descriptors using Superpoint and Flownet
"""
#general
import numpy as np
import code

#torch imports
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_

#Superpoint imports
from superpoint.Train_model_heatmap import Train_model_heatmap

#TrianFlow imports
from TrianFlow.core.networks.model_depth_pose import Model_depth_pose 

class SuperFlow(torch.nn.Module):
    def __init__(self, cfg):
        """
        Model consists of two modules for correspondences:
            TrianFlow : https://github.com/B1ueber2y/TrianFlow
            SuperPoint : https://github.com/eric-yyjau/pytorch-superpoint
        """
        super(SuperFlow, self).__init__()

        #TrianFlow
        self.trianFlow = Model_depth_pose(cfg["trianflow"])

        #SuperPoint
        self.superpoint = Train_model_heatmap(cfg["superpoint"], device='cuda')
    
    def load_modules(self, cfg):
        """
        Loads specific modules that were pretrained into the pipeline, rather than the entire model
        """
        #load trian flow
        weights = torch.load(cfg["trianflow"].pretrained)
        self.trianFlow.load_state_dict(weights['model_state_dict'])

        #load superpoint
        self.superpoint.loadModel()

        pass

    def load_model(self):
        """
        Loads the entire model for this class as one, rather than separate modules from which it consists of
        """
        pass

    def inference(self, image1, image2, K, K_inv, match_num):
        """ Forward pass computes keypoints, descriptors, and 3d-2d correspondences.
        Input
            image1, image2: input pair images
            K, K_inverse : intrinsic matrix, and its inverse
            match_num : number of matches to output
        Output
            output: (2d-2d correspondences, image1_3d_points)
        """
        outs = {}

        #trianflow
        correspondences, image1_depth_map, image2_depth_map = self.trianFlow.infer_vo(image1_t, image2_t, K, K_inverse, match_num)
        outs['correspondences'] = correspondences
        outs['image1_depth'] = image1_depth_map 
        outs['image2_depth'] = image2_depth_map 

        #superpoint
        # #TODO : Make sure that I'm using the right img format
        # heatmap_batch = self.superPoint.run(images.to(K.device))  
        # pts = self.superPoint.heatmap_to_pts()
        # if subpixel:
        #     pts = self.superPoint.soft_argmax_points(pts, patch_size=patch_size)
        # desc_sparse = self.superPoint.desc_to_sparseDesc()
        # outs = {"pts": pts[0], "desc": desc_sparse[0]}

        return outs


    def forward(self, x):
        """ Forward pass computes keypoints, descriptors, and 3d-2d correspondences.
        Input
            x: Input pair of images N x 2 x H x W
        Output
            output: Losses 
        """
        pass

   
