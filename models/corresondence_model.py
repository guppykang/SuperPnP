"""
Sub Module for finding correspondences, keypoints, and descriptors using Superpoint and Flownet
"""
#general
import numpy as np
import code
import cv2

#torch imports
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_

#Superpoint imports
from superpoint.Train_model_heatmap import Train_model_heatmap
from superpoint.utils.var_dim import toNumpy
from superpoint.utils.utils import flattenDetection

#TrianFlow imports
from TrianFlow.core.networks.model_depth_pose import Model_depth_pose 

#My Utils
from utils.utils import desc_to_sparseDesc, prep_superpoint_image, prep_trianflow_image

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

    def inference(self, image1, image2, K, K_inv, match_num, hw):
        """ Forward pass computes keypoints, descriptors, and 3d-2d correspondences.
        Input
            image1, image2: input pair images
            K, K_inverse : intrinsic matrix, and its inverse
            match_num : number of matches to output
        Output
            output: (2d-2d correspondences, image1_3d_points)
        """
        outs = {}
        
        #TrianFlow
        image1_t = prep_trianflow_image(image1, hw)
        image2_t = prep_trianflow_image(image2, hw)
        K = torch.from_numpy(K).cuda().float().unsqueeze(0)
        K_inverse = torch.from_numpy(K_inv).cuda().float().unsqueeze(0)

        correspondences, image1_depth_map, image2_depth_map = self.trianFlow.infer_vo(image1_t, image2_t, K, K_inverse, match_num)
        outs['correspondences'] = correspondences
        outs['image1_depth'] = image1_depth_map 
        outs['image2_depth'] = image2_depth_map 

        
        
        #superpoint
        image1_t = prep_superpoint_image(image1, hw)
        image2_t = prep_superpoint_image(image2, hw)

        #inference on superpoint
        with torch.no_grad():
            code.interact(local=locals())
            outs['image1_superpoint_out'] = self.superpoint.net(image1_t)
            outs['image2_superpoint_out'] = self.superpoint.net(image2_t)

        #get the heatmap for each semi dense keypoint detection
        for out_key in ['image1_superpoint_out', 'image2_superpoint_out']:
            channel = outs[out_key]['semi'].shape[1]
            if channel == 64:
                heatmap = self.superpoint.flatten_64to1(outs[out_key]['semi'], cell_size=self.superpoint.cell_size)
            elif channel == 65:
                heatmap = flattenDetection(outs[out_key]['semi'], tensor=True)
            
            #get the exact 2d keypoints from the heatmaps
            heatmap_np = toNumpy(heatmap)
            pts = self.superpoint.heatmap_nms(heatmaps) #refer to heatmap_nms static function
            outs[out_key]['pts'] = pts
            outs[out_key]['sparse_desc'] = desc_to_sparseDesc(outs[out_key])
            
            
            #TODO : can also get matches using the func : get_matches(deses_SP) in SuperPointNet_gauss2.py
            
        return outs


    def forward(self, x):
        """ Forward pass computes keypoints, descriptors, and 3d-2d correspondences.
        Input
            x: Batch size B's of images : B x (2H) x W
        Output
            output: Losses 
        """

        pass

   
