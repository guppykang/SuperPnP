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
from superpoint.utils.var_dim import toNumpy, squeezeToNumpy
from superpoint.utils.utils import flattenDetection
from superpoint.models.model_utils import SuperPointNet_process


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
        
        self.device = 'cuda:0'

        #TrianFlow
        self.trianFlow = Model_depth_pose(cfg["trianflow"])

        #SuperPoint
        self.superpoint = Train_model_heatmap(cfg["superpoint"], device=self.device)
        
        self.superpoint_processor_params = {
            'out_num_points': 500,
            'patch_size': 5,
            'device': self.device,
            'nms_dist': 4,
            'conf_thresh': 0.015
        }
        self.superpoint_processor = SuperPointNet_process(**self.superpoint_processor_params)
    
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
        pair_input_tensor = torch.cat((image1_t, image2_t), 0)
        with torch.no_grad():
            superpoint_out = self.superpoint.net(pair_input_tensor)

        processed_superpoint_out = self.superpoint.net.process_output(self.superpoint_processor)
        outs['image1_superpoint_out'], outs['image2_superpoint_out'] = {}, {}
        for out_key in processed_superpoint_out.keys():
            #TODO : don't forget here that we detached the tensor
            for img_idx, img_key in enumerate(['image1_superpoint_out', 'image2_superpoint_out']):
                outs[img_key][out_key] = processed_superpoint_out[out_key][img_idx]
                
                
        #TODO : can also get matches using the func : get_matches(deses_SP) in SuperPointNet_gauss2.py
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

   
