"""
Sub Module for finding correspondences, keypoints, and descriptors using Superpoint and Flownet
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

#superpoint imports 
from superpoint.utils.var_dim import toNumpy, squeezeToNumpy

#SuperGlue imports
from superglue.models.matching import Matching

#TrianFlow imports
from TrianFlow.core.networks.model_depth_pose import Model_depth_pose 

#My Utils
from utils.utils import desc_to_sparseDesc, prep_superpoint_image, prep_trianflow_image, get_2d_matches, dense_sparse_hybrid_correspondences

class SuperGlueFlow(torch.nn.Module):
    def __init__(self, model_cfg, general_cfg):
        """
        Model consists of two modules for correspondences:
            TrianFlow : https://github.com/B1ueber2y/TrianFlow
            Superglue : https://github.com/magicleap/SuperGluePretrainedNetwork
        """
        super(SuperGlueFlow, self).__init__()
        
        self.device = 'cuda:0'
        self.num_matches = general_cfg["ransac_num_matches"]
        self.ransac_num_matches = general_cfg["ransac_num_matches"]

        #TrianFlow
        self.trianFlow = Model_depth_pose(model_cfg["trianflow"])
    
        #Superglue
        self.superglue_matcher = Matching(model_cfg)

        #Load pretrained Modules
        self.did_load_modules = False
        self.load_modules(model_cfg)
       
    
    def load_modules(self, cfg):
        """
        Loads specific modules that were pretrained into the pipeline, rather than the entire model
        """
        
        if self.did_load_modules:
            return 
        
        print('Loading Superglueflow with learned weights')
        #load trian flow
        weights = torch.load(cfg["trianflow"].pretrained)
        self.trianFlow.load_state_dict(weights['model_state_dict'])

        #load superpoint
        #superglue matcher loads superoint and superglue in their resepctive __init__ functions
        
        self.did_load_modules = True

        pass

    def load_model(self):
        """
        Loads the entire model for this class as one, rather than separate modules from which it consists of
        """
        pass

    def inference(self, image1, image2, K, K_inv, hw):
        """ Forward pass computes keypoints, descriptors, and 3d-2d correspondences.
        Input
            image1, image2: input pair images
            K, K_inverse : intrinsic matrix, and its inverse
        Output
            outs: {
                   flownet_correspondences, 
                   superglue_correspondences,
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
                   superglueflow_correspondences
                   }
        """
        outs = {}
        start_time = datetime.utcnow()


        #superpoint
        image1_t = prep_superpoint_image(image1, hw)
        image2_t = prep_superpoint_image(image2, hw)
#         print(f'superpoint shape : {image1_t.shape}')
        pred = self.superglue_matcher({'image0' : image1_t, 'image1' : image2_t})
        pred = {k: toNumpy(v[0]) for k, v in pred.items()}
        outs['keypoints'] = [pred['keypoints0'], pred['keypoints1']]
        matches, conf = pred['matches0'], pred['matching_scores0']
        
        # Keep the matching keypoints.
        valid = matches > -1
        outs['superglue_correspondences'] = np.concatenate((outs['keypoints'][0][valid], outs['keypoints'][1][matches[valid]]), axis=1)
        outs['superglue_scores'] = conf[valid]
        
        
        #TrianFlow
        image1_t, image1_resized = prep_trianflow_image(image1, hw)
        image2_t, image2_resized = prep_trianflow_image(image2, hw)
#         print(f'flownet shape : {image1_t.shape}')

        outs['inputs'] = { 'image1' : image1_resized , 'image2' : image2_resized }
        K = torch.from_numpy(K).cuda().float().unsqueeze(0)
        K_inverse = torch.from_numpy(K_inv).cuda().float().unsqueeze(0)
        correspondences, image1_depth_map, image2_depth_map = self.trianFlow.infer_vo(image1_t, image2_t, K, K_inverse, self.num_matches)

        #post process
        outs['flownet_correspondences'] = squeezeToNumpy(correspondences.T)
        outs['image1_depth'] = squeezeToNumpy(image1_depth_map)
        outs['image2_depth'] = squeezeToNumpy(image2_depth_map)
        
        mid_time = datetime.utcnow()
        print(f'Took {mid_time - start_time} to run')
        
        
        #SuperglueFLOW
        outs['matches'] = dense_sparse_hybrid_correspondences(outs['keypoints'][0], outs['keypoints'][1], outs['flownet_correspondences'], outs['superglue_correspondences'], self.ransac_num_matches)
        print(f'num matches for ransac : {self.ransac_num_matches}')

        return outs
    
    def inference_preprocessed(self, image1, image2, image1_gray, image2_gray, K, K_inv, attention_map=None):
        """ 
        Inferences a pair of images that were outputted by the appropriate dataloader
        """
        outs = {}
        
        outs['inputs'] = { 'image1' : squeezeToNumpy(image1).transpose(1,2,0) , 'image2' : squeezeToNumpy(image2).transpose(1,2,0) }
              
              
        #SuperGlue
        pred = self.superglue_matcher({'image0' : image1_gray, 'image1' : image2_gray})
        pred = {k: toNumpy(v[0]) for k, v in pred.items()}
        outs['keypoints'] = [pred['keypoints0'], pred['keypoints1']]
        matches, conf = pred['matches0'], pred['matching_scores0']
        valid = matches > -1
        outs['superglue_correspondences'] = np.concatenate((outs['keypoints'][0][valid], outs['keypoints'][1][matches[valid]]), axis=1)
        outs['superglue_scores'] = conf[valid]


        #TrianFlow
        K = K.float().unsqueeze(0)
        K_inv = K_inv.float().unsqueeze(0)
        correspondences, image1_depth_map, image2_depth_map = self.trianFlow.infer_vo(image1, image2, K, K_inv, self.num_matches)
        outs['flownet_correspondences'] = squeezeToNumpy(correspondences.T)
        outs['image1_depth'] = squeezeToNumpy(image1_depth_map)
        outs['image2_depth'] = squeezeToNumpy(image2_depth_map)


        #SuperFLOW
        outs['matches'] = dense_sparse_hybrid_correspondences(outs['keypoints'][0], outs['keypoints'][1], outs['flownet_correspondences'], outs['superglue_correspondences'], self.ransac_num_matches, attention_map=attention_map)
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
        
        raise RuntimeError('Not implemented yet')
        pass

   
