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
from utils.utils import desc_to_sparseDesc, prep_superpoint_image, prep_trianflow_image, get_2d_matches, dense_sparse_hybrid_correspondences, sample_random_k, crop_or_pad_choice



    
class SiftFlow(torch.nn.Module):
    def __init__(self, model_cfg, general_cfg):
        """
        Model consists of two modules for correspondences:
            TrianFlow : https://github.com/B1ueber2y/TrianFlow
            SuperPoint : https://github.com/eric-yyjau/pytorch-superpoint
        """
        super(SiftFlow, self).__init__()
        
        self.device = 'cuda:0'
        self.num_matches = general_cfg["ransac_num_matches"]
        self.ransac_num_matches = general_cfg["ransac_num_matches"]

        #TrianFlow
        self.trianFlow = Model_depth_pose(model_cfg["trianflow"])
        #Load pretrained Modules
        self.did_load_modules = False
        self.load_modules(model_cfg)
        
        #SIFT
    
    def load_modules(self, cfg):
        """
        Loads specific modules that were pretrained into the pipeline, rather than the entire model
        """
        #load trian flow
        pretrained_file = cfg["trianflow"].pretrained
        print(f"load siftflow model from: {pretrained_file}")
        weights = torch.load(pretrained_file)
        self.trianFlow.load_state_dict(weights['model_state_dict'])
        
        self.did_load_modules = True
        
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

    def inference(self, image1, image2, K, K_inv, hw=None, 
                  preprocess=True):
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
        #TrianFlow
        if preprocess:
            image1_t, image1_resized = prep_trianflow_image(image1, hw)
            image2_t, image2_resized = prep_trianflow_image(image2, hw)
            outs['inputs'] = { 'image1' : image1_resized , 'image2' : image2_resized }
            K = torch.from_numpy(K).cuda().float().unsqueeze(0)
            K_inverse = torch.from_numpy(K_inv).cuda().float().unsqueeze(0)
        else:
            image1_t, image2_t = image1, image2
            image1_resized = squeezeToNumpy(image1).transpose(1,2,0)*255
            image2_resized = squeezeToNumpy(image2).transpose(1,2,0)*255
            #print(f"image1_resized: {image1_resized}")
            outs['inputs'] = { 'image1' : image1 , 'image2' : image2 }
            K, K_inverse = K, K_inv
        
        #code.interact(local=locals())
        correspondences, image1_depth_map, image2_depth_map = self.trianFlow.infer_vo(image1_t, image2_t, K.unsqueeze(0), K_inverse.unsqueeze(0), self.num_matches)

        #post process
        outs['flownet_correspondences'] = squeezeToNumpy(correspondences.T)
        outs['image1_depth'] = squeezeToNumpy(image1_depth_map)
        outs['image2_depth'] = squeezeToNumpy(image2_depth_map)
        
        #SIFT 
        outs['image1_sift_keypoints'], outs['image1_sift_descriptors'] = classical_detector_descriptor(image1_resized, image1_resized)
        outs['image2_sift_keypoints'], outs['image2_sift_descriptors'] = classical_detector_descriptor(image2_resized, image2_resized)
        
        image1_sift_matches, image2_sift_matches, _, good_matches_indices = KNN_match(outs['image1_sift_descriptors'], outs['image2_sift_descriptors'], outs['image1_sift_keypoints'], outs['image2_sift_keypoints'], None, None, None, None)
        
        outs['sift_correspondences'] = np.concatenate((image1_sift_matches, image2_sift_matches), axis=1)
        
        #if too many sift correspondences
        if outs['sift_correspondences'].shape[0] > self.num_matches:
            outs['sift_correspondences'] = sample_random_k(outs['sift_correspondences'], self.num_matches, outs['sift_correspondences'].shape[0])
        if outs['image1_sift_keypoints'].shape[0] > 1000:
            outs['image1_sift_keypoints'] = sample_random_k(outs['image1_sift_keypoints'], 1000, outs['image1_sift_keypoints'].shape[0])
        if outs['image2_sift_keypoints'].shape[0] > 1000:
            outs['image2_sift_keypoints'] = sample_random_k(outs['image2_sift_keypoints'], 1000, outs['image2_sift_keypoints'].shape[0])

        mid_time = datetime.utcnow()
        print(f'SIFT and flownet took {mid_time - start_time} to run')
        
        #SIFTFLOW
        matches = dense_sparse_hybrid_correspondences(outs['image1_sift_keypoints'], outs['image2_sift_keypoints'], outs['flownet_correspondences'], outs['sift_correspondences'], self.ransac_num_matches)
        outs['matches'] = matches
        
        # get depth from bilinear sampling
        def get_depth_from_image(pts, depth_map):
            """ extract sparse depth
            params: 
                pts: tensor [N, 2] (should be the same device as desc)
                depth_map: [1, D, Hc, Wc]
            return:
                depth: [N, D]
            """
            samp_pts = pts.transpose(0,1)
            H, W = image1_depth_map.shape[-2], image1_depth_map.shape[-1]
            # Interpolate into descriptor map using 2D point locations.
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            # samp_pts = samp_pts.to(self.device)
            desc = torch.nn.functional.grid_sample(depth_map, samp_pts, align_corners=True) # tensor [batch_size(1), D, 1, N]
            # desc = desc.data.cpu().numpy().reshape(D, -1)
            # desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
            desc = desc.squeeze(1).squeeze(0).transpose(0,1)
            return desc
        
        
        # get depth for matches
        matches_tensor = torch.from_numpy(matches).float().to(self.device)
        pnt_depth_1 = get_depth_from_image(matches_tensor[:,:2], image1_depth_map)
        pnt_depth_2 = get_depth_from_image(matches_tensor[:,:2], image2_depth_map)
        #code.interact(local=locals())
        outs['matches_depth'] = torch.stack([pnt_depth_1, pnt_depth_2], dim=1)
        outs['matches_tensor'] = matches_tensor
        
        return outs


    def forward(self, x):
        """ Forward pass computes keypoints, descriptors, and 3d-2d correspondences.
        Input
            x: Batch size B's of images : B x (2H) x W
        Output
            output: outs['matches'], Losses 
        """

        #superpoint
        #out
        #nms (val fastnms or process_output())
        #pts
        #desc to sparse
        
        ## torch
        (images, images_gray, K, K_inv) = (x[0], x[1], x[2], x[3])
        img_h, img_w = int(images.shape[2] / 2), images.shape[3] 
        image1, image2 = images[:,:,:img_h,:], images[:,:,img_h:,:]
        image1_gray, image2_gray = images_gray[:,:,:img_h,:], images_gray[:,:,img_h:,:]        
        
        ## batch inference with for loop
        outs_list = []
        outs_select = {'matches': None}
        loss = 0.
        batch_size = K.shape[0]
        for i in range(batch_size):
            outs_list.append(self.inference(image1[i].unsqueeze(0), image2[i].unsqueeze(0), 
                                            K[i].float().unsqueeze(0), K_inv[i].float().unsqueeze(0), 
                                            preprocess=False))
            print(f"matches: {outs_list[-1]['matches'].shape}")
        for i, en in enumerate(outs_select):
            #code.interact(local=locals())
            outs_select[en] = torch.stack([torch.from_numpy(outs[en]).float() \
                                           for outs in outs_list]).to(self.device)
        ## put results back to torch
        return outs_select, loss
        pass

   
