"""
Sub Module for finding correspondences, keypoints, and descriptors using Superpoint and Flownet
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
from models.unet_parts import *
import numpy as np
from pytorch-superoint.utils.loader import get_module

class SuperFlow(torch.nn.Module):
    def __init__(self, cfg):

        #TrianFlow
        self.trianFlow = Model_depth_pose(cfg["trianFlow"])

        #SuperPoint
        self.superPoint = get_module("", cfg["front_end_model"])
    

    def loadModules(self):
        #load trian flow
        weights = torch.load(args.pretrained_model)
        self.trianFlow.load_state_dict(weights['model_state_dict'])

        #load superpoint
        self.superPoint = Val_model_heatmap(config["model"])
        self.superPoint.loadModel()

        pass

    def loadModel(self):
        pass

    def inference(self, image1, image2, K, K_inverse, match_num):
        """ Forward pass computes keypoints, descriptors, and 3d-2d correspondences.
        Input
            image1, image2: input pair images
            K, K_inverse : intrinsic matrix, and its inverse
            match_num : number of matches to output
        Output
            output: (2d-2d correspondences, image1_3d_points)
        """
        image1 = torch.from_numpy(np.transpose(img1 / 255.0, [2,0,1])).cuda().float().unsqueeze(0)
        image2 = torch.from_numpy(np.transpose(img2 / 255.0, [2,0,1])).cuda().float().unsqueeze(0)
        images = torch.from_numpy(np.array([image1, image2])).float().unsqueeze(0)
        K = torch.from_numpy(K).cuda().float().unsqueeze(0)
        K_inverse = torch.from_numpy().cuda().float().unsqueeze(0)

        #trianflow
        correspondences, image1_depth_map, image2_depth_map = self.trianFlow.infer_vo(images[0], images[1], K, K_inverse, match_num)

        #superpoint
        #TODO : Make sure that I'm using the right img format
        heatmap_batch = self.superPoint.run(images.to(K.device))  
        pts = self.superPoint.heatmap_to_pts()
        if subpixel:
            pts = self.superPoint.soft_argmax_points(pts, patch_size=patch_size)
        desc_sparse = self.superPoint.desc_to_sparseDesc()
        outs = {"pts": pts[0], "desc": desc_sparse[0]}

        #TODO : do stuff here


        return outs


    def forward(self, x):
        """ Forward pass computes keypoints, descriptors, and 3d-2d correspondences.
        Input
            x: Input pair of images N x 2 x H x W
        Output
            output: Losses 
        """
        pass
        return output

    def process_output(self, sp_processer):
        """
        input:
          N: number of points
        return: -- type: tensorFloat
          pts: tensor [batch, N, 2] (no grad)  (x, y)
          pts_offset: tensor [batch, N, 2] (grad) (x, y)
          pts_desc: tensor [batch, N, 256] (grad)
        """
        from utils.utils import flattenDetection
        # from models.model_utils import pred_soft_argmax, sample_desc_from_points
        output = self.output
        semi = output['semi']
        desc = output['desc']
        # flatten
        heatmap = flattenDetection(semi) # [batch_size, 1, H, W]
        # nms
        heatmap_nms_batch = sp_processer.heatmap_to_nms(heatmap, tensor=True)
        # extract offsets
        outs = sp_processer.pred_soft_argmax(heatmap_nms_batch, heatmap)
        residual = outs['pred']
        # extract points
        outs = sp_processer.batch_extract_features(desc, heatmap_nms_batch, residual)

        # output.update({'heatmap': heatmap, 'heatmap_nms': heatmap_nms, 'descriptors': descriptors})
        output.update(outs)
        self.output = output
        return output


def get_matches(deses_SP):
    from models.model_wrap import PointTracker
    tracker = PointTracker(max_length=2, nn_thresh=1.2)
    f = lambda x: x.cpu().detach().numpy()
    # tracker = PointTracker(max_length=2, nn_thresh=1.2)
    # print("deses_SP[1]: ", deses_SP[1].shape)
    matching_mask = tracker.nn_match_two_way(f(deses_SP[0]).transpose(), f(deses_SP[1]).transpose(), nn_thresh=1.2)
    # print("matching_mask: ", matching_mask.shape)
    # f_mask = lambda pts, maks: pts[]
    # pts_m = []
    # pts_m_res = []
    # for i in range(2):
    #     idx = xs_SP[i][matching_mask[i, :].astype(int), :]
    #     res = reses_SP[i][matching_mask[i, :].astype(int), :]
    #     print("idx: ", idx.shape)
    #     print("res: ", idx.shape)
    #     pts_m.append(idx)
    #     pts_m_res.append(res)
    #     pass

    # pts_m = torch.cat((pts_m[0], pts_m[1]), dim=1)
    # matches_test = toNumpy(pts_m)
    # print("pts_m: ", pts_m.shape)

    # pts_m_res = torch.cat((pts_m_res[0], pts_m_res[1]), dim=1)
    # # pts_m_res = toNumpy(pts_m_res)
    # print("pts_m_res: ", pts_m_res.shape)
    # # print("pts_m_res: ", pts_m_res)
        
    # pts_idx_res = torch.cat((pts_m, pts_m_res), dim=1)
    # print("pts_idx_res: ", pts_idx_res.shape)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SuperPointNet_gauss2()
    model = model.to(device)


    # check keras-like model summary using torchsummary
    from torchsummary import summary
    summary(model, input_size=(1, 240, 320))

    ## test
    image = torch.zeros((2,1,120, 160))
    outs = model(image.to(device))
    print("outs: ", list(outs))

    from utils.print_tool import print_dict_attr
    print_dict_attr(outs, 'shape')

    from models.model_utils import SuperPointNet_process 
    params = {
        'out_num_points': 500,
        'patch_size': 5,
        'device': device,
        'nms_dist': 4,
        'conf_thresh': 0.015
    }

    sp_processer = SuperPointNet_process(**params)
    outs = model.process_output(sp_processer)
    print("outs: ", list(outs))
    print_dict_attr(outs, 'shape')

    # timer
    import time
    from tqdm import tqdm
    iter_max = 50

    start = time.time()
    print("Start timer!")
    for i in tqdm(range(iter_max)):
        outs = model(image.to(device))
    end = time.time()
    print("forward only: ", iter_max/(end - start), " iter/s")

    start = time.time()
    print("Start timer!")
    xs_SP, deses_SP, reses_SP = [], [], []
    for i in tqdm(range(iter_max)):
        outs = model(image.to(device))
        outs = model.process_output(sp_processer)
        xs_SP.append(outs['pts_int'].squeeze())
        deses_SP.append(outs['pts_desc'].squeeze())
        reses_SP.append(outs['pts_offset'].squeeze())
    end = time.time()
    print("forward + process output: ", iter_max/(end - start), " iter/s")

    start = time.time()
    print("Start timer!")
    for i in tqdm(range(len(xs_SP))):
        get_matches([deses_SP[i][0], deses_SP[i][1]])
    end = time.time()
    print("nn matches: ", iter_max/(end - start), " iters/s")


if __name__ == '__main__':
    main()



