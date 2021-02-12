import numpy as np
import cv2
import copy
import os
import sys
import code
from pathlib import Path

import torch
import torch.utils.data as data


import torch
import torch.utils.data
import pdb

class TARTANAIR_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, num_scales=3, img_hw=(256, 832), num_iterations=None, stride=1):
        super(TARTANAIR_Dataset, self).__init__()
        self.data_dir = f"{data_dir}/image_left" #full path

        self.num_scales = num_scales
        self.img_hw = img_hw
        self.num_iterations = num_iterations     

    def count(self):
        return len(os.listdir(self.data_dir))-1 #size-1 pairs of images

    def rand_num(self, idx):
        num_total = self.count()
        np.random.seed(idx)
        num = np.random.randint(num_total)
        return num

    def __len__(self):
        return len(os.listdir(self.data_dir))-1 #size-1 pairs of images

    def resize_img(self, img, img_hw):
        '''
        Input size (N*H, W, 3)
        Output size (N*H', W', 3), where (H', W') == self.img_hw
        '''
        img_h, img_w = img.shape[0], img.shape[1]
        img_hw_orig = (int(img_h / 2), img_w) 
        
        if len(img.shape) == 3: #rgb
            img1, img2 = img[:img_hw_orig[0], :, :], img[img_hw_orig[0]:, :, :]
            img1_new = cv2.resize(img1, (img_hw[1], img_hw[0])) #hi my name is opencv I like to swap (h,w) convention every other day
            img2_new = cv2.resize(img2, (img_hw[1], img_hw[0]))
            img_new = np.concatenate([img1_new, img2_new], 0)            
        elif len(img.shape) == 2: #grayscale
            img1, img2 = img[:img_hw_orig[0], :], img[img_hw_orig[0]:, :]
            img1_new = cv2.resize(img1, (img_hw[1], img_hw[0]))
            img2_new = cv2.resize(img2, (img_hw[1], img_hw[0]))
            img_new = np.concatenate([img1_new, img2_new], 0)

        return img_new
    
    

    def random_flip_img(self, img):
        is_flip = (np.random.rand() > 0.5)
        if is_flip:
            img = cv2.flip(img, 1)
        return img

    def preprocess_img(self, img, img_hw=None, is_test=False):
        if img_hw is None:
            img_hw = self.img_hw
        img = self.resize_img(img, img_hw)
        if not is_test:
            img = self.random_flip_img(img)
        img = img / 255.0
        return img

    def read_cam_intrinsic(self, fname):
        with open(fname, 'r') as f:
            lines = f.readlines()
        data = lines[-1].strip('\n').split(' ')[1:]
        data = [float(k) for k in data]
        data = np.array(data).reshape(3,4)
        cam_intrinsics = data[:3,:3]
        return cam_intrinsics

    def rescale_intrinsics(self, K, img_hw_orig, img_hw_new):
        K[0,:] = K[0,:] * img_hw_new[0] / img_hw_orig[0]
        K[1,:] = K[1,:] * img_hw_new[1] / img_hw_orig[1]
        return K

    def get_intrinsics_per_scale(self, K, scale):
        K_new = copy.deepcopy(K)
        K_new[0,:] = K_new[0,:] / (2**scale)
        K_new[1,:] = K_new[1,:] / (2**scale)
        K_new_inv = np.linalg.inv(K_new)
        return K_new, K_new_inv

    def get_multiscale_intrinsics(self, K, num_scales):
        K_ms, K_inv_ms = [], []
        for s in range(num_scales):
            K_new, K_new_inv = self.get_intrinsics_per_scale(K, s)
            K_ms.append(K_new[None,:,:])
            K_inv_ms.append(K_new_inv[None,:,:])
        K_ms = np.concatenate(K_ms, 0)
        K_inv_ms = np.concatenate(K_inv_ms, 0)
        return K_ms, K_inv_ms

    def __getitem__(self, idx):
        '''
        Returns:
        - img		torch.Tensor (N * H, W, 3)
        - K	torch.Tensor (num_scales, 3, 3)
        - K_inv	torch.Tensor (num_scales, 3, 3)
        '''
        idx_left = str(self.rand_num(idx))
        idx_right = str(self.rand_num(idx)+1)
        for i in range(6 - len(idx_left)):
            idx_left = "0" + idx_left
        for i in range(6 - len(idx_right)):
            idx_right = "0" + idx_right

        data_path_left = f"{self.data_dir}/{idx_left}_left.png"
        data_path_right = f"{self.data_dir}/{idx_right}_left.png"
        
        # load img
        img_left = cv2.imread(data_path_left)
        img_right = cv2.imread(data_path_right)
        img = cv2.vconcat([img_left, img_right])

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_hw_orig = (int(img.shape[0] / 2), img.shape[1])
        
        #rgb
        flownet_input = self.preprocess_img(img, self.img_hw) # (img_h * 2, img_w, 3)
        flownet_input = flownet_input.transpose(2,0,1)
        
        #grayscale
        superpoint_input = self.preprocess_img(img_gray, self.img_hw) # (img_h * 2, img_w)
        superpoint_input = superpoint_input[np.newaxis, ...]
        
        return torch.from_numpy(flownet_input).float().cuda(), torch.from_numpy(superpoint_input).float().cuda()

if __name__ == '__main__':
    pass

