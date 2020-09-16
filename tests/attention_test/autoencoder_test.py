#!/usr/bin/env python 

from models.attention import encoder, decoder, encoderDilation, decoderDilation
from utils.utils import load_image_pair, get_random_sequence
from pathlib import Path
import code
import torch
import numpy as np 

sequence = get_random_sequence()

vo_sequences_root = Path('/jbk001-data1/datasets/kitti/kitti_vo/vo_dataset/sequences')
image1, image2 = load_image_pair(vo_sequences_root, sequence)

image1 = torch.from_numpy(np.transpose(image1/255.0, [2,0,1])).cuda().float().unsqueeze(0)
image2 = torch.from_numpy(np.transpose(image2/255.0, [2,0,1])).cuda().float().unsqueeze(0)

encoder = encoder().cuda()
decoder = decoder().cuda()

x1, x2, x3, x4, x5 = encoder(image1)
output = decoder(image1, x1, x2, x3, x4, x5)

print(f'input shape : {image1.shape}')
print(f'encoded shape : {x5.shape}')
print(f'output shape : {output.shape}')

print('Passed')