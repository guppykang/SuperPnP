#!/usr/bin/env python 

import os, sys
sys.path.append("/jbk001-data1/git/SuperPnP")
import yaml
from TrianFlow.core.dataset.kitti_odo import KITTI_Odo

from utils.TUM_prepare import TUM_Prepare
from utils.TUM_dataset import TUM_Dataset

from utils.utils import load_image_pair, load_camera_intrinsics, pObject, get_configs, get_random_sequence

from collections import OrderedDict
import torch
import torch.utils.data
from tqdm import tqdm
import shutil
import pickle
import pdb
import code
from tensorboardX import SummaryWriter
import datetime
from pathlib import Path

from utils.logging import *

def save_model(iter_, model_dir, filename, model, optimizer):
    torch.save({"iteration": iter_, "model_state_dict": model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(model_dir, filename))

def load_model(model_dir, filename, model, optimizer):
    data = torch.load(os.path.join(model_dir, filename))
    iter_ = data['iteration']
    model.load_state_dict(data['model_state_dict'])
    optimizer.load_state_dict(data['optimizer_state_dict'])
    return iter_, model, optimizer

def freeze_all_but_depth(model, mode):
    """
    
    """
    if mode != 'depth_pose': 
        return
    
    print('Finetuning the decoder of the depthnet')
    
    for param in model.depth_net.encoder.parameters():
        param.requires_grad = False
        
    for param in model.model_pose.parameters():
        param.requires_grad = False
        
    return model

def train(model, cfg):
    # load model and optimizer
    print(type(model))
    
    model = model.cuda()
    optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': cfg.lr}])

    # load dataset
    # code.interact(local=locals())
    data_dir = cfg.data['procressed_data_path'] # debug
    # data_dir = cfg['prepared_base_dir'] # 
    if not os.path.exists(os.path.join(data_dir, 'train.txt')):
        if cfg.dataset == 'kitti_odo':
            kitti_raw_dataset = KITTI_Odo(cfg.raw_base_dir, cfg.vo_gts)
            kitti_raw_dataset.prepare_data_mp(data_dir, stride=cfg.stride)
        elif cfg.dataset == 'tum':
            tum_raw_dataset = TUM_Prepare(cfg.raw_base_dir)
            tum_raw_dataset.prepare_data_mp(data_dir, stride=cfg.stride)
        else:
            raise NotImplementedError
        
    if cfg.dataset == 'kitti_odo':
        from utils.KITTI_dataset import KITTI_Dataset as KITTI_Prepared
        dataset = KITTI_Prepared(data_dir, num_scales=cfg.num_scales, img_hw=cfg.img_hw, num_iterations=(cfg.num_iterations - cfg.iter_start) * cfg.batch_size, stride=cfg.stride)
    elif cfg.dataset == 'tum':
        dataset = TUM_Dataset(data_dir, num_scales=cfg.num_scales, img_hw=cfg.img_hw, num_iterations=(cfg.num_iterations - cfg.iter_start) * cfg.batch_size, stride=cfg.stride)
    else:
        raise NotImplementedError
        
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    #logging
    if not os.path.isdir('./tensorboard'):
        os.mkdir('./tensorboard')
    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
    writer = SummaryWriter(f'./tensorboard/trianflow_{start_time}', flush_secs=1)
    f = open(f'./tensorboard/logs_{start_time}.txt', 'w+')

    # training
    print('starting iteration: {}.'.format(cfg.iter_start))
    for iter_, inputs in enumerate(tqdm(dataloader)):
        if (iter_ + 1) % cfg.test_interval == 0 and (not cfg.no_test):
            model.eval()
            if cfg.multi_gpu:
                model_eval = model.module
            else:
                model_eval = model
                
        model.train()
        iter_ = iter_ + cfg.iter_start
        
        optimizer.zero_grad()
        trianflow_inputs = (inputs[0], inputs[1], inputs[2], inputs[3])
        outs, loss_pack = model(trianflow_inputs)
        #if iter_ % cfg.log_interval == 0:
        #    visualizer.print_loss(loss_pack, iter_=iter_)
            
        #in case tensorboard shits the bed
        #f.write(f'{loss_pack["pt_depth_loss"].mean().data.item()}, {loss_pack["pj_depth_loss"].mean().data.item()}, {loss_pack["depth_smooth_loss"].mean().data.item()}\n')
        #f.flush()
        
        #writer.add_scalar('loss_train/triangulation loss', loss_pack['pt_depth_loss'].mean().data.item(), iter_)
        #writer.add_scalar('loss_train/reprojection loss', loss_pack['pj_depth_loss'].mean().data.item(), iter_)
        #writer.add_scalar('loss_train/depth smooth loss', loss_pack['depth_smooth_loss'].mean().data.item(), iter_)
        #writer.flush()

        #loss_list = []
        #for key in list(loss_pack.keys()):
        #    loss_list.append((loss_weights_dict[key] * loss_pack[key].mean()).unsqueeze(0))
        #loss = torch.cat(loss_list, 0).sum()
        model.plot_tb(writer, task='train', n_iter=iter_)
        loss = loss_pack['all'].sum()
        logging.info(f"loss: {loss}")
        loss.backward()
        optimizer.step()
        if (iter_ + 1) % cfg.save_interval == 0:
            save_model(iter_, cfg.model_dir, 'iter_{}.pth'.format(iter_), model, optimizer)
            save_model(iter_, cfg.model_dir, 'last.pth'.format(iter_), model, optimizer)
    
    if cfg.dataset == 'kitti_depth':
        if cfg.mode == 'depth' or cfg.mode == 'depth_pose':
            eval_depth_res = test_eigen_depth(cfg, model_eval)

if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(
        description="TrianFlow training pipeline."
    )
    arg_parser.add_argument('-c', '--config_file', default='./configs/train/superglueflow.yaml', help='config file.')
    arg_parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id.')
    arg_parser.add_argument('--iter_start', type=int, default=0, help='starting iteration.')
    arg_parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    arg_parser.add_argument('--num_workers', type=int, default=1, help='number of workers.')
    arg_parser.add_argument('--log_interval', type=int, default=10, help='interval for printing loss.')
    arg_parser.add_argument('--test_interval', type=int, default=2000, help='interval for evaluation.')
    arg_parser.add_argument('--save_interval', type=int, default=2000, help='interval for saving models.')
    
    arg_parser.add_argument('--mode', type=str, default='superglueflow', help='[superglueflow, siftflow]')
#     arg_parser.add_argument('--model_dir', type=str, default='./logs/', help='directory for saving models')
    
    arg_parser.add_argument('--dataset', type=str, default='kitti', help='[kitti, tum]')
    arg_parser.add_argument('--sequence', type=str, default='10', help='Which sequence to run on the specified dataset')
    
    arg_parser.add_argument('--resume', action='store_true', help='to resume training.')
    arg_parser.add_argument('--stride', default=1, help='Stride between image pairs to train under')
#     arg_parser.add_argument('--multi_gpu', action='store_true', help='to use multiple gpu for training.')

    args = arg_parser.parse_args()
    
    #configs
    if args.config_file is None:
        raise ValueError('config file needed. -c --config_file.')
    #do config stuff

    # set model
    if args.mode == 'superglueflow':
        model_cfg, cfg = get_configs(args.config_file, mode='superglueflow')    
        from models.superglueflow import SuperGlueFlow as Model
    elif args.mode == 'siftflow_deepF':
        model_cfg, cfg = get_configs(args.config_file, mode='siftflow')    
        from models.siftflow_deepF import SiftFlow_deepF as Model
    else : 
        raise ValueError('Model type not implemented yet')
    model = Model(model_cfg, cfg)
    model.load_modules(model_cfg)

    # set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    print(f"cfg: {cfg}")
    
    class pObject(object):
        def __init__(self):
            pass
    cfg_new = pObject()
    for attr in list(cfg.keys()):
        setattr(cfg_new, attr, cfg[attr])
    
    
    # main function 
    train(model, cfg_new)

