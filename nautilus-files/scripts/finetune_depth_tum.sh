#!/bin/bash

cd /jbk001-data1/git/SuperPnP/
./setup_nautilus.sh
echo himom

cd ./TrianFlow/
./train.py --finetune_depth True --config_file ./config/tum_3stage.yaml --depth_pretrained_model /jbk001-data1/git/SuperPnP/TrianFlow/models/pretrained/kitti_depth_pretrained.pth --mode depth_pose