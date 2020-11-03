#!/bin/bash

cd /jbk001-data1/git/SuperPnP/
./setup_nautilus.sh
echo himom

cd ./TrianFlow/
./train.py --config_file ./config/kitti_depth.yaml --flow_pretrained_model /jbk001-data1/git/SuperPnP/TrianFlow/models/pretrained/kitti_flow.pth --mode depth
