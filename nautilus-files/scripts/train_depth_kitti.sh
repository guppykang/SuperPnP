#!/bin/bash

cd /jbk001-data1/git/SuperPnP/
echo himom

cd ./TrianFlow/
/jbk001-data1/git/conda/py36-josh-1015/bin/python train.py --config_file ./config/kitti_depth.yaml --flow_pretrained_model /jbk001-data1/git/SuperPnP/TrianFlow/models/pretrained/kitti_flow.pth --mode depth
