#!/bin/bash

cd /jbk001-data1/git/SuperPnP/
./setup_nautilus.sh
echo himom
source ~/.bashrc


cd ./TrianFlow/
/opt/conda/bin/python train.py