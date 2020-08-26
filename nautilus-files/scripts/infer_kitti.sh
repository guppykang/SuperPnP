#!/bin/bash

#$1 : model, $2 : absolute or relative

cd /jbk001-data1/git/SuperPnP/
./setup_nautilus.sh
echo himom

if [ "$1" == "siftflow" ]; then 
    echo "siftflow"
    if [ "$2" == 'absolute' ]; then 
        /opt/conda/bin/python infer_kitti.py --model siftflow --absolute
    else
        /opt/conda/bin/python infer_kitti.py --model siftflow     
    fi
elif [ "$1" == "superflow2" ]; then 
    echo "superflow2"
    if [ "$2" == 'absolute' ]; then 
        /opt/conda/bin/python infer_kitti.py --model superflow2 --absolute
    else
        /opt/conda/bin/python infer_kitti.py --model superflow2     
    fi
elif [ "$1" == "superglueflow" ]; then 
    echo "superglueflow"
    if [ "$2" == 'absolute' ]; then 
        /opt/conda/bin/python infer_kitti.py --model superglueflow --absolute
    else
        /opt/conda/bin/python infer_kitti.py --model superglueflow     
    fi
    
else
    echo "bad model"
fi