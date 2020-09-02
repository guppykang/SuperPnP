#!/bin/bash

#$1 : model, $2 : absolute or relative, $3 : stride

cd /jbk001-data1/git/SuperPnP/
./setup_nautilus.sh
echo himom

if [ "$1" == "siftflow" ]; then 
    echo "siftflow"
    if [ "$2" == 'absolute' ]; then 
        /opt/conda/bin/python infer_kitti.py --model siftflow --mode absolute
    else
        /opt/conda/bin/python infer_kitti.py --model siftflow --stride $3
    fi
elif [ "$1" == "superflow" ]; then 
    echo "superflow"
    if [ "$2" == 'absolute' ]; then 
        /opt/conda/bin/python infer_kitti.py --model superflow --mode absolute
    else
        /opt/conda/bin/python infer_kitti.py --model superflow --stride $3
    fi
elif [ "$1" == "superflow2" ]; then 
    echo "superflow2"
    if [ "$2" == 'absolute' ]; then 
        /opt/conda/bin/python infer_kitti.py --model superflow2 --mode absolute
    else
        /opt/conda/bin/python infer_kitti.py --model superflow2 --stride $3
    fi
elif [ "$1" == "superglueflow" ]; then 
    echo "superglueflow"
    if [ "$2" == 'absolute' ]; then 
        /opt/conda/bin/python infer_kitti.py --model superglueflow --mode absolute
    else
        /opt/conda/bin/python infer_kitti.py --model superglueflow --stride $3
    fi
elif [ "$1" == "trianflow" ]; then 
    echo "trianflow"
    if [ "$2" == 'absolute' ]; then 
        /opt/conda/bin/python infer_kitti.py --model superglueflow --mode absolute
    else
        /opt/conda/bin/python infer_kitti.py --model trianflow --stride $3
    fi

    
else
    echo "bad model"
fi