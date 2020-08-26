#!/bin/bash

#$1 : model, $2 : absolute or relative

cd /jbk001-data1/git/SuperPnP/
./setup_nautilus.sh
echo himom

if [ "$1" == "siftflow" ]; then 
    echo "siftflow"
    if [ "$2" == 'absolute' ]; then 
        echo "not implemented"
#         /opt/conda/bin/python infer_tum.py --model siftflow 
    else
        /opt/conda/bin/python infer_tum.py --model siftflow     
    fi
elif [ "$1" == "superflow2" ]; then 
    echo "superflow2"
    if [ "$2" == 'absolute' ]; then 
        echo "not implemented"
#         /opt/conda/bin/python infer_tum.py --model superflow2 
    else
        /opt/conda/bin/python infer_tum.py --model superflow2     
    fi
elif [ "$1" == "superglueflow" ]; then 
    echo "superglueflow"
    if [ "$2" == 'absolute' ]; then 
        echo "not implemented"
#         /opt/conda/bin/python infer_tum.py --model superglueflow 
    else
        /opt/conda/bin/python infer_tum.py --model superglueflow     
    fi
    
else
    echo "bad model"
fi