cd /jbk001-data1/git/SuperPnP/


if [ $1 = "-d" ]; then
    /jbk001-data1/git/conda/py36-josh-1015/bin/python3.6 run_eval.py latest_test -m superglueflow -d kitti --run -py --iter 10 #/jbk001-data1/yyjau/conda/py36-superpnp_deepF/bin/ 
else
    /jbk001-data1/git/conda/py36-josh-1015/bin/python3.6 run_eval.py latest_test -m superglueflow -d kitti --run #-py /jbk001-data1/yyjau/conda/py36-superpnp_deepF/bin/
fi
# python run_eval.py test_1007 -m siftflow -d kitti --eval
