cd /jbk001-data1/yyjau/Documents/SuperPnP/

if [ $1 = "-d" ]; then
    /jbk001-data1/yyjau/conda/py36-superpnp_deepF/bin/python run_eval.py test_deepF_2k_1007 -m siftflow -d kitti --run --deepF -py /jbk001-data1/yyjau/conda/py36-superpnp_deepF/bin/ --iter 10
else
    /jbk001-data1/yyjau/conda/py36-superpnp_deepF/bin/python run_eval.py test_deepF_2k_1007 -m siftflow -d kitti --run --deepF -py /jbk001-data1/yyjau/conda/py36-superpnp_deepF/bin/
fi
# python run_eval.py test_deepF_1007 -m siftflow -d kitti --eval
