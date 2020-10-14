cd /jbk001-data1/yyjau/Documents/SuperPnP/

# ModelList="superflow siftflow siftflow_deepF siftflow_scsfm superglueflow_scsfm superglueflow trianflow"
ModelList="siftflow siftflow_deepF"
DatasetList="kitti tum"

# Iterate the string variable using for loop
for model in $ModelList; do
    for dataset in $DatasetList; do
        echo $model $dataset
        /jbk001-data1/yyjau/conda/py36-superpnp_deepF/bin/python run_eval.py infer_vo_test.sh -m $model -d $dataset --run -py /jbk001-data1/yyjau/conda/py36-superpnp_deepF/bin/ --iter 10 --debug
    done
done

# if [ $1 = "-d" ]; then
#     /jbk001-data1/yyjau/conda/py36-superpnp_deepF/bin/python run_eval.py test_ran1k_1007 -m siftflow_deepF -d tum --run -py /jbk001-data1/yyjau/conda/py36-superpnp_deepF/bin/ --iter 10
# else
#     /jbk001-data1/yyjau/conda/py36-superpnp_deepF/bin/python run_eval.py test_ran1k_1007 -m siftflow_deepF -d tum --run -py /jbk001-data1/yyjau/conda/py36-superpnp_deepF/bin/
# fi
# python run_eval.py test_deepF_1007 -m siftflow -d kitti --eval

