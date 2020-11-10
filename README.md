## SuperFlow
SuperPnP is a deep pipeline for end to end learning for SFM, and Visual Odometry.  
Developed by Joshua Kang and You-Yi Jau.  
Advised under Manmohan Chandraker. 

## Prereqs
### .bashrc
Must add the root of the repo to the python path (sorry) : 
```bash 
export PYTHONPATH="/home/joshuakang/git/cvlab/SuperPnP:$PYTHONPATH"
```
### Python Packages
- conda env
```
conda create -p /jbk001-data1/yyjau/conda/py36-superpnp_1015 python=3.6 pip
conda activate /jbk001-data1/yyjau/conda/py36-superpnp_1015
```
Must install exact versioning as given in the reqs.txt (most important being torch and cv2):  
```bash 
pip install -r ./requirements_torch.txt
pip install -r ./requirements.txt
pip install -r ./requirements_cv.txt
```
- install [superpoint module](https://github.com/eric-yyjau/pytorch-superpoint.git)
```
git clone https://github.com/eric-yyjau/pytorch-superpoint.git
cd pytorch-superpoint
git checkout module_20200707
# install
pip install --upgrade setuptools wheel
python setup.py bdist_wheel
pip install -e .
```
- Install [deepFEPE module](https://github.com/eric-yyjau/pytorch-deepFEPE.git)
```
export GIT_LFS_SKIP_SMUDGE=1
git clone https://github.com/eric-yyjau/pytorch-deepFEPE.git
git checkout module_20201003
# install
pip install --upgrade setuptools wheel
python setup.py bdist_wheel
pip install -e .
<!-- pip install -r requirements.txt -->
```

### Datasets
This project uses the KITTI VO dataset and the TUM dataset.  
Install KITTI vo from the kitti website, and install TUM from 
```bash
./utils/download_tum.py
```
### One line setup
```bash
./setup_nautilus.sh
```
- install python and local packages
```
./setup.sh
```

## Run inference
- Basic command
    - `model_type`: siftflow, siftflow_deepF, superflow, trianflow
    - `dataset`: kitti, tum
    - `sequence`: kitti [00-10], tum[rgbd_dataset_freiburg2_360_kidnap] 
```
python infer_deepF.py --model <model_type> --sequence <sequence name>  --traj_save_dir ./results/test_model_dataset/kitti/ --iters 10 --dataset <dataset>
```
- ex: siftflow_deepF on kitti
```bash
python infer_deepF.py --model siftflow_deepF --sequence 00  --traj_save_dir ./results//test_model_dataset/kitti/ --iters 10 --dataset kitti
```
- ex: siftflow on tum
```bash
python infer_deepF.py --model siftflow --sequence rgbd_dataset_freiburg2_360_kidnap  --traj_save_dir ./results//test_model_dataset/kitti/ --iters 10 --dataset tum
```

## KITTI (deprecated)
### Inference on superglueflow
To infernce on superglueflow (superpoint + superglue + flownet correspondences) : 
```bash
python infer_kitti --traj_save_dir path/to/save/kitti_vo/preds --sequence 09 --sequence_root_dir /path/to/kitti_vo/dataset --model superglueflow
python ./TrianFlow/core/evaluation/eval_odom.py --gt_txt /path/to/saved/gts.txt --result_txt /path/to/saved/preds.txt
```
### Inference on siftflow
To inference on siftflow (sift + flownet correspondences) : 
```bash
python infer_kitti --traj_save_dir path/to/save/kitti_vo/preds --sequence 09 --sequence_root_dir /path/to/kitti_vo/dataset --model siftflow
python ./TrianFlow/core/evaluation/eval_odom.py --gt_txt /path/to/saved/gts.txt --result_txt /path/to/saved/preds.txt
```

## TUM (deprecated)
### Inference on superglueflow
To infernce on superglueflow (superpoint + superglue + flownet correspondences) : 
```bash
python infer_tum --traj_save_dir path/to/save/tum/preds --sequence 09 --sequence_root_dir /path/to/tum/dataset --model superglueflow

evo_ape tum -s -a /jbk001-data1/datasets/tum/rgbd_dataset_freiburg3_long_office_household/groundtruth.txt /jbk001-data1/datasets/tum/vo_pred/rgbd_dataset_freiburg3_long_office_household/superglueflow/preds_20200828-023715.tum --save_plot /jbk001-data1/datasets/tum/vo_pred/rgbd_dataset_freiburg3_long_office_household/superglueflow/plot_20200828-023715.pdf
```
### Inference on siftflow
To inference on siftflow (sift + flownet correspondences) : 
```bash
python infer_tum --traj_save_dir path/to/save/tum/preds --sequence 09 --sequence_root_dir /path/to/tum/dataset --model siftflow

evo_ape tum -s -a /jbk001-data1/datasets/tum/rgbd_dataset_freiburg3_long_office_household/groundtruth.txt /jbk001-data1/datasets/tum/vo_pred/rgbd_dataset_freiburg3_long_office_household/superglueflow/preds_20200828-023715.tum --save_plot /jbk001-data1/datasets/tum/vo_pred/rgbd_dataset_freiburg3_long_office_household/superglueflow/plot_20200828-023715.pdf
```

### siftflow_scsfm
To inference on siftflow + sc-sfm-learner depthNet (sift + flownet correspondences + scsfm depthNet) : 
- SC-SfMLearner pretrained models are at `/jbk001-data1/git/SuperPnP/TrianFlow/models/pretrained/kitti_odo.pth`
```
python infer_tum.py --model siftflow_scsfm --sequence rgbd_dataset_freiburg2_360_kidnap  --traj_save_dir ./results/test/tum/ --iters 10
```

## DeepF models
### KITTI dataset
```
python infer_deepF.py --model siftflow --sequence 10    --traj_save_dir ./results/test/kitti/ \
--iters 10 --sequences_root_dir /media/yoyee/Big_re/kitti/sequences
```


## Run the code - batch testing and evaluation
### Run the inference
- KITTI
```
python run_eval.py <exp_name> --model siftflow --dataset kitti --run
python run_eval.py test -m siftflow -d kitti --run
# deepF pipeline 
python run_eval.py test_deepF -m siftflow_deepF -d kitti --run 

```
- TUM
```
python run_eval.py <exp_name> --model siftflow --dataset tum --run
python run_eval.py test -m siftflow -d tum --run
```

### Run the evaluation scripts (evo)
```
python run_eval.py test -m siftflow -d tum --eval
```
### Print out evaluation results
- Noted! The folders `rpe_xy` and `ape_xy` should be unzipped.
```
python run_eval.py test -m siftflow -d tum --table
# rpe_xy
python run_eval.py test -m siftflow -d tum --table --metric rpe_xy
```

## Testing (for development)
```
# test models on 2 datasets
pytest tests/infer_deepF_test -v
```
Should see this:
```
==================================================== test session starts ====================================================
platform linux -- Python 3.6.11, pytest-6.1.1, py-1.9.0, pluggy-0.13.1 -- /jbk001-data1/yyjau/conda/py36-superpnp_deepF/bin/python3.6
cachedir: .pytest_cache
rootdir: /jbk001-data1/yyjau/Documents/SuperPnP
collected 4 items

tests/infer_deepF_test/test_model_dataset.py::TestClass::test_xxx[siftflow] PASSED                                    [ 25%]
tests/infer_deepF_test/test_model_dataset.py::TestClass::test_xxx[siftflow_deepF] PASSED                              [ 50%]
tests/infer_deepF_test/test_model_dataset.py::TestClass::test_xxx[superflow] PASSED                                   [ 75%]
tests/infer_deepF_test/test_model_dataset.py::TestClass::test_xxx[trianflow] PASSED                                   [100%]

=============================================== 4 passed in 487.09s (0:08:07) ===============================================
```



## Proposal  
[Google Slides](https://docs.google.com/presentation/d/1brf3iFONtdu1KqmHxVsGKzNr6s91WSIuEdFgtHnTdQY/edit?usp=sharing)

