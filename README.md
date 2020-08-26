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
Must install exact versioning as given in the reqs.txt (most important being torch and cv2):  
```bash 
pip install -r ./requirements_torch.txt
pip install -r ./requirements.txt
```
### Datasets. 
This project uses the KITTI VO dataset and the TUM dataset.  
Install KITTI vo from the kitti website, and install TUM from 
```bash
./utils/download_tum.py
```

## KITTI
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

## TUM
### Inference on superglueflow
To infernce on superglueflow (superpoint + superglue + flownet correspondences) : 
```bash
python infer_tum --traj_save_dir path/to/save/tum/preds --sequence 09 --sequence_root_dir /path/to/tum/dataset --model superglueflow

evo_ape tum -s --align /jbk001-data1/datasets/tum/rgbd_dataset_freiburg3_long_office_household/groundtruth.txt  /jbk001-data1/datasets/tum/vo_pred/rgbd_dataset_freiburg3_long_office_household_superglueflow_20200826-060456.txt
```
### Inference on siftflow
To inference on siftflow (sift + flownet correspondences) : 
```bash
python infer_tum --traj_save_dir path/to/save/tum/preds --sequence 09 --sequence_root_dir /path/to/tum/dataset --model siftflow

evo_ape tum -s --align /jbk001-data1/datasets/tum/rgbd_dataset_freiburg3_long_office_household/groundtruth.txt  /jbk001-data1/datasets/tum/vo_pred/rgbd_dataset_freiburg3_long_office_household_siftflow_20200826-062406.txt
```



## Proposal  
[Google Slides](https://docs.google.com/presentation/d/1brf3iFONtdu1KqmHxVsGKzNr6s91WSIuEdFgtHnTdQY/edit?usp=sharing)

