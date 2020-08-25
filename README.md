## Prereqs
### .bashrc
Must add the root of the repo to the python path (sorry) : 
```bash 
export PYTHONPATH="/home/joshuakang/git/cvlab/SuperPnP:$PYTHONPATH"
```
### Datasets. 
This project uses the KITTI VO dataset and the TUM dataset.  
Install KITTI vo from the kitti website, and install TUM from 
```bash
./utils/download_tum.py
```


## SuperFlow
SuperPnP is a deep pipeline for end to end learning for SFM, and Visual Odometry.  
Developed by Joshua Kang and You-Yi Jau.  
Advised under Manmohan Chandraker. 
### Inference on superflow
To infernce on superflow (superpoint + flownet correspondences) : 
```bash
./infer_pipeline --traj_save_dir path/to/save/kitti_vo/preds --sequence 09 --sequence_root_dir /path/to/kitti_vo/dataset
```
### Inference on siftflow
To inference on siftflow (sift + flownet correspondences) : 
```bash
./infer_siftflow --traj_save_dir path/to/save/kitti_vo/preds --sequence 09 --sequence_root_dir /path/to/kitti_vo/dataset
```




## Proposal  
[Google Slides](https://docs.google.com/presentation/d/1brf3iFONtdu1KqmHxVsGKzNr6s91WSIuEdFgtHnTdQY/edit?usp=sharing)

