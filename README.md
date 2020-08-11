## SuperPnP  
SuperPnP is a deep pipeline for end to end learning for SFM, and Visual Odometry.  
Developed by Joshua Kang and You-Yi Jau.  
Advised under Manmohan Chandraker. 

## Prereqs
### .bashrc
Must add the root of the repo to the python path (sorry) : 
```bash 
export PYTHONPATH="/home/joshuakang/git/cvlab/SuperPnP:$PYTHONPATH"
```

## TrianNet
### Evaluate VO (flownet correspondences/mask + DepthNet + PnPRansac):
```bash
python infer_vo.py --config_file ./config/odo.yaml --gpu 0 --traj_save_dir_txt /jbk001-data1/kitti_vo/vo_preds/09.txt --sequences_root_dir /jbk001-data1/kitti_vo/vo_dataset/sequences/ --sequence 09 --pretrained_model models/pretrained/kitti_odo.pth
python core/evaluation/eval_odom.py --gt_txt /jbk001-data1/kitti_vo/vo_gts/09.txt --result_txt /jbk001-data1/kitti_vo/vo_preds/09.txt
```

## Proposal  
[Google Slides](https://docs.google.com/presentation/d/1brf3iFONtdu1KqmHxVsGKzNr6s91WSIuEdFgtHnTdQY/edit?usp=sharing)

