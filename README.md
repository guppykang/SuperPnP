## SuperPnP  
SuperPnP is a deep pipeline for end to end learning for SFM, and Visual Odometry.  
Developed by Joshua Kang and You-Yi Jau.  
Advised under Manmohan Chandraker. 

##TrianNet
### Evaluate VO (flownet correspondences/mask + DepthNet + PnPRansac):
```bash
python infer_vo.py --config_file ./config/odo.yaml --gpu 0 --traj_save_dir_txt odo_preds.txt --sequences_root_dir /jbk001-data1/dataset/sequences/ --pretrained_model ./models/pretrained/kitti_odo.pth
```

## Proposal  
[Google Slides](https://docs.google.com/presentation/d/1brf3iFONtdu1KqmHxVsGKzNr6s91WSIuEdFgtHnTdQY/edit?usp=sharing)

