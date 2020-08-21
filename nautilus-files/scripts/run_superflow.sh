cd /jbk001-data1/git/SuperPnP/
./setup_nautilus.sh
echo himom
/opt/conda/bin/python infer.py --mode superflow --traj_save_dir /jbk001-data1/kitti_vo/vo_preds/superflow --sequence 09
