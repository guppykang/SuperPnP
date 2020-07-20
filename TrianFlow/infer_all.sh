for VARIABLE in 00  01  02  03  04  05  06  07  08  09  10  11  12  13  14  15  16  17  18  19  20  21
do
    echo evaluating sequence $VARIABLE
    python infer_vo.py --config_file ./config/odo.yaml --gpu 0 --traj_save_dir_txt /jbk001-data1/kitti/vo_preds/$VARIABLE.txt --sequences_root_dir /jbk001-data1/dataset/sequences/ --pretrained_model ./models/pretrained/kitti_odo.pth  
    python core/evaluation/eval_odom.py --gt_txt /jbk001-data1/kitti/vo_gts/$VARIABLE.txt --result_txt /jbk001-data1/kitti/vo_preds/$VARIABLE.txt
done