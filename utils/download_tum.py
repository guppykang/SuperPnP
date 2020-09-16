#!/usr/bin/env python 

# extract tars

import subprocess
import glob


if __name__ == "__main__":
    # download
    sequences = [
        #train
        'freiburg1/rgbd_dataset_freiburg1_360',
        'freiburg1/rgbd_dataset_freiburg1_floor',
        'freiburg1/rgbd_dataset_freiburg1_room',
        'freiburg1/rgbd_dataset_freiburg1_desk', 
        'freiburg1/rgbd_dataset_freiburg1_desk2',
        'freiburg2/rgbd_dataset_freiburg2_xyz',
        'freiburg1/rgbd_dataset_freiburg1_plant',
        'freiburg1/rgbd_dataset_freiburg1_teddy',
        'freiburg2/rgbd_dataset_freiburg2_coke',
        'freiburg3/rgbd_dataset_freiburg3_teddy',
        'freiburg2/rgbd_dataset_freiburg2_flowerbouquet',
        'freiburg3/rgbd_dataset_freiburg3_sitting_xyz',
        'freiburg3/rgbd_dataset_freiburg3_sitting_halfsphere', 
        'freiburg2/rgbd_dataset_freiburg2_pioneer_slam',
        'freiburg2/rgbd_dataset_freiburg2_pioneer_slam2',
        'freiburg3/rgbd_dataset_freiburg3_nostructure_notexture_far',
        'freiburg3/rgbd_dataset_freiburg3_nostructure_texture_far',
        'freiburg3/rgbd_dataset_freiburg3_structure_notexture_near',
        'freiburg3/rgbd_dataset_freiburg3_structure_texture_near',


        #test
        'freiburg2/rgbd_dataset_freiburg2_desk',
        'freiburg2/rgbd_dataset_freiburg2_360_kidnap',
        'freiburg2/rgbd_dataset_freiburg2_pioneer_360',
        'freiburg2/rgbd_dataset_freiburg2_pioneer_slam3',
        'freiburg3/rgbd_dataset_freiburg3_large_cabinet',
        'freiburg3/rgbd_dataset_freiburg3_sitting_static', 
        'freiburg3/rgbd_dataset_freiburg3_nostructure_notexture_near_withloop',
        'freiburg3/rgbd_dataset_freiburg3_nostructure_texture_near_withloop',
        'freiburg3/rgbd_dataset_freiburg3_structure_notexture_far', 
        'freiburg3/rgbd_dataset_freiburg3_structure_texture_far'
    ]
    # wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
    # wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk2.tgz
    # wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz
    # wget https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz
    # wget https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz
    # wget https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz
    # wget https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_nostructure_texture_far.tgz

    base_path = "https://vision.in.tum.de/rgbd/dataset/"

    if_download = True
    if_untar = True

    if if_download:
        for seq in sequences:
            subprocess.run(f"wget {base_path + seq + '.tgz'}", shell=True, check=True)

    if if_untar:
        # unzip
        tar_files = glob.glob("*.tgz")
        for f in tar_files:
            subprocess.run(f"tar -zxf {f}", shell=True, check=True)