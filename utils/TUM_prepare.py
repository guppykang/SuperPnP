import os, sys
import numpy as np
import imageio
from tqdm import tqdm
import torch.multiprocessing as mp
from pathlib import Path
import code

def process_folder(q, data_dir, output_dir, stride=1, do_reverse=False):
    while True:
        if q.empty():
            break
        folder = q.get()
        image_path = os.path.join(data_dir, folder, 'rgb/')
        dump_image_path = os.path.join(output_dir, folder)
        if not os.path.isdir(dump_image_path):
            os.makedirs(dump_image_path)
        f = open(os.path.join(dump_image_path, 'train.txt'), 'w')
        
        # Note. the os.listdir method returns arbitary order of list. We need correct order.
        numbers = len(os.listdir(image_path))
        paths = os.listdir(image_path)
        paths.sort()
        
        for n in range(numbers - stride):
            s_idx = n
            e_idx = s_idx + stride
            curr_image = imageio.imread(os.path.join(image_path, paths[s_idx]))
            next_image = imageio.imread(os.path.join(image_path, paths[e_idx]))
            seq_images = np.concatenate([curr_image, next_image], axis=0)
            imageio.imsave(os.path.join(dump_image_path, '%.6d'%s_idx)+'.png', seq_images.astype('uint8'))

            # Write training files
            filename = '%.6d'%s_idx            
            f.write(f'{os.path.join(folder, filename)}.png\n')
        print(f'Done processing data dir : {folder}')
        
def load_poses(q, data_dir, output_dir, stride=1):
    f_out = open(os.path.join(output_dir, 'gts.txt'), 'w')
    q.sort()

    for seq in q:
        
        f = open(os.path.join(data_dir, seq, 'groundtruth.txt'), 'r')
        pose_lines = f.readlines()
        f.close()

        used_poses = pose_lines[stride+3:]
        for line in used_poses:
            f_out.write(line)
        
        print(seq)
    f_out.close()
    print('Finished processing groundtruth poses')
    pass
    


class TUM_Prepare(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_seqs = [
            #REFER TO utils/download_tum.py
            'rgbd_dataset_freiburg1_360',
            'rgbd_dataset_freiburg1_floor',
            'rgbd_dataset_freiburg1_room',
            'rgbd_dataset_freiburg1_desk', 
            'rgbd_dataset_freiburg1_desk2',
            'rgbd_dataset_freiburg2_xyz',
            'rgbd_dataset_freiburg1_plant',
            'rgbd_dataset_freiburg1_teddy',
            'rgbd_dataset_freiburg2_coke',
            'rgbd_dataset_freiburg3_teddy',
            'rgbd_dataset_freiburg2_flowerbouquet',
            'rgbd_dataset_freiburg3_sitting_xyz',
            'rgbd_dataset_freiburg3_sitting_halfsphere', 
            'rgbd_dataset_freiburg2_pioneer_slam',
            'rgbd_dataset_freiburg2_pioneer_slam2',
            'rgbd_dataset_freiburg3_nostructure_notexture_far',
            'rgbd_dataset_freiburg3_nostructure_texture_far',
            'rgbd_dataset_freiburg3_structure_notexture_near',
            'rgbd_dataset_freiburg3_structure_texture_near'
        ]

    def __len__(self):
        raise NotImplementedError

    def prepare_data_mp(self, output_dir, stride=1):
        output_dir = Path(str(output_dir) + f"_stride{stride}")
        num_processes = 16
        processes = []
        q = mp.Queue()
        #image pairs
        if not os.path.isfile(os.path.join(output_dir, 'train.txt')):
            os.makedirs(output_dir)
            #f = open(os.path.join(output_dir, 'train.txt'), 'w')
            print('Preparing sequence data....')
            if not os.path.isdir(self.data_dir):
                raise
            dirlist = os.listdir(self.data_dir)
            total_dirlist = []
            # Get the different folders of images
            for d in dirlist:
                if d in self.train_seqs:
                    q.put(d)
                    print(f'Sequence {d} to process')
            # Process every folder
            for rank in range(num_processes):
                p = mp.Process(target=process_folder, args=(q, self.data_dir, output_dir, stride))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        
            f = open(os.path.join(output_dir, 'train.txt'), 'w')
            for d in self.train_seqs:
                train_file = open(os.path.join(output_dir, d, 'train.txt'), 'r')
                for l in train_file.readlines():
                    f.write(l)

        #gt poses
        if not os.path.isfile(os.path.join(output_dir, 'gts.txt')):
            q = []
            print('Preparing ground truth data....')
            if not os.path.isdir(self.data_dir):
                raise
            dirlist = os.listdir(self.data_dir)
            # Get the different folders of images
            
            for d in dirlist:
                if d in self.train_seqs:
                    q.append(d)
            load_poses(q, self.data_dir, output_dir, stride)

            
        
        print('Data Preparation Finished.')

    def __getitem__(self, idx):
        raise NotImplementedError

