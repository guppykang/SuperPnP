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
        image_path = os.path.join(data_dir, folder, 'image_2/')
        dump_image_path = os.path.join(output_dir, folder)
        if not os.path.isdir(dump_image_path):
            os.makedirs(dump_image_path)
        f = open(os.path.join(dump_image_path, 'train.txt'), 'w')
        
        # Note. the os.listdir method returns arbitary order of list. We need correct order.
        numbers = len(os.listdir(image_path))
        for n in range(numbers - stride):
            s_idx = n
            e_idx = s_idx + stride
            curr_image = imageio.imread(os.path.join(image_path, '%.6d'%s_idx)+'.png')
            next_image = imageio.imread(os.path.join(image_path, '%.6d'%e_idx)+'.png')
            seq_images = np.concatenate([curr_image, next_image], axis=0)
            imageio.imsave(os.path.join(dump_image_path, '%.6d'%s_idx)+'.png', seq_images.astype('uint8'))

            # Write training files
            f.write('%s %s\n' % (os.path.join(folder, '%.6d'%s_idx)+'.png', os.path.join(folder, 'calib.txt')))
        print(folder)
    print('Finished processing image pairs')
        
def load_poses(q, gt_dir, output_dir, stride=1):
    f_out = open(os.path.join(output_dir, 'gts.txt'), 'w')
    q.sort()

    for file in q:
        
        f = open(os.path.join(gt_dir, file), 'r')
        pose_lines = f.readlines()
        f.close()

        used_poses = pose_lines[stride:]
        for line in used_poses:
            f_out.write(line)
        
        print(file)
    f_out.close()
    print('Finished processing groundtruth poses')
    pass
    


class KITTI_Odo(object):
    def __init__(self, data_dir, gt_dir):
        self.data_dir = data_dir
        self.gt_dir = gt_dir
        self.train_seqs = ['00','01','02','03','04','05','06','07','08']

    def __len__(self):
        raise NotImplementedError

    def prepare_data_mp(self, output_dir, stride=1):
        output_dir = Path(str(output_dir) + f"_stride{stride}")
        num_processes = 16
        processes = []
        q = mp.Queue()
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
                if d in self.train_seqs and not os.path.isdir(os.path.join(self.data_dir, d)):
                    q.put(d)
            # Process every folder
            print(f'Sequences to process : {q}')
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

                command = 'cp ' + os.path.join(self.data_dir, d, 'calib.txt') + ' ' + os.path.join(output_dir, d, 'calib.txt')
                os.system(command)
           
        if not os.path.isfile(os.path.join(output_dir, 'gts.txt')) and self.gt_dir is not None:
            q = []
            print('Preparing ground truth data....')
            if not os.path.isdir(self.gt_dir):
                raise
            dirlist = os.listdir(self.gt_dir)
            # Get the different folders of images
            for d in dirlist:
                if d[:-4] in self.train_seqs:
                    q.append(d)
            load_poses(q, self.gt_dir, output_dir, stride)

            
        
        print('Data Preparation Finished.')

    def __getitem__(self, idx):
        raise NotImplementedError


if __name__ == '__main__':
    data_dir = '/home4/zhaow/data/kitti'
    dirlist = os.listdir('/home4/zhaow/data/kitti')
    output_dir = '/home4/zhaow/data/kitti_seq/data_generated_s2'
    total_dirlist = []
    # Get the different folders of images
    for d in dirlist:
        seclist = os.listdir(os.path.join(data_dir, d))
        for s in seclist:
            if os.path.isdir(os.path.join(data_dir, d, s)):
                total_dirlist.append(os.path.join(d, s))
    
    F = open(os.path.join(output_dir, 'train.txt'), 'w')
    for p in total_dirlist:
        traintxt = os.path.join(os.path.join(output_dir, p), 'train.txt')
        f = open(traintxt, 'r')
        for line in f.readlines():
            F.write(line)
        print(traintxt)