#!/usr/bin/env python 
from models.corresondence_model import SuperFlow

import argparse
import yaml
import code

def test_inference(cfg):
    pass


class pObject(object):
    def __init__(self):
        pass

# def load_images(self):
#         path = self.img_dir
#         seq = self.seq_id
#         new_img_h = self.new_img_h
#         new_img_w = self.new_img_w
#         seq_dir = os.path.join(path, seq)
#         image_dir = os.path.join(seq_dir, 'image_2')
#         num = len(os.listdir(image_dir))
#         images = []
#         for i in range(num):
#             image = cv2.imread(os.path.join(image_dir, '%.6d'%i)+'.png')
#             image = cv2.resize(image, (new_img_w, new_img_h))
#             images.append(image)
#         return images

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Unit Tests for Correspondence_model")
    arg_parser.add_argument('-c', '--config_file', default='./../../configs/train.yaml', help='config file.')
    arg_parser.add_argument('-g', '--gpu', type=str, default=0, help='gpu id.')
    args = arg_parser.parse_args()

    #do config stuff
    with open(args.config_file, "r") as f:
        cfg = yaml.load(f)
        #create the model

    #cfg setup (cuz why not)
    trianflow_cfg = pObject()
    for attr in list(cfg["models"]["trianflow"].keys()):
        setattr(trianflow_cfg, attr, cfg["models"]["trianflow"][attr])

    cfg = { 'trianflow' : trianflow_cfg }

    model = SuperFlow(cfg)


    print('pass')