
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from infer_vo import infer_vo_tum

if __name__ == '__main__':
    vo_test = infer_vo_tum("rgbd_dataset_freiburg3_long_office_household", "./data/tum_data/")
    images = vo_test.load_images()