import sys
sys.path.append('../')
from internal.datasets import Dataset
import internal.datasets
from internal import configs, math, utils

import os
import time
import cv2
import numpy as np
import jax
from PIL import Image
import queue
import threading
import json
import gin


def load_dataset(split, train_dir, config):
    "loads a split of a dataset using the data_loader specified by 'config'." 
    dataset_dict = {
        'blender_video': Blender_video,
        'multicam_video': Multicam_video,
    }



def write_meta(meta, config, save_meta=True):
  if save_meta and jax.host_id() == 0:
    os.makedirs(config.checkpoint_dir)
    with open(config.checkpoint_dir + '/meta.txt', 'w') as f:
      # write the meta txt file
      f.write(json.dumps(meta, indent=4))



class Multicam_video(Dataset):

    def _load_renderings(self, config):
        videos = []
        cap = cv2.VideoCapture(self._path)
        self.time_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("we have {self.time_frame_num} frames in total for each videos from each camera.")
        
        if config.render_path:
            raise ValueError('render_path not supported for multicam video dataset.')
        
        # read the meta data:
        with utils.open_file(os.path.join(self._path, 'metadata.json'), 'r') as f:
            print("Now we are loading the metadata from {self.split} split.", self.split)
            self.meta = json.load(f)[self.split]

        self.meta = {k: np.array(self.meta[k]) for k in self.meta}

        # write the self.meta into the out path for saving the meta data, check if it is correct
        # or not???? 

        with utils.open_file(os.path.join(self._path, 'metadata.json'), 'w') as f:
            json.dump(self.meta, f, indent=4)
            





class Blender_video(Dataset):
    def _load_renderings(self, config):
        videos = []