import os
import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob
from copy import deepcopy
import data_util
import matplotlib.pyplot as plt
import random



class TestDataset():
    def __init__(self,
                 pose_dir):
        super().__init__()

        all_pose_paths = sorted(glob(os.path.join(pose_dir, '*.txt')))
        self.all_poses = [torch.from_numpy(data_util.load_pose(path)) for path in all_pose_paths]

    def __len__(self):
        return len(self.all_poses)



    def __getitem__(self, idx):
        return self.all_poses[idx]

class TrainDataset():
    def __init__(self,
                 root_dir,
                 img_size=[512,512],
                 num_inpt_views=4,
                 num_trgt_views=1):
        super().__init__()

        self.img_size = img_size
        self.num_inpt_views = num_inpt_views
        self.num_trgt_views = num_trgt_views

        self.color_dir = os.path.join(root_dir, 'rgb')
        self.pose_dir = os.path.join(root_dir, 'pose')

        if not os.path.isdir(self.color_dir):
            print("Error! root dir is wrong")
            return

        self.all_color = sorted(data_util.glob_imgs(self.color_dir))
        self.all_poses = sorted(glob(os.path.join(self.pose_dir, '*.txt')))
        
        print("Buffering files...")
        self.all_views = []
        for i in range(len(self.all_color)):
            if not i % 10:
                print(i)
            self.all_views.append(self.read_view_tuple(i))
            
    def __len__(self):
        return len(self.all_color)//(self.num_inpt_views + self.num_trgt_views)
        
    def load_rgb(self, path):
        img = data_util.load_img(path, square_crop=True, downsampling_order=1, target_size=self.img_size)
        img = img[:, :, :3].astype(np.float32) / 255. - 0.5
        img = img.transpose(2,0,1)
        return img
    
    def read_view_tuple(self, idx):
        gt_rgb = self.load_rgb(self.all_color[idx])
        pose = data_util.load_pose(self.all_poses[idx])

        this_view = {'gt_rgb': torch.from_numpy(gt_rgb),
                     'pose': torch.from_numpy(pose)}
        return this_view
    
    def __getitem__(self, idx):
        
        inpt_views = self.all_views[idx*(self.num_inpt_views+self.num_trgt_views):
                                    (idx+1)*self.num_inpt_views+idx*self.num_trgt_views]
        inpt_views = random.sample(inpt_views, len(inpt_views))
        trgt_views = self.all_views[(idx+1)*self.num_inpt_views+idx*self.num_trgt_views:
                                   (idx+1)*(self.num_inpt_views+self.num_trgt_views)]

        return inpt_views, trgt_views

