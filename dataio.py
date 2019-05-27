import os
import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob
from copy import deepcopy
import data_util
import matplotlib.pyplot as plt
import random

from sys import getsizeof



class TestDataset():
    def __init__(self,
                 root_dir):
        super().__init__()

        pose_dir = os.path.join(root_dir, 'test_pose')
        all_pose_paths = sorted(glob(os.path.join(pose_dir, '*.txt')))
        self.all_poses = [torch.from_numpy(data_util.load_pose(path)) for path in all_pose_paths]

    def __len__(self):
        return len(self.all_poses)



    def __getitem__(self, idx):
        return self.all_poses[idx]

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_dir,
                 img_size=[512,512],
                 num_inpt_views=4,
                 num_trgt_views=1,
                 num_views=10):
        super().__init__()

        self.img_size = img_size
        self.num_inpt_views = num_inpt_views
        self.num_trgt_views = num_trgt_views
        self.num_views = num_views

        self.color_dir = os.path.join(root_dir, 'train_rgb')
        self.pose_dir = os.path.join(root_dir, 'train_pose')

        if not os.path.isdir(self.color_dir):
            print("Error! root dir is wrong")
            return

        self.all_color = sorted(data_util.glob_imgs(self.color_dir))
        self.all_poses = sorted(glob(os.path.join(self.pose_dir, '*.txt')))
        
            
    def __len__(self):
        return len(self.all_color)//self.num_views
        
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
        
        views = list()
        for i in range(idx*self.num_views, idx*self.num_views+
                         self.num_inpt_views+self.num_trgt_views):
            views.append(self.read_view_tuple(i)) 
        
        views = random.sample(views, len(views))
        
        inpt_views = views[:self.num_inpt_views]
        trgt_views = views[self.num_inpt_views:]

        return inpt_views, trgt_views
    
class PretestDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_dir,
                 img_size=[512,512],
                 num_inpt_views=4,
                 ttl_num_views=20):
        super().__init__()

        self.img_size = img_size
        self.num_inpt_views = num_inpt_views
        self.ttl_num_views = ttl_num_views

        self.color_dir = os.path.join(root_dir, 'pretest_rgb')
        self.pose_dir = os.path.join(root_dir, 'pretest_pose')

        if not os.path.isdir(self.color_dir):
            print("Error! root dir is wrong")
            return

        self.all_color = sorted(data_util.glob_imgs(self.color_dir))
        self.all_poses = sorted(glob(os.path.join(self.pose_dir, '*.txt')))
        
            
    def __len__(self):
        return self.ttl_num_views//self.num_inpt_views
        
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
        
        views = list()
        for i in range(idx*self.num_inpt_views, (idx+1)*self.num_inpt_views):
            views.append(self.read_view_tuple(i)) 
        
        views = random.sample(views, len(views))
        return views