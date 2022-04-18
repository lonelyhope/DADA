import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
import os, random
import numpy as np
import time

from tools.utils import _all_images, pil_loader
import tools.functional as Func

from . import util
from constant import *

class Dataset(data.Dataset):
    def __init__(self, opt, path, split='train'):
        super().__init__()
        self.opt = opt

        print('Dataset split', split)
        self.split = split
        sub_dir = 'train'
        if split == 'test': sub_dir = 'test'
        self.hr_list = _all_images(os.path.join(path, sub_dir+'_HR'))
        print('Read hr from', os.path.join(path, sub_dir+'_HR'))
        self.lr_list = _all_images(os.path.join(path, sub_dir+'_LR'))
        print('Read lr from', os.path.join(path, sub_dir+'_LR'))
        
        # 在读取训练数据时，先 filter camera，再 shuffle，再选择 train/valid part
        lr_list, hr_list = [], []
        for camera in opt.filter_cameras:
            lists = util.filter_camera(camera, [self.lr_list, self.hr_list])
            if split != 'test':
                lists = util.shuffle(lists)
            if split == 'train':
                lists = [lst[VAL_NUM:(opt.train_num//len(opt.filter_cameras))+VAL_NUM] for lst in lists]
            elif split == 'valid':
                lists = [lst[:opt.valid_num] for lst in lists]    
            lr_list.extend(lists[0])
            hr_list.extend(lists[1])
            print('choose %d of %s'%(len(lists[0]), CAMERA[camera]))
        if len(lr_list):
            self.lr_list, self.hr_list = lr_list, hr_list

        print('SPLIT=%s; Choose %d img total.'%(split, len(self.lr_list)))
        print('Choose camreras:')
        for camera_idx in opt.filter_cameras:
            print(' ', CAMERA[camera_idx])
        print('----------')
        
        self.rgb_range = opt.rgb_range
        self.patch_size = opt.patch_size
        self.scale = opt.scale
        self.ts = util.random_pre_process_pair

    def __len__(self):
        return len(self.lr_list)

    def __getitem__(self, index):
        data = {}
        hr_ = pil_loader(self.hr_list[index], mode='RGB')
        lr_ = pil_loader(self.lr_list[index], mode='RGB')
        
        if self.split == 'train':
            hr, lr = self.ts(hr_, lr_, self.patch_size, self.scale)
        else:
            hr, lr = hr_, lr_
        
        data['lr'] = Func.to_tensor(lr) * self.rgb_range 
        data['hr'] = Func.to_tensor(hr) * self.rgb_range
        
        data['lr_path'] = self.lr_list[index]
        data['hr_path'] = self.hr_list[index]

        return data

