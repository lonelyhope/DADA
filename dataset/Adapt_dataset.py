import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
import os, random
import numpy as np
import time

from tools.utils import _all_images, pil_loader
import tools.functional as Func

from .util import *
from constant import *

def filter_one(lst, name='sony_184_'):
    res = []
    for file in lst:
        if name in file:
            res.append(file)
    print('Filter', name, len(res))
    return res

class Adapt_dataset(data.Dataset):
    def __init__(self, opt, path, split='train'):
        super().__init__()
        self.opt = opt

        s_camera = opt.s_camera
        t_camera = opt.t_camera
        self.patch_size = opt.patch_size
        self.scale = opt.scale
        self.rgb_range = opt.rgb_range
        self.split = split
        print('source %s, target %s, split %s'%(CAMERA[s_camera], CAMERA[t_camera], split))
        
        sub_dir = 'train'
        if split == 'test': sub_dir = 'test'
        
        self.hr_list = _all_images(os.path.join(path, sub_dir + '_HR'))
        print('Read hr from', os.path.join(path, sub_dir+'_HR'))
        self.lr_list = _all_images(os.path.join(path, sub_dir + '_LR'))
        print('Read lr from', os.path.join(path, sub_dir+'_LR'))
        
        self.s_lr_list, self.s_hr_list = self.filter_camera(self.lr_list, self.hr_list, s_camera)
        self.t_lr_list, self.t_hr_list = self.filter_camera(self.lr_list, self.hr_list, t_camera)
        
        t_len = opt.t_len
        if split != 'test':
            self.s_lr_list, self.s_hr_list, _ = self.shuffle(self.s_lr_list, self.s_hr_list)
            self.t_lr_list, self.t_hr_list, _ = self.shuffle(self.t_lr_list, self.t_hr_list)
            if t_len == 0: t_len = len(self.t_lr_list)
        if split == 'train':
            self.t_lr_list = self.t_lr_list[VAL_NUM:VAL_NUM+t_len]
            self.t_hr_list = self.t_hr_list[VAL_NUM:VAL_NUM+t_len]
            self.s_lr_list = self.s_lr_list[VAL_NUM:]
            self.s_hr_list = self.s_hr_list[VAL_NUM:]
        elif split == 'valid':
            self.t_lr_list = self.t_lr_list[:opt.valid_num]
            self.t_hr_list = self.t_hr_list[:opt.valid_num]
            self.s_lr_list = self.s_lr_list[:opt.valid_num]
            self.s_hr_list = self.s_hr_list[:opt.valid_num]

        print('source length:', len(self.s_lr_list))
        print('target length:', len(self.t_lr_list))

    def filter_camera(self, lr_list, hr_list, idx):
        filtered_lr, filtered_hr = [], []
        for i, file in enumerate(lr_list):
            if CAMERA[idx] in file:
                filtered_lr.append(file)
                filtered_hr.append(hr_list[i])
            elif idx == 3: # canon is 3+4
                if CAMERA[4] in file:
                    filtered_lr.append(file)
                    filtered_hr.append(hr_list[i])
        return filtered_lr, filtered_hr

    def shuffle(self, lr_list, hr_list, num_lst=None):
        if num_lst is None:
            random.seed(SEED)
            num_lst = list(range(len(lr_list)))
            random.shuffle(num_lst)
        shuffle_lr = np.array(lr_list)[num_lst]
        shuffle_hr = np.array(hr_list)[num_lst]
        return list(shuffle_lr), list(shuffle_hr), num_lst

    def __len__(self):
        return len(self.t_lr_list)

    def __getitem__(self, index):
        data = {}
        random.seed( int(round(time.time() * 1000000)) )
        def _to_tensor(img):
            return Func.to_tensor(img) * self.rgb_range

        t_hr = pil_loader(self.t_hr_list[index], mode="RGB")
        t_lr = pil_loader(self.t_lr_list[index], mode='RGB')

        if self.split == 'train':
            s_idx = random.randint(0, len(self.s_lr_list) - 1)
            s_hr = pil_loader(self.s_hr_list[s_idx], mode="RGB")
            s_lr = pil_loader(self.s_lr_list[s_idx], mode='RGB')
        
            memory = {}
            t_hr, t_lr = random_pre_process_pair(t_hr, t_lr, self.patch_size, self.scale, memory, use_memory=False)
            s_hr, s_lr = random_pre_process_pair(s_hr, s_lr, self.patch_size, self.scale, memory, use_memory=True)
        
            data['s_lr'] = _to_tensor(s_lr)
            data['s_hr'] = _to_tensor(s_hr)
        
        data['t_lr'] = _to_tensor(t_lr)
        data['t_hr'] = _to_tensor(t_hr)
        
        data['hr_path'] = self.t_lr_list[index]
        return data
