import torch
import torch.nn as nn
import torch.nn.functional as F

from .Base_model import Base_model
from .get_model import get_up
from .model_tools import get_patch_out, compute_metrics, load_weights
from .cdc_tools import *


class Up_model_cdc(Base_model):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = get_up(opt)        
        self.optimizer = torch.optim.Adam(self.model.parameters(), opt.lr)
        decay_step = [2e5, 4e5, 6e5, 8e5]
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, 
                            milestones=decay_step,
                            gamma=0.5)
        self.criteria = nn.L1Loss()
        
        self.model = self.model.cuda()
        self.model = load_weights(self.model, '', 1, init_method='kaiming', scale=0.1, just_weight=False, strict=True)
        self.criteria = self.criteria.cuda()

    def adjust_lr(self):
        self.scheduler.step()

    def train_a_step(self, data):
        self.model.train()
        hr = data['hr'].cuda()
        lr = data['lr'].cuda()

        sr_loss, loss_items, _ = compute_sr_loss_for_cdc(self.model, lr, hr, self.criteria, gw_loss=False)
        
        loss = sr_loss
        loss_items['loss'] = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_items

    def test_a_step(self, data):
        lr = data['lr']
        hr = data['hr']
        
        sr = get_patch_out(lr, self.model, forward_func=lambda m, x: m(x)[0][-1] )
        sr = torch.clamp(sr, 0, 1)

        metrics = compute_metrics(sr, hr, self.opt.scale)
        return sr, metrics

    def load_model(self, path):
        p = torch.load(path, map_location=lambda storage, loc: storage)
        if 'state_dict' in p:
            self.model.load_state_dict(p['state_dict'])
            print('Load model from', path)
        else:
            self.model.load_state_dict(p['model'], strict=False)
            print('Load model from', path)
            return p['break_info']

    