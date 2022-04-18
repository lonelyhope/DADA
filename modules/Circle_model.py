import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from .Base_model import Base_model
from .get_model import get_up, get_down
from .model_tools import get_patch_out, compute_metrics
from .cdc_tools import *

class Circle_model(Base_model):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        self.up = get_up(opt)        
        self.down = get_down(opt)

        if opt.train:
            self.optimizer = torch.optim.Adam([
                {'params': self.up.parameters()},
                {'params': self.down.parameters()}
            ], opt.lr)
            
            self.sr_weight = 1
            self.lr_weight = 1

            decay_step = [2e5, 4e5, 6e5, 8e5]
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, 
                                milestones=decay_step,
                                gamma=0.5)
            self.criteria = nn.L1Loss()
            self.criteria = self.criteria.cuda()
            self.down = self.down.cuda()
        
        self.up = self.up.cuda()

    def eval_(self):
        self.up.eval()
        self.down.eval()

    def load_model(self, path):
        p = torch.load(path, map_location=lambda storage, loc: storage)
        state = p['model']['up']
        # state = p['state_dict']
        self.up.load_state_dict(state)

    def save_model(self, path, break_info):
        model_dict = {
            'up': self.up.state_dict(),
            'down': self.down.state_dict()
        }
        ckp = {
            'break_info': break_info,
            'model': model_dict
        }
        torch.save(ckp, path)

    def train_a_step(self, data):
        self.up.train()
        self.down.train()

        hr = data['hr'].cuda()
        lr = data['lr'].cuda()
        
        loss_items = {}
        
        sr = self.up(lr)
        sr_loss = self.criteria(sr, hr)
        lr_recon = self.down(sr)
        recon_loss = self.criteria(lr_recon, lr)
        loss = sr_loss * self.sr_weight + recon_loss * self.lr_weight

        loss_items['sr_loss'] = sr_loss.item()
        loss_items['recon_loss'] = recon_loss.item()
        loss_items['loss'] = loss.item()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_items
    
    def load_up(self, path):
        ckp = torch.load(path, map_location=lambda storage, loc: storage)
        state = ckp['model']
        self.up.load_state_dict(state)
        print('Load up model from', path)

    def test_recon(self, data):
        lr = data['lr'].cuda()
        
        sr = self.up(lr)
        lr_recon = self.down(sr)

        metrics = compute_metrics(lr_recon, lr, 1)
        return sr, metrics


    def test_a_step(self, data):
        if self.opt.train: return self.test_recon(data)

        lr = data['lr'].cuda()
        hr = data['hr']
        
        sr = get_patch_out(lr, self.up) # , sf=4, block_h=360, block_w=360, pw=12, ph=12)
        sr = torch.clamp(sr, 0, 1)

        metrics = compute_metrics(sr, hr, self.opt.scale)
        
        return sr, metrics

class Circle_model_cdc(Circle_model):
    def __init__(self, opt):
        super().__init__(opt)
        self.train_down_only = False
        params = [{'params': self.down.parameters()}]
        if not self.train_down_only:
            params.append({'params': self.up.parameters()})
        self.optimizer = torch.optim.Adam(params, opt.lr)

        self.sr_weight = 1
        self.lr_weight = 1

    def get_sr_loss(self, lr, hr, loss_items):
        sr_loss, loss_items, sr = \
            compute_sr_loss_for_cdc(self.up, lr, hr, self.criteria, loss_items)
        return sr, sr_loss

    def train_a_step(self, data):
        self.up.train()
        self.down.train()

        hr = data['hr'].cuda()
        lr = data['lr'].cuda()
        
        loss_items = {}
        loss = 0
        if not self.train_down_only:
            sr, sr_loss = self.get_sr_loss(lr, hr, loss_items)
            loss += self.sr_weight * sr_loss
            loss_items['sr_loss'] = sr_loss.item()
        else:
            sr = self.up(lr)[0][-1]

        lr_recon = self.down(sr)
        recon_loss = self.criteria(lr_recon, lr)
        loss += recon_loss * self.lr_weight

        loss_items['recon_loss'] = recon_loss.item()
        loss_items['loss'] = loss.item()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_items

    def load_up(self, path):
        ckp = torch.load(path, map_location=lambda storage, loc: storage)
        # state = ckp['state_dict']
        state = ckp['model']
        self.up.load_state_dict(state)
        print('Load up model from', path)

    def test_recon(self, data):
        lr = data['lr']
        size = 256
        lr = lr[:, :, :size, :size]
        
        sr = self.up(lr.cuda())[0][-1]
        lr_recon = self.down(sr)
        lr_recon = torch.clamp(lr_recon, 0, 1)
        lr_recon = lr_recon.cpu()

        metrics = compute_metrics(lr_recon, lr, 1)
        
        return lr_recon, metrics

    def test_sr(self, data):
        lr = data['lr']
        hr = data['hr']

        sr = get_patch_out(lr, self.up, forward_func=cdc_sr_forward)
        sr = torch.clamp(sr, 0, 1)

        metrics = compute_metrics(sr, hr, self.opt.scale)
        
        return sr, metrics

    def test_a_step(self, data):
        if self.opt.train or self.opt.test_recon:
            return self.test_recon(data)
        else:
            return self.test_sr(data)
