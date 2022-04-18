from os import getpgid
import torch
import torch.nn.functional as F

from .cdc_tools import compute_sr_loss_for_cdc
from .get_model import get_up, get_down, get_D

from .model_tools import compute_metrics, get_patch_out, to_ycbcr
from tools.perceptual_loss import PerceptualLoss
from .Base_adapt_model import Base_adapt_model


class Adaptation(Base_adapt_model):
    def __init__(self, opt):
        super().__init__(opt)
        self.up_0 = get_up(opt).cuda()
        self.up_s = get_up(opt)
        self.up_t = get_up(opt)

        self.down_t = get_down(opt)
        self.down_s = get_down(opt)
        keys = ['up_s', 'up_t', 'down_t', 'down_s']
        self.add_to_model(keys)
        
        if opt.train:
            self.optimizer = self.get_optimizer(keys)
        self.freeze_dict_version({'up_0': self.up_0})
        
        if opt.train:
            in_dim = 1
            if 'in_dim' in opt: in_dim = opt.in_dim
            self.d_s = get_D(opt, in_dim)   # 判断 s_model 的输入来自于 s/t
            self.d_t = get_D(opt, in_dim)   # 判断 t_model 的输入来自于 s/t
            in_dim = 3
            self.d_sm = get_D(opt, in_dim)  # 判断 s 的输出来自 up_s/up_t
            self.d_tm = get_D(opt, in_dim)  # 判断 d 的输出来自 up_s/up_t

            self.add_to_model(['d_s', 'd_t', 'd_sm', 'd_tm'])
            self.optimizer_ds = self.get_optimizer(['d_s'])
            self.optimizer_dt = self.get_optimizer(['d_t'])
            self.optimizer_dsm = self.get_optimizer(['d_sm'])
            self.optimizer_dtm = self.get_optimizer(['d_tm'])

        self.name = 'Adaptation'
        self.ps_criteria = PerceptualLoss()

    def adjust_lr(self):
        pass
        
    def init_all(self, path):
        ckp = torch.load(path, map_location=lambda storage, loc: storage)
        up = ckp['model']['up']
        self.up_s.load_state_dict(up)
        self.up_t.load_state_dict(up)
        self.up_0.load_state_dict(up)
        print('Load up_s, up_t and up_0 from', path)

        down = ckp['model']['down']
        self.down_t.load_state_dict(down)
        self.down_s.load_state_dict(down)
        print('Load down_t and down_s from', path)

        return up

    def load_model(self, path1, path0):
        ckp = torch.load(path1, map_location=lambda storage, loc: storage)['model']
        up_t = ckp['up_t']
        down_t = ckp['down_t']
        self.up_t.load_state_dict(up_t)
        self.down_t.load_state_dict(down_t)
        print('load upt and downt from', path1)

        up_s = ckp['up_s']
        down_s = ckp['down_s']
        self.up_s.load_state_dict(up_s)
        self.down_s.load_state_dict(down_s)
        print('load ups and downs from', path1)

        up_0 = torch.load(path0, map_location=lambda storage, loc: storage)['model']['up']
        self.up_0.load_state_dict(up_0)
        print('load up0 from', path0)
        return ckp

    def get_cdc_sr(self, cdc, lr, map):
        result = cdc.forward_with_map(lr, map)
        return result[-1]

    def train_a_step(self, data):
        self.train_()
        self.optimizer.zero_grad()

        lr_s, hr_s = data['s_lr'].cuda(), data['s_hr'].cuda()
        lr_t = data['t_lr'].cuda()

        loss_items = {}

        self.freeze(['d_s', 'd_t', 'd_sm', 'd_tm'])

        sr_t_in0, map_t_in0 = self.up_0(lr_t)
        sr_s_in0, map_s_in0 = self.up_0(lr_s)

        sr_s_ins_loss, _, sr_s_ins = compute_sr_loss_for_cdc(self.up_s, lr_s, hr_s, self.criteria, 
                    loss_items, prefix='s_', inter_supervis=True, gw_loss=True, SR_map_help=map_s_in0)
        # sr_s_int = self.get_cdc_sr(self.up_t, lr_s, map_s_in0)
        sr_s_int_loss, _, sr_s_int = compute_sr_loss_for_cdc(self.up_t, lr_s, hr_s, self.criteria, 
                    loss_items, prefix='s_int_', inter_supervis=True, gw_loss=True, SR_map_help=map_s_in0)
        
        sr_t_ins = self.get_cdc_sr(self.up_s, lr_t, map_t_in0)
        sr_t_int_loss, _, sr_t_int = compute_sr_loss_for_cdc(self.up_t, lr_t, sr_t_ins, self.criteria, 
                    loss_items, prefix='t_', inter_supervis=True, gw_loss=True, SR_map_help=map_t_in0)

        sr_t_bic = sr_t_in0[-1]
        sr_t_ps_loss, _ = self.ps_criteria(sr_t_int, sr_t_bic)

        sr_s_int_m_gan_loss = self.d_criteria(self.d_sm(sr_s_int), True)
        sr_t_int_m_gan_loss = self.d_criteria(self.d_tm(sr_t_int), True)

        sr_t_ins_y = to_ycbcr(sr_t_ins)
        sr_t_int_y = to_ycbcr(sr_t_int)
        sr_t_ins_gan_loss = self.d_criteria(self.d_s(sr_t_ins_y), True)
        sr_t_int_gan_loss = self.d_criteria(self.d_t(sr_t_int_y), True)

        lr_t_int = self.down_t(sr_t_int)
        lr_t_int_loss = self.criteria(lr_t_int, lr_t)
        lr_t_ins = self.down_s(sr_t_ins)
        lr_t_ins_loss = self.criteria(lr_t_ins, lr_t)

        gan_w = 0.005
        loss = sr_s_ins_loss + sr_s_int_loss * 0.05 + \
               sr_t_int_loss + \
               sr_t_ins_gan_loss * gan_w + \
               sr_t_int_gan_loss * gan_w + \
               sr_t_int_m_gan_loss * gan_w + \
               sr_s_int_m_gan_loss * gan_w + \
               lr_t_int_loss * 0.1 + lr_t_ins_loss * 0.1 + \
               sr_t_ps_loss * 0.01

        loss_items['sr_s_ins_loss'] = sr_s_ins_loss.item()
        loss_items['sr_t_int_loss'] = sr_t_int_loss.item()
        loss_items['sr_s_int_loss'] = sr_s_int_loss.item()
        loss_items['sr_t_ins_gan_loss'] = sr_t_ins_gan_loss.item()
        loss_items['sr_t_int_gan_loss'] = sr_t_int_gan_loss.item()
        loss_items['sr_t_int_m_gan_loss'] = sr_t_int_m_gan_loss.item()
        loss_items['sr_s_int_m_gan_loss'] = sr_s_int_m_gan_loss.item()
        loss_items['sr_t_ps_loss'] = sr_t_ps_loss.item()
        loss_items['lr_t_int_loss'] = lr_t_int_loss.item()
        loss_items['loss'] = loss.item()


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # train D
        self.unfreeze(['d_s', 'd_t', 'd_sm', 'd_tm'])
        sr_s_ins_y = to_ycbcr(sr_s_ins)
        sr_s_int_y = to_ycbcr(sr_s_int)
        self.train_d(self.d_s, self.optimizer_ds, sr_s_ins_y, sr_t_ins_y, loss_items, 'ds_')
        self.train_d(self.d_t, self.optimizer_dt, sr_s_int_y, sr_t_int_y, loss_items, 'dt_')
        self.train_d(self.d_sm, self.optimizer_dsm, sr_s_ins, sr_s_int, loss_items, 'dsm_')
        self.train_d(self.d_tm, self.optimizer_dtm, sr_t_ins, sr_t_int, loss_items, 'dtm_')
        
        return loss_items

    def test_a_step(self, data):
        t_lr = data['t_lr']
        t_lr = t_lr.cuda()

        forward_func = lambda m, x: m.forward_with_map(x, self.up_0(x)[1])[-1]
        t_sr = get_patch_out(t_lr, self.up_t, forward_func=forward_func)
        
        t_hr = data['t_hr']
        return t_sr, compute_metrics(t_sr.cuda(), t_hr.cuda(), self.opt.scale)
