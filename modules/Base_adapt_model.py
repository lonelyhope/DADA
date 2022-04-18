import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from .Base_model import Base_model
from .model_tools import get_patch_out, compute_metrics, GANLoss
from .cdc_tools import split_hr_to_fec, GW_loss

def apply_func(objs, func):
    modified = []
    for obj in objs:
        modified.append(func(obj))
    return modified

class Base_adapt_model(Base_model):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        self.criteria = nn.L1Loss().cuda()
        self.d_criteria = GANLoss().cuda()

        self.model = {}
        self.name = 'Base_adapt_model'
        self.tst_sr = True
        
    def add_to_model(self, model_names):
        for name in model_names:
            print('Add %s to model'%(name))
            exec('self.%s = self.%s.cuda()'%(name, name))
            exec('self.model[\'%s\'] = self.%s'%(name, name))

    def get_optimizer(self, model_names, lr=None):
        if lr is None: lr = self.opt.lr

        params = []
        for name in model_names:
            print('Add %s to optimizer'%(name))
            params.append({'params': self.model[name].parameters()})
        return torch.optim.Adam(params, lr)

    def get_optimizer_dict_version(self, dict_):
        params = []
        for name, _ in dict_.items():
            print('Add %s to optimizer'%(name))
            params.append({'params': dict_[name].parameters()})
        return torch.optim.Adam(params, self.opt.lr) 

    def freeze(self, model_names, lb=False):
        for name in model_names:
            for param in self.model[name].parameters():
                param.requires_grad = lb

    def unfreeze(self, model_names):
        self.freeze(model_names, True)

    def freeze_dict_version(self, dict_):
        for name, _ in dict_.items():
            print('Freeze', name)
            for param in dict_[name].parameters():
                param.requires_grad = False

    def say_name(self):
        print('Use adapt model:', self.name)

    def train_(self):
        apply_func(list(self.model.values()), lambda m: m.train())

    def eval_(self):
        apply_func(list(self.model.values()), lambda m: m.eval())

    def get_cdc_sr_loss(self, model, lr_var, hr_var, 
                        SR_map, loss_items, prefix='', inter_supervis=True, gw_loss=True):
        sr_var_ = model.forward_with_map(lr_var, SR_map)
        
        if inter_supervis: 
            coe_list = split_hr_to_fec(hr_var.detach().cpu())
            sr_var = sr_var_
        else:
            sr_var = [sr_var_[-1]]
        
        sr_loss = 0
        inte_loss_weight = [1, 2, 5, 1]
        for i in range(len(sr_var)):
            if i != len(sr_var) - 1:
                coe = coe_list[i]
                single_srloss = inte_loss_weight[i] * self.criteria(coe*sr_var[i], coe*hr_var)
            else:
                if gw_loss:
                    single_srloss = inte_loss_weight[i] * GW_loss(sr_var[i], hr_var)
                else:
                    single_srloss = inte_loss_weight[i] * self.criteria(sr_var[i], hr_var)
            loss_items[prefix+"single_srloss"+str(i)] = single_srloss.item()
            sr_loss += single_srloss
        
        return sr_var_[-1], sr_loss


    def comp_d_loss(self, D, real, fake, loss_items, pre_fix):
        loss_real = self.d_criteria(D(real.detach()), True)
        loss_fake = self.d_criteria(D(fake.detach()), False)
        loss = (loss_real + loss_fake) * 0.5
        loss_items[pre_fix+'real'] = loss_real.item()
        loss_items[pre_fix+'fake'] = loss_fake.item()
        loss.backward()

    def train_d(self, d, optim, real, fake, loss_items, pre_fix):
        optim.zero_grad()
        self.comp_d_loss(d, real, fake, loss_items, pre_fix)
        optim.step()

    def train_a_step(self, data):
        pass

    def test_sr(self, data, forward_func=lambda m, x: m(x)[0][-1]):
        t_lr = data['t_lr']
        t_hr = data['t_hr']

        t_sr = get_patch_out(t_lr, self.model[self.tst_up_name], forward_func=forward_func)
        return t_sr, compute_metrics(t_sr, t_hr, self.opt.scale)

    def test_recon(self, data):
        t_lr = data['t_lr'].cuda()
        
        if self.opt.train:
            t_sr = self.model[self.tst_up_name](t_lr)
            t_lr2 = self.model[self.tst_down_name](t_sr)
        else:
            t_sr = get_patch_out(t_lr, self.model[self.tst_up_name])
            s = 4
            t_lr2 = get_patch_out(t_sr, self.model[self.tst_down_name], sf=1/4, block_h=480*s, block_w=480*s, pw=16*s, ph=16*s)

        t_lr2 = torch.clamp(t_lr2, 0, 1)    
        t_lr = t_lr.cpu()
        return t_lr2, compute_metrics(t_lr2, t_lr, self.opt.scale)

    def test_a_step(self, data, forward_func=lambda m, x: m(x)[0][-1]):
        pass

    def save_model(self, path, break_info):
        model_dict = {}
        for key in self.model:
            model_dict[key] = self.model[key].state_dict()
        ckp = {
            'break_info': break_info,
            'model': model_dict
        }
        torch.save(ckp, path)

    def load_model(self, path, path1=None):
        ckp = torch.load(path, map_location=lambda storage, loc: storage)
        state = ckp['model'][self.tst_up_name]
        # state = ckp['state_dict']
        self.model[self.tst_up_name].load_state_dict(state)
        # for key in state:
        #     print('Load', key)
        #     self.model[key].load_state_dict(state[key])
        print('Load model from', path)
        # return ckp['break_info']
    
    def load_sr_model(self, path, load_key=None):
        ckp = torch.load(path, map_location=lambda storage, loc: storage)
        state = ckp['model']
        key = self.tst_up_name
        if load_key is None:
            self.model[key].load_state_dict(state)
        else:
            self.model[key].load_state_dict(state[load_key])
        print('Load %s from %s'%(key, path))
        return ckp['break_info']

