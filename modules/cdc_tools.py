import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


def split_hr_to_fec(hr):
    hr_Y = (0.257*hr[:, :1, :, :] + 0.564*hr[:, 1:2, :, :] + 0.098*hr[:, 2:, :, :] + 16/255.0) * 255.0
    map_corner = hr_Y.new(hr_Y.shape).fill_(0)
    map_edge = hr_Y.new(hr_Y.shape).fill_(0)
    map_flat = hr_Y.new(hr_Y.shape).fill_(0)
    hr_Y_numpy = np.transpose(hr_Y.numpy(), (0, 2, 3, 1))
    for i in range(hr_Y_numpy.shape[0]):
        dst = cv2.cornerHarris(hr_Y_numpy[i, :, :, 0], 3, 3, 0.04)
        thres1 = 0.01*dst.max()
        thres2 = -0.001*dst.max()
        map_corner[i, :, :, :] = torch.from_numpy(np.float32(dst > thres1))
        map_edge[i, :, :, :] = torch.from_numpy(np.float32(dst < thres2))
        map_flat[i, :, :, :] = torch.from_numpy(np.float32((dst > thres2) & (dst < thres1)))
    map_corner = map_corner.cuda()
    map_edge = map_edge.cuda()
    map_flat = map_flat.cuda()
    coe_list = []
    coe_list.append(map_flat)
    coe_list.append(map_edge)
    coe_list.append(map_corner)
    return coe_list

def GW_loss(x1, x2):
    sobel_x = [[-1,0,1],[-2,0,2],[-1,0,1]]
    sobel_y = [[-1,-2,-1],[0,0,0],[1,2,1]]
    b, c, w, h = x1.shape
    sobel_x = torch.FloatTensor(sobel_x).expand(c, 1, 3, 3)
    sobel_y = torch.FloatTensor(sobel_y).expand(c, 1, 3, 3)
    sobel_x = sobel_x.type_as(x1)
    sobel_y = sobel_y.type_as(x1)
    weight_x = nn.Parameter(data=sobel_x, requires_grad=False)
    weight_y = nn.Parameter(data=sobel_y, requires_grad=False)
    Ix1 = F.conv2d(x1, weight_x, stride=1, padding=1, groups=c)
    Ix2 = F.conv2d(x2, weight_x, stride=1, padding=1, groups=c)
    Iy1 = F.conv2d(x1, weight_y, stride=1, padding=1, groups=c)
    Iy2 = F.conv2d(x2, weight_y, stride=1, padding=1, groups=c)
    dx = torch.abs(Ix1 - Ix2)
    dy = torch.abs(Iy1 - Iy2)
#     loss = torch.exp(2*(dx + dy)) * torch.abs(x1 - x2)
    loss = (1 + 4*dx) * (1 + 4*dy) * torch.abs(x1 - x2)
    return torch.mean(loss)


def compute_sr_loss_for_cdc(cdc, lr_var, hr_var, criteria, 
                    loss_items={}, prefix='', inter_supervis=True, pseudo_supervis=False,
                    gw_loss=True, features=None, SR_map_help=None):
    ''' return sr_loss, loss_items, sr_var[-1] '''
    if SR_map_help is not None:
        sr_var = cdc.forward_with_map(lr_var, SR_map_help)
    else:
        sr_var, SR_map = cdc(lr_var, features)
    
    inter_loss_weight = [1, 2, 5, 1]
    if inter_supervis: 
        coe_list = split_hr_to_fec(hr_var.detach().cpu())
    elif pseudo_supervis:
        inter_loss_weight = [1, 1, 1, 1]
    else:
        sr_var = [sr_var[-1]]
    
    sr_loss = 0
    for i in range(len(sr_var)):
        if i != len(sr_var) - 1:
            if inter_supervis:
                coe = coe_list[i]
                single_srloss = inter_loss_weight[i] * criteria(coe*sr_var[i], coe*hr_var)
            elif pseudo_supervis:
                single_srloss = inter_loss_weight[i] * criteria(sr_var[i], hr_var)
        else:
            if gw_loss:
                single_srloss = inter_loss_weight[i] * GW_loss(sr_var[i], hr_var)
            else:
                single_srloss = inter_loss_weight[i] * criteria(sr_var[i], hr_var)
        loss_items[prefix+"single_srloss"+str(i)] = single_srloss.item()
        
        sr_loss += single_srloss
    
    return sr_loss, loss_items, sr_var[-1]


def compute_sr_loss_for_hourglass(model, lr_var, hr_var, criteria, 
                    loss_items={}, prefix='', gw_loss=True):
    ''' return sr_loss, loss_items, sr_var[-1] '''
    sr_vars = model(lr_var)
    
    inter_loss_weight = [0.5, 0.5, 1]
    
    sr_loss = 0
    for i in range(len(sr_vars)):
        if i != len(sr_vars) - 1:
            single_srloss = inter_loss_weight[i] * criteria(sr_vars[i], hr_var)
        else:
            if gw_loss:
                single_srloss = inter_loss_weight[i] * GW_loss(sr_vars[i], hr_var)
            else:
                single_srloss = inter_loss_weight[i] * criteria(sr_vars[i], hr_var)
        loss_items[prefix+"single_srloss"+str(i)] = single_srloss.item()
        
        sr_loss += single_srloss
    
    return sr_loss, loss_items, sr_vars[-1]


def cdc_sr_forward(model, x):
    return model(x)[0][-1]
