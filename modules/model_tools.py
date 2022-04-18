import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.nn import init

import sys

from tools import Metrics, pytorch_ssim


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)

def load_weights(model, weights='', gpus=1, init_method='kaiming', strict=True, scale=0.1, resume=False, just_weight=False):
    """
    load model from weights, remove "module" if weights is dataparallel
    :param model:
    :param weights:
    :param gpus:
    :param init_method:
    :return:
    """
    # Initiate Weights
    if weights == '':
        print('Training from scratch......')
        if init_method == 'xavier':
            model.apply(weights_init_xavier)
        elif init_method == 'kaiming':
            weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
            model.apply(weights_init_kaiming_)
        else:
            model.apply(weights_init_xavier)
        print('Init weights with %s' % init_method)
    # Load Pre-train Model
    else:
        model_weights = torch.load(weights)
        # print('checkpoint keys:', model_weights['state_dict'].keys())
        # input()
        # print('model keys:', model.state_dict().keys())
        # input()
        if not just_weight:
            model_weights = model_weights['optim'] if resume else model_weights['state_dict']
        model.load_state_dict(model_weights, strict=strict)
        # try:
        #     print('Try load')
        #     model.load_state_dict(model_weights, strict=strict)
        #     input()
        # except:
        #     input()
        #     print('Loading from DataParallel module......')
        #     model = _rm_module(model, model_weights)
        print('Loading %s success.....' % weights)
    if gpus > 1:
        model = nn.DataParallel(model, device_ids=[i for i in range(gpus)])
    sys.stdout.flush()
    return model


def get_patch_out(img, model, sf=4, block_h=480, block_w=480, pw=16, ph=16, 
    forward_func=lambda m, x: m(x)):
    if block_h < 100:
        return get_patch_out_batch(img, model, sf, block_h, block_w, pw, ph, forward_func)

    def _f(x):
        return int(x * sf)
    n, c, h, w = img.shape
    if h * min(1, sf) < 100 and w * min(1, sf) < 100:
        return forward_func(model, img.cuda())

    padding_img = F.pad(img, (pw, pw, ph, ph), 'replicate')
    inp_h, inp_w = block_h + ph * 2, block_w + pw * 2
    out_shape = (n, c, _f(h), _f(w))
    out_img = torch.zeros(out_shape)
    for i in range(0, h, block_h):
        for j in range(0, w, block_w):
            ei = min(i + block_h, h)
            ej = min(j + block_w, w)
            input_block = padding_img[:, :, i:ei+ph*2, j:ej+pw*2]
            n, c, bh, bw = input_block.shape
            if (bh < inp_h) or (bw < inp_w):
                input_block = F.pad(input_block, (0, inp_w - bw, 0, inp_h - bh), 'replicate')
            out_block = forward_func(model, input_block.cuda()).cpu()
            out_img[:, :, _f(i):_f(ei), _f(j):_f(ej)] = \
                out_block[:, :, _f(ph):_f(ph+ei-i), _f(pw):_f(pw+ej-j)]
    return out_img

batch = False
def get_patch_out_batch(img, model, sf=4, block_h=38, block_w=38, pw=5, ph=5, 
    forward_func=lambda m, x: m(x)):
    # get patch out 的 batch forward 版本
    global batch
    if not batch:
        print('Use get_patch_out_batch')
        batch = True
    
    def _f(x):
        return int(x * sf)
    
    n, c, h, w = img.shape
    padding_img = F.pad(img, (pw, pw, ph, ph), 'replicate')
    inp_h, inp_w = block_h + ph * 2, block_w + pw * 2
    out_shape = (n, c, _f(h), _f(w))
    out_img = torch.zeros(out_shape)

    batch_size = 64
    batch_idx = 0
    input_blocks = []
    block_idxes = []
    for i in range(0, h, block_h):
        for j in range(0, w, block_w):
            ei = min(i + block_h, h)
            ej = min(j + block_w, w)
            
            input_block = padding_img[:, :, i:ei+ph*2, j:ej+pw*2]
            n, c, bh, bw = input_block.shape
            if (bh < inp_h) or (bw < inp_w):
                input_block = F.pad(input_block, (0, inp_w - bw, 0, inp_h - bh), 'replicate')

            input_blocks.append(input_block)
            block_idxes.append([i, j, ei, ej])
            batch_idx = batch_idx + 1
            
            if batch_idx == batch_size or (ei == h and ej == w):
                inp = torch.cat(input_blocks, 0)
                out = forward_func(model, inp.cuda()).cpu()
                for k, idx in enumerate(block_idxes):
                    i, j, ei, ej = idx
                    out_block = out[k]
                    out_img[0, :, _f(i):_f(ei), _f(j):_f(ej)] = \
                        out_block[:, _f(ph):_f(ph+ei-i), _f(pw):_f(pw+ej-j)]
                
                batch_idx = 0
                input_blocks = []
                block_idxes = []

                        
    return out_img


from PerceptualSimilarity import perceptual_similarity
ps_loss = None
def compute_metrics(sr, hr, scale=4):
    sr = torch.clamp(sr, 0, 1)
    global ps_loss
    if ps_loss is None:
        ps_loss = perceptual_similarity.PerceptualSimilarityLoss(use_cuda=True)
    rgb_range = 1.
    sr_var, hr_var = sr.cuda(), hr.cuda()
    
    with torch.no_grad():
        ssim = pytorch_ssim.ssim(sr_var, hr_var).item()
        LPIPS_single = ps_loss((sr_var / rgb_range * 2 - 1),
                            (hr_var / rgb_range * 2 - 1)).item()
        ycbcr_psnr = Metrics.YCbCr_psnr(sr, hr, scale=scale)
    return [ycbcr_psnr, ssim, LPIPS_single]

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode='vanilla', target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

def to_ycbcr(tensor):
    gray_coeffs = [65.738, 129.057, 25.064]
    convert = tensor.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
    tensor = tensor.mul(convert).sum(dim=1)
    return tensor.unsqueeze(1)

def rescale(tensor, sf):
    return F.interpolate(tensor, scale_factor=sf, mode='bicubic')