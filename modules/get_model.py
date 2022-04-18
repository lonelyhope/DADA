from architectures.de_resnet import De_resnet

from .model_tools import *

def get_up(opt):
    from architectures.cdc_model import HourGlassNetMultiScaleInt
    return HourGlassNetMultiScaleInt(in_nc=3, out_nc=3, upscale=opt.scale, nf=64,
                        res_type='res', n_mid=2, n_HG=6, inter_supervis=True,
                        mscale_inter_super=False)

def get_down(opt):
    return De_resnet(n_res_blocks=8, scale=opt.scale)

def get_D(opt, in_dim=3):
    from architectures.NLayerDiscriminator import NLayerDiscriminator
    
    norm='instance'
    init_type='normal'
    init_gain=0.02
    
    norm_layer = get_norm_layer(norm_type=norm)
    net = NLayerDiscriminator(in_dim, 64, n_layers=3, norm_layer=norm_layer)
    init_weights(net, init_type, init_gain=init_gain)
    return net

