import argparse
import torch
import os

from solver import Solver
from tools.utils import mkdir
from constant import *
from modules.Up_model_cdc import Up_model_cdc
from modules.Circle_model import Circle_model_cdc
from dataset.Dataset import Dataset

from init_path.cdc_path import path

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=48, help='input patch size')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--scale', type=int, default=4, help='super resolution scale')
    parser.add_argument('--rgb_range', type=float, default=1)
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--test_per_epoch', type=int, default=10000, help='test interval')
    parser.add_argument('--test_interval', type=int, default=10e9)
    parser.add_argument('--save_interval', type=int, default=100)

    parser.add_argument('--module', type=str, default='up', help='[up, circle] the module name')
    parser.add_argument('--valid_num', type=int, default=VAL_NUM, help='validation number')
    parser.add_argument('--train_num', type=int, default=10000000, help='training upperbound')

    parser.add_argument('--trn_path', type=str, default=TRN_PATH)
    parser.add_argument('--tst_path', type=str, default=TST_PATH)
    parser.add_argument('--save_path', type=str, default='experiments_sony')
    parser.add_argument('--save_name', type=str, default='cdc_sony_kaiming_init')
    parser.add_argument('--filter_cameras', type=int, nargs='+', default=[], help='filter idx, -1 not filter')
    parser.add_argument('--cuda', type=int, default=5)
    parser.add_argument('--restore', type=int, default=0)
    
    parser.add_argument('--save_tst_out', type=str, default=None)
    parser.add_argument('--save_img', type=int, default=0)
    parser.add_argument('--save_fix', type=str, default="")
    parser.add_argument('--resume', type=str, default='best.pth')
    parser.add_argument('--pretrain', type=str, default='', help='pretrain model path')

    parser.add_argument('--test_block', type=int, default=256, help='test crop block size')
    parser.add_argument('--test_recon', type=int, default=0, help='test reconstruted LR for circle module')

    return parser.parse_args()


def train_solver(opt, train):
    if opt.module == 'up':
        module = Up_model_cdc(opt)
    else:
        module = Circle_model_cdc(opt)

    trn_dset = Dataset(opt, opt.trn_path, split='train')
    tst_dset = Dataset(opt, opt.tst_path, split='test')
    
    if not train:
        module.load_model(os.path.join(opt.save_path, opt.resume))
    if train and 'circle' in opt.module:
        module.load_up(path['sr_model'][CAMERA[opt.filter_cameras[0]]])
    if train and opt.pretrain:
        module.load_model(opt.pretrain)

    solver = Solver(opt, trn_dset, tst_dset, module)
    if opt.restore:
        break_info = module.load_model(os.path.join(opt.save_path, 'current1.pth'))
        solver.restore_break_info(break_info)

    return solver


if __name__ == "__main__":
    opt = get_opt()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda)
    opt.save_path = os.path.join(SAVE_ROOT, opt.save_path)
    mkdir(opt.save_path)
    train = bool(opt.train)
    print('TRAIN:', str(train))
    opt.save_path = os.path.join(opt.save_path, opt.save_name)
    print('save:', opt.save_path)

    solver = train_solver(opt, train)

    if not train:
        solver.test()
    else:
        solver.train()