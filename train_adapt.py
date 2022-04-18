import argparse
from ast import parse
from PIL.Image import SAVE
import os
import shutil

from constant import *
from solver import Solver
from tools.utils import mkdir
from modules.Adaptation import Adaptation
from dataset.Adapt_dataset import Adapt_dataset
from init_path.cdc_path import path

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=48, help='input patch size')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--scale', type=int, default=4, help='super resolution scale')
    parser.add_argument('--rgb_range', type=float, default=1)
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--best_epochs', type=int, default=100, help='save best for every best_epochs')
    parser.add_argument('--test_per_epoch', type=int, default=1000, help='test interval')
    parser.add_argument('--test_interval', type=int, default=10e10)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--save_iter', type=int, default=10e9)
    
    parser.add_argument('--s_camera', type=int, default=1, help='source camera')
    parser.add_argument('--t_camera', type=int, default=2, help='target camera')

    parser.add_argument('--t_len', type=int, default=0, help='target camera img number')
    parser.add_argument('--valid_num', type=int, default=40, help='validation number for target camera')

    parser.add_argument('--trn_path', type=str, default=TRN_PATH)
    parser.add_argument('--tst_path', type=str, default=TST_PATH)
    parser.add_argument('--save_path', type=str, default='experiments_adapt')
    parser.add_argument('--save_name', type=str, default='CDC_adapt_no_supv') 
    parser.add_argument('--cuda', type=int, default=4)
    parser.add_argument('--restore', type=int, default=0)
    
    parser.add_argument('--save_tst_out', type=str, default="sony")
    parser.add_argument('--save_img', type=int, default=0)
    parser.add_argument('--save_numpy', type=int, default=0)
    parser.add_argument('--save_fix', type=str, default="")
    parser.add_argument('--resume', type=str, default='best.pth')

    parser.add_argument('--write_dir', type=str, default='/data3/xxq/SR/result/mask')
    parser.add_argument('--filter_cameras', type=int, nargs='+', default=[], help='filter idx, -1 not filter')
    
    return parser.parse_args()


def train_adapt(opt, train=True):
    # get module
    module = Adaptation(opt)
    module.say_name()

    # get dataset and init parameters
    trn_dset = Adapt_dataset(opt, opt.trn_path, split='train')
    init_path = path[IP]['circle_model'][CAMERA[opt.s_camera]]
    if train:
        val_dset = Adapt_dataset(opt, opt.tst_path, split='test')
        module.init_all(init_path)
    else:
        val_dset = Adapt_dataset(opt, opt.tst_path, split='test')
        module.load_model(os.path.join(opt.save_path, opt.resume), init_path)

    solver = Solver(opt, trn_dset, val_dset, module)
    if opt.restore:
        break_info = module.load_model(os.path.join(opt.save_path, 'current0.pth'), init_path)
        solver.restore_break_info(break_info)

    return solver, module


if __name__ == "__main__":
    print(SAVE_ROOT)
    opt = get_opt()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda)
    opt.save_path = os.path.join(SAVE_ROOT, opt.save_path)
    mkdir(opt.save_path)
    train = bool(opt.train)
    print('TRAIN:', str(train))
    opt.save_path = os.path.join(opt.save_path, opt.save_name)
    print('save:', opt.save_path)

    solver, module = train_adapt(opt, train)

    if not train:
        solver.test()
    else:
        solver.train()