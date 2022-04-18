import argparse
from .models import dist_model as dm
# from models import dist_model as dm
# from .util import util
from torch import nn
from PIL import Image
# from TorchTools.DataTools.Loaders import to_tensor
from tools.functional import to_tensor
import cv2

class PerceptualSimilarityLoss(nn.Module):
    def __init__(self, model='net-lin', net='alex', use_cuda=True):
        super(PerceptualSimilarityLoss, self).__init__()
        self.model = dm.DistModel()
        self.model.initialize(model=model, net=net, use_gpu=use_cuda)

    def forward(self, img0, img1):
        return self.model.forward(img0, img1)

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    loss = PerceptualSimilarityLoss()

    fake_hr = '/media/data4/xxq/SR/exp/result2/experiments_DASR/SRN/results/SRN_Adapt_P2S(with_sr_pretrian)/DRealSR_2500/imgs/panasonic_103_x1.png'
    real_hr = '/media/data4/xxq/SR/dataset/DRealSR/x4/Test_x4/test_HR/panasonic_103_x4.png'

    # hr_down = to_tensor(Image.open('./imgs/test1/DSC_1454_x4.png')) * 2 - 1.
    # lr = to_tensor(Image.open('./imgs/test1/DSC_1454_x1.png')) * 2 - 1

    hr_down = to_tensor(Image.open(fake_hr)) * 2 - 1.
    lr = to_tensor(Image.open(real_hr)) * 2 - 1
    print(loss(hr_down.unsqueeze(0), lr.unsqueeze(0)))
