import torch
import torch.nn as nn
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        return x + residual

class De_resnet(nn.Module):
    def __init__(self, n_res_blocks=8, scale=4):
        super(De_resnet, self).__init__()
        self.block_input = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(64) for _ in range(n_res_blocks)])
        self.scale = scale
        if self.scale == 4:
            self.down_sample = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.PReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.PReLU()
            )
        elif self.scale == 2:
            self.down_sample = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.PReLU(),
            )
        self.block_output = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        block = self.block_input(x)
        for res_block in self.res_blocks:
            block = res_block(block)
        if self.down_sample:
            block = self.down_sample(block)
        block = self.block_output(block)
        return torch.sigmoid(block)

class Down_net(nn.Module):
    def __init__(self, opt, scale=4):
        super().__init__()
        negval = 0.2
        nFeat = 16
        n_colors = 3

        in_channels = n_colors
        out_channels = n_colors

        down_block = [
            nn.Sequential(
                nn.Conv2d(in_channels, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=negval, inplace=True)
            )
        ]

        for _ in range(1, int(np.log2(scale))):
            down_block.append(
                nn.Sequential(
                    nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=negval, inplace=True)
                )
            )

        down_block.append(nn.Conv2d(nFeat, out_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.down_module = nn.Sequential(*down_block)

    def forward(self, x):
        x = self.down_module(x)
        return x
