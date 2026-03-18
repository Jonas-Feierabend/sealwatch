"""Implementation of UcNet architecture, introduced in

K. Wei, W. Luo, S. Tan, J. Huang.
Universal deep network for steganalysis of color image based on channel representation.
IEEE TIFS, 2022.

Inspired by the authors' implementation
https://github.com/revere7/UCNet_Steganalysis/

"""

import cv2
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

from .srm_kernel_filters import all_normalized_hpf_list


def build_filters():
    """Builds the rich model/Gabor filters.

    Taken from the authors' implementation.
    https://github.com/revere7/UCNet_Steganalysis/
    """
    filters = []
    ksize = [5]
    lamda = np.pi / 2.0
    sigma = [0.5, 1.0]
    phi = [0, np.pi / 2]
    for hpf_item in all_normalized_hpf_list:
        row_1 = int((5 - hpf_item.shape[0]) / 2)
        row_2 = int((5 - hpf_item.shape[0]) - row_1)
        col_1 = int((5 - hpf_item.shape[1]) / 2)
        col_2 = int((5 - hpf_item.shape[1]) - col_1)
        hpf_item = np.pad(hpf_item, pad_width=((row_1, row_2), (col_1, col_2)), mode='constant')
        filters.append(hpf_item)
    for theta in np.arange(0, np.pi, np.pi / 8):  # gabor 0 22.5 45 67.5 90 112.5 135 157.5
        for k in range(2):
            for j in range(2):
                kern = cv2.getGaborKernel(
                    (ksize[0], ksize[0]),
                    sigma[k],
                    theta,
                    sigma[k] / .56,
                    .5,
                    phi[j],
                    ktype=cv2.CV_32F)
                filters.append(kern)
    return np.array(filters)


class Block2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.basic = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),  # kernel size not in the paper
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=2, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = F.relu(self.basic(x) + self.shortcut(x))
        return x


class Block3(nn.Module):
    def __init__(self, in_channels: int = 32, out_channels: int = 64):
        super().__init__()
        self.basic = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, stride=2, groups=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = F.relu(self.basic(x) + self.shortcut(x))
        return x



class UcNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 2):
        super(UcNet, self).__init__()

        # === preprocessing module ===
        self.preprocessing = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 62, groups=in_channels, kernel_size=5, padding=2, bias=False),
            nn.Hardtanh(min_val=-2, max_val=2))  # TLU
        filters = torch.Tensor(build_filters()).view(62, 1, 5, 5).repeat(in_channels, 1, 1, 1)
        with torch.no_grad():
            self.preprocessing[0].weight.copy_(filters)  # hard-set filters
        self.preprocessing[0].weight.requires_grad = False  # non-trainable

        # === convolutional module ===
        self.block_1a = nn.Sequential(
            nn.Conv2d(in_channels*62, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.block_2a = Block2(32, 32)
        self.block_3 = Block3(32, 64)
        self.block_2b = Block2(64, 128)
        self.block_1b = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # === classification module ===
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.fc = nn.Linear(256, out_channels)

    def forward(self, x):
        # preprecessing module
        x = self.preprocessing(x)
        # convolutional module
        x = self.block_1a(x)
        x = self.block_2a(x)
        x = self.block_3(x)
        x = self.block_2b(x)
        x = self.block_1b(x)
        # classification module
        x = self.gap(x)
        y = self.fc(x)
        return y

    @classmethod
    def pretrained(cls, in_channels: int = 3, *args, **kw):

        # create model
        model = cls(in_channels=in_channels, *args, **kw)

        # download weights
        guangzhou_checkpoint_url = 'https://github.com/revere7/UCNet_Steganalysis/raw/refs/heads/main/J-UNIWARD-pretrain-parameters.pt'
        state_dict_guangzhou = torch.hub.load_state_dict_from_url(guangzhou_checkpoint_url, map_location='cpu')['original_state']
        state_dict_guangzhou = {k.split('module.')[1]: v for k, v in state_dict_guangzhou.items()}

        # modify input
        state_dict = {}
        for k, v in state_dict_guangzhou.items():
            if k.startswith('fc1'):
                state_dict[k.replace('fc1', 'fc')] = v
            elif k.startswith('group1'):
                state_dict[k.replace('group1.hpf', 'preprocessing.0')] = v.repeat(in_channels, 1, 1, 1)
            elif k.startswith('group2.0'):
                state_dict[k.replace('group2.0', 'block_1a.0')] = v
            elif k.startswith('group2.1'):
                state_dict[k.replace('group2.1', 'block_1a.1')] = v
            elif k.startswith('group2.3'):
                state_dict[k.replace('group2.3', 'block_1a.3')] = v
            elif k.startswith('group2.4'):
                state_dict[k.replace('group2.4', 'block_1a.4')] = v
            elif k.startswith('group2.6'):
                state_dict[k.replace('group2.6', 'block_2a')] = v
            elif k.startswith('group3'):
                state_dict[k.replace('group3', 'block_3')] = v
            elif k.startswith('group4'):
                state_dict[k.replace('group4', 'block_2b')] = v
            elif k.startswith('group5'):
                state_dict[k.replace('group5', 'block_1b')] = v
            else:
                print(k)

        #
        model.load_state_dict(state_dict, strict=True)
        logging.info(f'using pretrained weights from {guangzhou_checkpoint_url}')
        return model


def pretrained(*args, **kw) -> UcNet:
    return UcNet.pretrained(*args, **kw)


def infere_single(
    x: np.ndarray,
    model: Callable = None,
    *,
    device: torch.nn.Module = torch.device('cpu'),
) -> np.ndarray:
    """Runs inference for a single image.

    :param x: image
    :type x:
    :param model:
    :type model:
    :param device:
    :type device:
    :return:
    :rtype:
    """
    # prepare data
    x = (x / 255.).transpose(2, 0, 1)[None]
    x_ = torch.from_numpy(x).to(dtype=torch.float32, device=device)
    # get model
    model = model.to(dtype=torch.float32, device=device)

    #
    with torch.no_grad():
        # infere
        logit = model(x_)
        y_ = torch.nn.functional.softmax(logit, dim=1)[0, 1]
        # convert back to numpy
        y = y_.detach().cpu().numpy()
    #
    return y
