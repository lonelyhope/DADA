import random
import numpy as np
from PIL import Image
from constant import *
import torch
import math

def camera_lb(img_name):
    for i, camera in enumerate(CAMERA):
        if camera in img_name:
            return i

def filter_camera(idx, lists):
    # lists 为 camera list 列表，通常为 [lr_list, hr_list]
    filtered_lists = [[] for _ in range(len(lists))]
    for i, file in enumerate(lists[0]):
        if CAMERA[idx] in file:
            for j in range(len(lists)):
                filtered_lists[j].append(lists[j][i])
    return filtered_lists

def shuffle(lists):
    # 打乱 lists 中每个 list 的顺序
    # lists 为 camera list 列表，通常为 [lr_list, hr_list]
    random.seed(SEED)
    num_lst = list(range(len(lists[0])))
    random.shuffle(num_lst)
    return [ list(np.array(lst)[num_lst]) for lst in lists ]

def random_pre_process_pair(hr, lr, lr_patch_size, scale, memory=None, use_memory=False):
    """
    For Real Paired Images,
    Crop hr, lr images to patches, and random pre-processing
    :param hr, lr: PIL.Image
    :param lr_patch_size: lr patches size
    :param scale: upsample scale
    :return: PIL.Image
    """
    w, h = lr.size
    if not use_memory:
        startx = random.randint(0, w - lr_patch_size)
        starty = random.randint(0, h - lr_patch_size)
        left_right_flip = bool(random.getrandbits(1))
        top_bottom_flip = bool(random.getrandbits(1))
        angle = random.randint(0, 3) * 90
    else:
        startx = memory['startx']
        starty = memory['starty']
        left_right_flip = memory['left_right_flip']
        top_bottom_flip = memory['top_bottom_flip']
        angle = memory['angle']

    hr_patch = hr.crop((startx * scale, starty * scale,
                        (startx + lr_patch_size) * scale, (starty + lr_patch_size) * scale))
    lr_patch = lr.crop((startx, starty, startx + lr_patch_size, starty + lr_patch_size))

    if left_right_flip:
        hr_patch = hr_patch.transpose(Image.FLIP_LEFT_RIGHT)
        lr_patch = lr_patch.transpose(Image.FLIP_LEFT_RIGHT)
    if top_bottom_flip:
        hr_patch = hr_patch.transpose(Image.FLIP_TOP_BOTTOM)
        lr_patch = lr_patch.transpose(Image.FLIP_TOP_BOTTOM)

    if memory is not None:
        memory['startx'] = startx
        memory['starty'] = starty
        memory['left_right_flip'] = left_right_flip
        memory['top_bottom_flip'] = top_bottom_flip
        memory['angle'] = angle

    return hr_patch.rotate(angle), lr_patch.rotate(angle)

def get_half(lst, idx=0):
    l = len(lst)
    hl = l // 2
    if idx == 1:
        return lst[:hl]
    elif idx == 2:
        return lst[hl:]
    else:
        return lst


# --------------------------------
# imresize for numpy image
# --------------------------------
def cubic(x):
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5*absx3 - 2.5*absx2 + 1) * ((absx <= 1).type_as(absx)) + \
        (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * (((absx > 1)*(absx <= 2)).type_as(absx))
        
def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)

def imresize_np(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC or HW [0,1]
    # output: HWC or HW [0,1] w/o round
    img = torch.from_numpy(img)
    need_squeeze = True if img.dim() == 2 else False
    if need_squeeze:
        img.unsqueeze_(2)

    in_H, in_W, in_C = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for j in range(out_C):
            out_1[i, :, j] = img_aug[idx:idx + kernel_width, :, j].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        for j in range(out_C):
            out_2[:, i, j] = out_1_aug[:, idx:idx + kernel_width, j].mv(weights_W[i])
    if need_squeeze:
        out_2.squeeze_()

    return out_2.numpy()
