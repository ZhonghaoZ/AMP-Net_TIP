import torch
import random
import math
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from scipy import io
import os
from torch.autograd import Variable
import glob
import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):

    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    return np.mean(np.mean(ssim_map))


def imread_CS_py(imgName):
    block_size = 33
    Iorg = np.array(Image.open(imgName), dtype='float32')  # 读图
    [row, col] = Iorg.shape  # 图像的 形状
    row_pad = block_size-np.mod(row,block_size)  # 求余数操作
    col_pad = block_size-np.mod(col,block_size)  # 求余数操作，用于判断需要补零的数量
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]


def img2col_py(Ipad, block_size):
    [row, col] = Ipad.shape  # 当前图像的形状
    row_block = row/block_size
    col_block = col/block_size
    block_num = int(row_block*col_block)  # 一共有多少个 模块
    img_col = np.zeros([block_size**2, block_num])  # 把每一块放进每一列中， 这就是容器
    count = 0
    for x in range(0, row-block_size+1, block_size):
        for y in range(0, col-block_size+1, block_size):
            img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].reshape([-1])
            # img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].transpose().reshape([-1])
            count = count + 1
    return img_col


def col2im_CS_py(X_col, row, col, row_new, col_new):
    block_size = 33
    X0_rec = np.zeros([row_new, col_new])
    count = 0
    for x in range(0, row_new-block_size+1, block_size):
        for y in range(0, col_new-block_size+1, block_size):
            X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size])
            # X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size]).transpose()
            count = count + 1
    X_rec = X0_rec[:row, :col]
    return X_rec


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def symetrize(img, N):
    img_pad = np.pad(img, ((N, N), (N, N)), 'symmetric')
    return img_pad


def add_gaussian_noise(im, sigma, seed=None):
    if seed is not None:
        np.random.seed(seed)
    im = im + (sigma * np.random.randn(*im.shape)).astype(np.int)
    im = np.clip(im, 0., 255., out=None)
    im = im.astype(np.uint8)
    return im

def ind_initialize(max_size, N, step):
    ind = range(N, max_size - N, step)
    if ind[-1] < max_size - N - 1:
        ind = np.append(ind, np.array([max_size - N - 1]), axis=0)
    return ind


def get_kaiserWindow(kHW):
    k = np.kaiser(kHW, 2)
    k_2d = k[:, np.newaxis] @ k[np.newaxis, :]
    return k_2d


def get_coef(kHW):
    coef_norm = np.zeros(kHW * kHW)
    coef_norm_inv = np.zeros(kHW * kHW)
    coef = 0.5 / ((float)(kHW))
    for i in range(kHW):
        for j in range(kHW):
            if i == 0 and j == 0:
                coef_norm[i * kHW + j] = 0.5 * coef
                coef_norm_inv[i * kHW + j] = 2.0
            elif i * j == 0:
                coef_norm[i * kHW + j] = 0.7071067811865475 * coef
                coef_norm_inv[i * kHW + j] = 1.414213562373095
            else:
                coef_norm[i * kHW + j] = 1.0 * coef
                coef_norm_inv[i * kHW + j] = 1.0

    return coef_norm, coef_norm_inv


def sd_weighting(group_3D):
    N = group_3D.size

    mean = np.sum(group_3D)
    std = np.sum(group_3D * group_3D)

    res = (std - mean * mean / N) / (N - 1)
    weight = 1.0 / np.sqrt(res) if res > 0. else 0.
    return weight

