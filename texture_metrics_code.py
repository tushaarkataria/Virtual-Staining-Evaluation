import os
import cv2 as cv
from PIL import Image
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from tqdm import tqdm
import argparse
import glob
import sys
import numpy as np
import torch
import time

def psnr_and_ssim_paths(result_path1,result_path2):
    PSNR = 0
    ssim = 0
    l2   = 0
    count = 0
    for i in tqdm(os.listdir(os.path.join(result_path1))):
        fake = cv.imread(os.path.join(result_path1,i))
        real = cv.imread(os.path.join(result_path2,i))
        l2   = l2  + np.sum((fake - real) ** 2)/(255*255)
        PSNR = PSNR + peak_signal_noise_ratio(fake, real)
        SSIM =0
        for channel in range(3):
            SSIM =SSIM+ structural_similarity(fake[:,:,channel], real[:,:,channel])
        SSIM = SSIM/3 
        ssim= ssim + SSIM
        count = count+1
    average_psnr=PSNR/count
    average_ssim=ssim/count
    average_l2 = l2/count
    return average_psnr, average_ssim, average_l2

def psnr_and_ssim(result_path):
    PSNR = 0
    ssim = 0
    l2   = 0
    count = 0
    for i in tqdm(os.listdir(os.path.join(result_path,'fake_B'))):
        fake = cv.imread(os.path.join(result_path,'fake_B',i))
        real = cv.imread(os.path.join(result_path,'real_B',i))
        l2   = l2  + np.sum((fake - real) ** 2)/(255*255)
        PSNR = PSNR + peak_signal_noise_ratio(fake, real)
        SSIM =0
        for channel in range(3):
            SSIM =SSIM+ structural_similarity(fake[:,:,channel], real[:,:,channel])
        SSIM = SSIM/3 
        ssim= ssim + SSIM
        count = count+1
    average_psnr=PSNR/count
    average_ssim=ssim/count
    average_l2 = l2/count
    return average_psnr, average_ssim, average_l2

@torch.no_grad()
def main():

    print("**************************************ACL_GAN***********************************")
    result_path = os.path.join('<Path Containing Both Real and Fake IHC(B)>')
    psnr_index, ssim_index, average_l2 =  psnr_and_ssim(result_path)
    print("PSNR, SSIM, L2 ",psnr_index, ssim_index, average_l2)


if __name__ == '__main__':
    main()

