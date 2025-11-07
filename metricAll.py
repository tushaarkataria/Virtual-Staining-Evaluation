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
from skimage.color import rgb2hed,hed2rgb,rgba2rgb,rgb2gray
import numpy as np
import torch
import cv2
import time
from skimage import io
from torch.utils.data import Dataset
import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label
from skimage.color import rgb2hed,hed2rgb,rgba2rgb,rgb2gray
from skimage import io
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.metrics import confusion_matrix

def psnr_and_ssim(result_path):
    PSNR = []
    ssim = []
    l2   = []
    count = 0
    file_list = os.listdir(os.path.join(result_path,'fake_B'))
    file_list= sorted(file_list, key=lambda x: int(os.path.splitext(x)[0]))
    count_positive = np.zeros(len(file_list))
    count
    for i in file_list:
        #print(i)
        fake = io.imread(os.path.join(result_path,'fake_B',i))
        real = io.imread(os.path.join(result_path,'real_B',i))
        real_HED = np.uint8(rgb2hed(real)[:,:,2]>0.07)
        kernel = np.ones((3,3),np.uint8)
        binary_img = cv2.dilate(real_HED,kernel,iterations =2)
        binary_img = cv2.erode(binary_img,kernel,iterations  =2)
        binary_img_real = cv2.medianBlur(binary_img,5)
        if(np.sum(binary_img_real)>0):
            count_positive[count]=1
        l2.append(np.sum((fake - real) ** 2)/(255*255))
        PSNR.append(peak_signal_noise_ratio(fake, real))
        SSIM =0
        for channel in range(3):
            SSIM =SSIM+ structural_similarity(fake[:,:,channel], real[:,:,channel])
        SSIM = SSIM/3
        ssim.append(SSIM)
        count = count+1
    average_psnr=np.mean(PSNR)
    average_ssim=np.mean(ssim)
    average_l2 = np.mean(l2)
    
    return average_psnr, average_ssim, average_l2, PSNR, ssim,l2, count_positive


def calculate_fpr(ground_truth_mask, predicted_mask):
    """
    Calculates the false positive rate (FPR) for image segmentation.

    Args:
        ground_truth_mask (numpy.ndarray): Binary ground truth mask.
        predicted_mask (numpy.ndarray): Binary predicted mask.

    Returns:
        float: The false positive rate.
    """
    # Ensure masks are boolean type
    ground_truth_mask = ground_truth_mask.astype(bool)
    predicted_mask = predicted_mask.astype(bool)

    # Calculate true negatives and false positives
    tn = np.sum(~ground_truth_mask & ~predicted_mask)
    fp = np.sum(~ground_truth_mask & predicted_mask)

    # Calculate FPR
    if (fp + tn) == 0:
      return 0
    fpr = fp / (fp + tn)
    return fpr


def calculate_fnr(ground_truth_mask, predicted_mask):
    """
    Calculates the false positive rate (FPR) for image segmentation.

    Args:
        ground_truth_mask (numpy.ndarray): Binary ground truth mask.
        predicted_mask (numpy.ndarray): Binary predicted mask.

    Returns:
        float: The false positive rate.
    """
    # Ensure masks are boolean type
    ground_truth_mask = ground_truth_mask.astype(bool)
    predicted_mask = predicted_mask.astype(bool)

    # Calculate true negatives and false positives
    tp = np.sum(ground_truth_mask & predicted_mask)
    fn = np.sum(ground_truth_mask & ~predicted_mask)

    # Calculate FPR
    if (tp + fn) == 0:
      return 0
    fnr = fn / (fn + tp)
    return fnr

def metrics_function(pix2pix_result_path):
    file_list = os.listdir(os.path.join(pix2pix_result_path,'fake_B'))
    file_list= sorted(file_list, key=lambda x: int(os.path.splitext(x)[0]))
    
    dice = np.zeros(len(file_list))
    jc   = np.zeros(len(file_list))
    hd   = np.zeros(len(file_list))
    asd  = np.zeros(len(file_list))
    true_positive_rate = np.zeros(len(file_list))
    true_negative_rate = np.zeros(len(file_list))
    false_positive_rate = np.zeros(len(file_list))
    count = 0
    count_positive = np.zeros(len(file_list))
    
    tn_array  = np.zeros(len(file_list))
    fp_array  = np.zeros(len(file_list))
    fn_array  = np.zeros(len(file_list))
    tp_array  = np.zeros(len(file_list))
    
    for i in file_list:
        fake = io.imread(os.path.join(pix2pix_result_path,'fake_B',i))
        real = io.imread(os.path.join(pix2pix_result_path,'real_B',i))
        real_HED = np.uint8(rgb2hed(real)[:,:,2]>0.07)
        fake_HED = np.uint8(rgb2hed(fake)[:,:,2]>0.07)
        kernel = np.ones((3,3),np.uint8)
        binary_img = cv2.dilate(fake_HED,kernel,iterations =2)
        binary_img = cv2.erode(binary_img,kernel,iterations  =2)
        binary_img_fake = cv2.medianBlur(binary_img,5)
        kernel = np.ones((3,3),np.uint8)
        binary_img = cv2.dilate(real_HED,kernel,iterations =2)
        binary_img = cv2.erode(binary_img,kernel,iterations  =2)
        binary_img_real = cv2.medianBlur(binary_img,5)
        
        true_positive_rate[count] = metric.binary.true_positive_rate(binary_img_fake, binary_img_real)
        true_negative_rate[count] = metric.binary.true_negative_rate(binary_img_fake, binary_img_real)
        false_positive_rate[count] = calculate_fpr(binary_img_real, binary_img_fake)
        #false_negative_rate[count] = calculate_fnr(binary_img_real, binary_img_fake)
        
        if(np.sum(binary_img_real)>0):
            dice[count] = metric.binary.dc(binary_img_fake, binary_img_real)
            jc[count] =(metric.binary.jc(binary_img_fake, binary_img_real))
            if(np.sum(binary_img_fake)>0):
                hd[count] = (metric.binary.hd95(binary_img_fake, binary_img_real))
                asd[count]= (metric.binary.asd(binary_img_fake, binary_img_real))
            count_positive[count] = 1
        count = count+1
    return dice, jc,hd,asd, true_positive_rate,true_negative_rate,count_positive


import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label
from skimage.color import rgb2hed,hed2rgb,rgba2rgb,rgb2gray
from skimage import io
import matplotlib.pyplot as plt
import os
import cv2
import monai
from sklearn.metrics import confusion_matrix
from monai.transforms import Compose,Activations,AsDiscrete

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def metrics_functionModelOutputs(pix2pix_result_path):
    file_list = os.listdir(os.path.join(pix2pix_result_path,'fake_B'))
    file_list= sorted(file_list, key=lambda x: int(os.path.splitext(x)[0]))
    
    dice = np.zeros(len(file_list))
    jc   = np.zeros(len(file_list))
    hd   = np.zeros(len(file_list))
    asd  = np.zeros(len(file_list))
    true_positive_rate = np.zeros(len(file_list))
    true_negative_rate = np.zeros(len(file_list))
    false_positive_rate = np.zeros(len(file_list))
    count = 0
    count_positive = np.zeros(len(file_list))
    
    tn_array  = np.zeros(len(file_list))
    fp_array  = np.zeros(len(file_list))
    fn_array  = np.zeros(len(file_list))
    tp_array  = np.zeros(len(file_list))

    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    saved_state_dict = torch.load('<Path to>/best_metric_model_ihc_segmentationCDX2.pth')
    model.load_state_dict(saved_state_dict, strict=True)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    model.eval()
    
    for i in file_list:
        fake = torch.tensor(io.imread(os.path.join(pix2pix_result_path,'fake_B',i)))
        real = torch.tensor(io.imread(os.path.join(pix2pix_result_path,'real_B',i)))
        fake = fake.permute(2,0,1)
        real = real.permute(2,0,1)
        fake = fake.unsqueeze(0)
        real = real.unsqueeze(0)
        
        fake =  fake.to(device,dtype=torch.float)
        real =  real.to(device,dtype=torch.float)
        
        #real_HED = np.uint8(rgb2hed(real)[:,:,2]>0.07)
        #fake_HED = np.uint8(rgb2hed(fake)[:,:,2]>0.07)

        real_HED = model(real)
        real_HED = post_trans(real_HED).cpu().numpy()
        real_HED = np.uint8(real_HED.squeeze())

        fake_HED = model(fake)
        fake_HED = post_trans(fake_HED).cpu().numpy()
        fake_HED = np.uint8(fake_HED.squeeze())
        
        kernel = np.ones((3,3),np.uint8)
        binary_img = cv2.dilate(fake_HED,kernel,iterations =2)
        binary_img = cv2.erode(binary_img,kernel,iterations  =2)
        binary_img_fake = cv2.medianBlur(binary_img,5)
        
        kernel = np.ones((3,3),np.uint8)
        binary_img = cv2.dilate(real_HED,kernel,iterations =2)
        binary_img = cv2.erode(binary_img,kernel,iterations  =2)
        binary_img_real = cv2.medianBlur(binary_img,5)
        
        true_positive_rate[count] = metric.binary.true_positive_rate(binary_img_fake, binary_img_real)
        true_negative_rate[count] = metric.binary.true_negative_rate(binary_img_fake, binary_img_real)
        false_positive_rate[count] = calculate_fpr(binary_img_real, binary_img_fake)
        #false_negative_rate[count] = calculate_fnr(binary_img_real, binary_img_fake)
        #print(np.sum(binary_img_real))
        if(np.sum(binary_img_real)>550):
            dice[count] = metric.binary.dc(binary_img_fake, binary_img_real)
            jc[count] =(metric.binary.jc(binary_img_fake, binary_img_real))
            if(np.sum(binary_img_fake)>0):
                hd[count] = (metric.binary.hd95(binary_img_fake, binary_img_real))
                asd[count]= (metric.binary.asd(binary_img_fake, binary_img_real))
            count_positive[count] = 1
        count = count+1
    return dice, jc,hd,asd, true_positive_rate,true_negative_rate,count_positive


pix2pix_result_path = "Path to Folder where both REAL and FAKE IHC are present as real_B and fake_B"
average_psnr, average_ssim, average_l2, PSNR, ssim,l2, count_positive = psnr_and_ssim(pix2pix_result_path)
print("Pix2Pix",average_psnr, average_ssim, average_l2)


## DAB segmentation Metrics
pix2pix_result_path = "Path to Folder where both REAL and FAKE IHC are present as real_B and fake_B"
dice, jc,hd,asd, true_positive_rate,true_negative_rate,count_positive =  metrics_function(pix2pix_result_path)
print("Pix2Pix",np.mean(dice),np.mean(jc),np.mean(hd),np.mean(true_positive_rate),np.mean(true_negative_rate))



## Model Metrics
pix2pix_result_path = "Path to Folder where both REAL and FAKE IHC are present as real_B and fake_B"
dice_model, jc_model,hd_model,asd_model, true_positive_rate_model,true_negative_rate_model,count_positive_model =  metrics_functionModelOutputs(pix2pix_result_path)
print("Pix2Pix_model",np.mean(dice_model),np.mean(jc_model),np.mean(hd_model),np.mean(true_positive_rate_model),np.mean(true_negative_rate_model))

