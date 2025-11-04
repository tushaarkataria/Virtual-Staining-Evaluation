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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import Dataset
from fid_score_UNI2 import calculate_fid_given_paths_act, calculate_fid_given_paths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import piq

class VOCDatasetLoader(Dataset):
    def __init__(self,root,transform=None):
        self.root  = root
        self.transforms  = transform
        self.fileNames = glob.glob(os.path.join(root,'*.png'))

    def __getitem__(self, index):
        imageName = self.fileNames[index]
        image   = np.array(Image.open(imageName).convert("RGB"))
        mask    = Image.open(imageName.replace('fake_B','real_B')).convert('RGB')
        mask1   = np.array(mask)
        mask = mask1
        if self.transforms is not None:
            t_image = self.transforms(image= image, mask = mask)
            image = t_image["image"]
            mask  = t_image["mask"]
        return image, mask

    def __len__(self):  # return count of sample we have
        return len(self.fileNames)

@torch.no_grad()
def main():



    print("**************************************Pix2Pix***********************************")
    evaluationPaths = []
    evaluationPaths.append(os.path.join('Real IHC path'))
    evaluationPaths.append(os.path.join('Fake IHC path'))

    feature_length = 1536
    fid_value = calculate_fid_given_paths(evaluationPaths,10,device,feature_length,4)
    print("Pix2Pix-CDX2-FID Metric", fid_value)
    x_features, y_features = calculate_fid_given_paths_act(evaluationPaths,10,device,1536,4)

    kid: torch.Tensor = piq.KID()(x_features, y_features)
    print(f"Pix2Pix-CDX2 KID: {kid:0.4f}")
    pr: tuple = piq.PR()(x_features, y_features)
    print(f"Pix2Pix-CDX2 Improved Precision and Recall: {pr[0]:0.4f} {pr[1]:0.4f}")



if __name__ == '__main__':
    main()

