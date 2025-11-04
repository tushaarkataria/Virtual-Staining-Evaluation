#import openslide
#import slideio
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage import feature
import cv2
from skimage import morphology as morph
import pandas as pd
import os
from monai.data import WSIReader
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.color import rgb2hed,hed2rgb
import cv2
from skimage import data, filters, measure, morphology, exposure
from scipy.ndimage import label as scilabel
from skimage import io
import skimage as ski
from skimage import morphology
from scipy import ndimage
from sklearn.model_selection import train_test_split
import ants
from skimage.color import rgb2gray
from skimage import filters
import cv2
import h5py
from skimage.filters import threshold_otsu
import pandas as pd
from skimage.color import rgb2hed, hed2rgb
import argparse
import glob
import geojson
from register_images import registerImages
from skimage.draw import polygon
from skimage.draw import polygon2mask


def GroundTruthCreationLymphoCDX2(TissueSample):                                                                                                                              
    TissueSample2 = rgb2hed(TissueSample)                                                                                                                                    
    aniso         = TissueSample2[:,:,2]                                                                                                                                     
    binary_img = np.uint8(aniso > aniso.mean())                                                                                                            
    kernel = np.ones((3,3),np.uint8)                                                                                                                                         
    binary_img = cv2.erode(binary_img,kernel,iterations = 5)                                                                                                                 
    binary_img = cv2.dilate(binary_img,kernel,iterations =20)                                                                                                                 
    binary_img = cv2.medianBlur(binary_img, 51)                                                                                                                              
    binary_img = morphology.area_opening(binary_img, area_threshold=2000, connectivity=2)                                                                                     
    return binary_img  

def fullGlandSegmentation(binary_img):
    contours1, hierarchy1 = cv2.findContours(np.uint8(binary_img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    M,N = binary_img.shape
    image_masked = np.zeros((M,N),dtype=np.uint8)
    for i in range(len(contours1)):
        poly_coordinates =(np.asarray(np.uint32(contours1[i])))
        poly_coordinates = np.asarray(poly_coordinates)
        r,c = poly_coordinates.T
        r = np.squeeze(r)
        c = np.squeeze(c)

        rr, cc = polygon(c,r,image_masked.shape)
        image_masked[rr, cc] =1
    return image_masked


def preprocessingPipeLine(args,folderName):
    print(folderName)
    path = args.path
    outputDir = os.path.join(args.outputDirectory,folderName)

    if(args.tissue=='CDX2'):
         he_fName  = glob.glob('<Path of Files>'+folderName+'/*6HE*')[0]
         ihc_fName = he_fName.replace('HE','CDX2')
         subfolderName = he_fName.split('/')[-1].split('HE')[0]
         outputDirHE  = os.path.join(outputDir,subfolderName,'HE')
         outputDirIHC = os.path.join(outputDir,subfolderName,'CDX2')

    if not os.path.exists(outputDirHE):
        os.makedirs(outputDirHE)
    if not os.path.exists(outputDirIHC):
        os.makedirs(outputDirIHC)
    
    image_reader = WSIReader()

    wsi = image_reader.read(os.path.join(path,ihc_fName))
    he_wsi = image_reader.read(os.path.join(path,he_fName))
   
    print("Loading Downsampled Images")
    # Downsampled
    he_img_data_down, he_meta_data = image_reader.get_data(he_wsi,level=2)
    he_img_data_down = np.moveaxis(he_img_data_down, 0, 2)
    
    img_data_down, he_meta_data = image_reader.get_data(wsi,level=2)
    img_data_down = np.moveaxis(img_data_down, 0, 2)
    
    print("Loading Full Resolution Images")
    # Not Downsampled    
    he_img_data, he_meta_data = image_reader.get_data(he_wsi)
    he_img_data = np.moveaxis(he_img_data, 0, 2)
    
    img_data, he_meta_data = image_reader.get_data(wsi)
    img_data = np.moveaxis(img_data, 0, 2)


    gray_he_img_data_down = rgb2gray(he_img_data_down)
    gray_img_data_down = rgb2gray(img_data_down)
        
    thresh = threshold_otsu(gray_he_img_data_down)
    Fixed1 = ants.from_numpy(np.uint8(gray_he_img_data_down < thresh))
    
    thresh = threshold_otsu(gray_img_data_down)
    Moving1 = ants.from_numpy(np.uint8(gray_img_data_down < thresh))
    
    print('Registration Running') 
    mytx = ants.registration(fixed=Fixed1, moving=Moving1, type_of_transform = 'SyNRA',outprefix='CD20_NEWTRANSFORM')

    thresh = threshold_otsu(gray_he_img_data_down)
    binary = np.uint8(gray_he_img_data_down < thresh)
    kernel = np.ones((3,3),np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=6)
    binary = cv2.medianBlur(binary, 51)
    cnts = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    rectList=[]
    area = []
    Image1 = he_img_data_down.copy()
    Image2 = img_data_down.copy()
    count=0
    for c in cnts:
        x1,y1,w1,h1 = cv2.boundingRect(c)
        rectList.append([x1,y1,w1,h1])
        if(w1*h1>50000):
            cv2.rectangle(Image1, (x1, y1), (x1 + w1, y1 + h1),(255, 0, 0),5)
            d = {'x': [y1, y1+h1], 'y': [x1, x1+w1]}
            pts = pd.DataFrame(data=d)
           
            wpts = ants.apply_transforms_to_points(2,pts,mytx['fwdtransforms'])
            x = np.int32(wpts['x'])
            y = np.int32(wpts['y'])
        
            if(x[0]<0):
                x[0]=0
            if(y[0]<0):
                y[0]=0
            cv2.rectangle(Image2, (y[0], x[0]), (y[0] + w1, x[0] + h1),(255, 0, 0),5)

            sub_he_img_data = he_img_data[16*y1:16*(y1+h1),16*x1:16*(x1+w1)]
            #manual_mask_sub_he_img_data = manualmask[16*y1:16*(y1+h1),16*x1:16*(x1+w1)]
            hf = h5py.File(os.path.join(outputDirHE,subfolderName+'_ROI_'+str(count)+'.h5'), 'w')
            hf.create_dataset('image',data=sub_he_img_data)
            #if(np.sum(manual_mask_sub_he_img_data)!=0):
            #    hf.create_dataset('GRmask',data=manual_mask_sub_he_img_data)
            hf.close()
            print("Writing File Number",count) 
            sub_img_data    = img_data[16*x[0]:16*(x[0]+h1),16*y[0]:16*(y[0]+w1)]
            if(args.tissue=='CDX2'):
                binary_img = np.uint8(GroundTruthCreationLymphoCDX2(sub_img_data))
                FullGland  = fullGlandSegmentation(binary_img)
            hf = h5py.File(os.path.join(outputDirIHC,subfolderName+'_ROI_'+str(count)+'.h5'), 'w')
            hf.create_dataset('image',data=sub_img_data)
            hf.create_dataset('glandSeg',data=binary_img)
            hf.create_dataset('glandSegFull',data=FullGland)
            hf.close()
            temp_image = sub_img_data.copy()
            contours1, hierarchy1 = cv2.findContours(FullGland, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(temp_image,contours1, -1,(0,0,0),2,10)
            io.imsave(subfolderName+str(count)+'.png',temp_image)
            count=count+1
    io.imsave(subfolderName+'HE_CDX2.png',Image1)
    io.imsave(subfolderName+'IHC_CDX2.png',Image2)
    directoryName2Registered = os.path.join(outputDir,subfolderName,'CDX2Reg') 
    if not os.path.exists(directoryName2Registered):
        os.makedirs(directoryName2Registered)
    print("Registering Directories")
    registerImages(outputDirHE,outputDirIHC,directoryName2Registered)
    

def main(args):
    Data = pd.read_csv("folderName.csv") # List of all file Name of HE images.
    folderNames = list(Data['FolderNames'])
    for i in range(len(folderNames)):
        folderName = folderNames[i]
        preprocessingPipeLine(args,folderName)

 
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('-dataResolution'   ,type=float  , action="store", dest='dataResolution' , default=1    )
    parser.add_argument('-dataPath'         ,type=str    , action="store", dest='path'           , default='<Path to Image Files>')
    parser.add_argument('-outputPath'       ,type=str    , action="store", dest='outputDirectory', default='<Output Directory File>')
    parser.add_argument('-tissue'           ,type=str    , action="store", dest='tissue', default='CDX2')

    args = parser.parse_args()
    main(args)         



