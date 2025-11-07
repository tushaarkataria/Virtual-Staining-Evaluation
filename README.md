# Virtual-Staining-Evaluation


- ```texture_metrics_code.py``` :  Code for PSNR, SSIM and MSE Calcuation.
- ```distributionMetrics.py```  : Code for Distribution Metrics, FID, KID, Precision and Recall.
- ```distributionMetrics-UNI2.py``` : Code Containing Distribution Metrics, FID, KID, Precision and Recall using UNI-2 Encoder. Might require uni-2.h in the same folder. 
- ```pytorch-fid/, fid_score_UNI2.py, fid_score1.py, fid_score_UNI.py ``` : Supporting Files for FID calculation.
- ```metricAll.py```: Calculated patch wise metrics, PSNR, SSIM, MSE and all segmentation metrics. Which can be later stored in a pandas dataframe. 
- ```preprocessing_WSI/```: Contains the preprocessing code H&E-IHC registration and Automated Annotation of Glands on H&E using CDX2 thresholding.

## Python Libraries
The libraries required to run the above code are listed in the ```requirement.txt``` file. The exact versions of these libraries are not critical, as the code only relies on standard modules whose functionality remains consistent across versions.

## How to Use

Most of the evaluation metrics codes assume that the real IHC images are stored in a folder named ```real_B```, and generated IHC by the model are stored in the same folder with a name ```fake_B```. Although that is not a hard constraint, as we have provided functions that work on two different paths as well.

In our evaluation and coding set-up the image files were named as numbers such as '0.png', '1.png', '2.png' and so on. So some of the code might assume that, rather than looking for named images. Minor code modification might be required adapt the code to run different types of naming conventions. Because of the scale of our study, we kept it uniform to make the comparisons easier.  

- ```texture_metrics_code.py``` : To run this code, you need to provide the path to the ```real_B``` and ```fake_B``` images in line 62. ```psnr_and_ssim_paths``` is the adaptation to different paths.

- ```distributionMetrics.py```  : To run this code, you again need to provide the paths of ```real_B``` and ```fake_B``` and add them to evaluation paths list at line 52 and 53. The feature length is by default set for Inception-net, with batch size fixed.

- ```distributionMetrics-UNI2.py```  : Works the same as ```distributionMetrics.py```, just the encoder used for latent feature representation is taken from UNI-2 encoder. The code requires that ```uni-h2.bin``` be present in the same folder as this file. The binary file can be downloaded from https://huggingface.co/MahmoodLab/UNI2-h.

- ```metricAll.py``` : Works similar to above mentioned code files where path to ```real_B``` and ```fake_B``` images need to be added in line 271, 276 and 284 for a single model evaluations. This code evaluates all patch-wise metrics for all images. This code requires access to a CDX2 segmentation model "best_metric_model_ihc_segmentationCDX2.pth" in line 218. Which can be requested from the corresponding author. All the other code just has installation dependencies.

- ```preprocessing_WSI/preprocessingPipelineNew-CDX2.py```: PreProcessing code for Tissue Sampling. This code required path to WSI images in line 195 '.svs/.tiff files'. And a path to outputdirectory, where tissue can be saved. The tissue are saved in .h5 files.


