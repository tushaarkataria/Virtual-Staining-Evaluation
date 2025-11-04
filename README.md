# Virtual-Staining-Evaluation


- ```texture_metrics_code.py``` :  Code for PSNR, SSIM and MSE Calcuation
- ```distributionMetrics.py```  : Code for Distribution Metrics, FID, KID, Precision and Recall.
- ```distributionMetrics-UNI2.py``` : Code Containing Distribution Metrics, FID, KID, Precision and Recall using UNI-2 Encoder. Might require uni-2.h in the same folder. 
- ```pytorch-fid/, fid_score_UNI2.py, fid_score1.py, fid_score_UNI.py ``` : Supporting Files for FID calculation.
- ```metricAll.py```: Calculated patch wise metrics, PSNR, SSIM, MSE and all segmentation metrics. Which can be later stored in a pandas dataframe. 
- ```preprocessing_WSI/```: Contains the preprocessing code H&E-IHC registration and Automated Annotation of Glands on H&E using CDX2 thresholding. 

