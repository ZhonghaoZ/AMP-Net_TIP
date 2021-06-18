# AMP-Net: Denoising-based Deep Unfolding for Compressive Image Sensing
This repository provides a pytorch-based implementation of the model proposed by the paper [**AMP-Net: Denoising based Deep Unfolding for Compressive Image Sensing**](https://ieeexplore.ieee.org/document/9298950) which is published in **IEEE Transactions on Image Processing**.

If you use this code, please kindly cite 
```
@article{zhang2021amp,
	author={Zhang, Zhonghao and Liu, Yipeng and Liu, Jiani and Wen, Fei and Zhu, Ce},
	journal={IEEE Transactions on Image Processing}, 
	title={AMP-Net: Denoising-Based Deep Unfolding for Compressive Image Sensing}, 
	year={2021},
	volume={30},
	pages={1487-1500},
	doi={10.1109/TIP.2020.3044472}}
```

If you have any question about the code or the paper, you can email zhonghaozhang@yeah.net.

## Prerequisites
* Python 3.6~3.7 (We did not test other versions)
* Pytorch 1.2~1.7 with NVIDIA GPU or CPU (We did not test other versions)
* numpy
* scipy

## Datasets
We use BSDS500 for training, validation and testing, and Set11 is for testing.
BSDS500 contains 500 colorful images. And we use its luminance componient for all experiments.
Users can download the pre-processed BSDS500 from [GoogleDrive](https://drive.google.com/file/d/1sghDOPR9Ehucq9yLfQ2pEiG2ckMu70cY/view?usp=sharing),
and extract it under **./dataset/**.

**dataset.py** contains two classes for packaged training sets. 

* class **dataset**: is developed for dataset contains images sized of 33*33.
* class **dataset_full**: is developed for dataset contains images sized of 99*99.

Users can generate and use packaged datasets using this two classes.

## Training
Four forms of AMP-Net are trained in the paper.

* **AMP-Net-*K***: AMP-Net with *K* denoising modules and without deblocking module and trained sampling matrix.
* **AMP-Net-*K*-B**: AMP-Net-*K* with deblocking modules.
* **AMP-Net-*K*-M**: AMP-Net-*K* with the trained sampling matrix.
* **AMP-Net-*K*-BM**: AMP-Net-*K* with deblocking modules and the trained sampling matrix.

`train_AMP_Net.py`, `train_AMP_Net_B.py`, `train_AMP_Net_M.py` and `train_AMP_Net_BM.py` are used to train these four models respectively. 
Trained models can be found in **./results/**.


## Testing
`test_AMP_Net.py`, `test_AMP_Net_B.py`, `test_AMP_Net_M.py` and `test_AMP_Net_BM.py` are used to test above four models respectively.
Using these files, two results can be obtained:

* Average PSNR and SSIM of the test set. 
* Reconstructed images named as **imageName_PSNR_SSIM.jpg**. 

The path of the test set can be set in the the function **get_val_result**.

Generated images are stored in the path as **results/generated_images/model_name/num1/num2/**, where **num1**% is the CS ratio and  **num2** is the number of the iteration.

## Pre-trained models
We provide the pre-trained models used in the paper so that users can use them for testing directly.

All pre-trained AMP-Net models can be found in [GoogleDrive](https://drive.google.com/drive/folders/1O_tX7T__ANWXIWGytpHciFMbMfXviRjv?usp=sharing). These models are stored in the path as
**model_name/num1/num2**,
where **num1**% is the CS ratio and  **num2** is the number of the iteration.
