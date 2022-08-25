# Super-resolution domain adaptation networks for semantic segmentation via pixel and output level aligning
Pytorch implementation of our method for adapting semantic segmentation 
from the low-resolution remote sensing dataset (source domain) to the high-resolution remote sensing dataset.

Contact: Zhenjie Tang (tangzhenjie.hebut@gmail.com) 

## Paper
Wu J, Tang Z, Xu C, Liu E, Gao L and Yan W (2022), Super-resolution domain adaptation networks for semantic segmentation via pixel and output level aligning. Front. Earth Sci. 10:974325. doi: 10.3389/feart.2022.974325

Please cite our paper if you find it useful for your research.

## Example Results

![](figure/github1.png)

## Quantitative Reuslts

![](figure/github2.png)

## Installation
* Install Pytorch 1.3.0 from http://pytorch.org with python 3.6 and CUDA 10.1

* Clone this repo
```
git clone https://github.com/tangzhenjie/SRDA-Net
cd SRDA-Net
```
## Dataset
* Download the [Massachusetts Buildings Dataset](https://www.cs.toronto.edu/~vmnih/data/) 
 Training Set as the source domain, and put it `./datasets` folder
 
 * Download the [Inria Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/)
 as the target domain, and put it to `./datasets` folder.
 
 * Create the Mass-Inria dataset
 ```
cd datasets
python create_train_oneclass.py
python create_val_oneclass.py
```
 ## Testing
 * Download the [checkpoint](https://pan.baidu.com/s/1NnwBMB2aqAMv5ufcmtH4XA) 提取码：dw0p
 to `./checkpoints/mass_inria/`
 
 * run
  ```
cd datasets
python val.py --name mass_inria --dataroot ./datasets/mass-inria  --model srdanet_step2 --num_classes 2 --dataset_mode srdanetval --resize_size 188
```
## Training Examples
* pre-training the SRDA-Net
 ```
cd datasets
python train.py --name mass_inria_step1 --dataroot ./datasets/mass-inria  --model srdanet_step1 --num_classes 2 --dataset_mode srdanet --A_crop_size 114 --B_crop_size 380
```
* copy the weight of re-training srdanet to the `./checkpoints/mass_inria_step2/` then, run (num is the epoch num of pre-training)
 ```
cd datasets
python train.py --name mass_inria_step2 --dataroot ./datasets/mass-inria  --model srdanet_step2 --num_classes 2 --dataset_mode srdanet --A_crop_size 114 --B_crop_size 380 --epoch num
```
## Acknowledgment
This code is heavily borrowed from [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

