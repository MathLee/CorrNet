# CorrNet
[TGRS2022] [CorrNet] Lightweight Salient Object Detection in Optical Remote Sensing Images via Feature Correlation

# Network Architecture
   <div align=center>
   <img src="https://github.com/MathLee/CorrNet/blob/main/images/CorrNet.png">
   </div>
   
# Accuracy v.s. Parameters
   <div align=center>
   <img src=https://github.com/MathLee/CorrNet/blob/main/images/accuracyVSparams.png width=52% />
   </div> 
   
   
# Requirements
   python 2.7 + pytorch 0.4.0 or
   
   python 3.7 + pytorch 1.9.0
   

# Saliency maps
   We provide saliency maps of [all compared methods](https://pan.baidu.com/s/1TP6An1VWygGUy4uvojL0bg) (code: 5np3) and [our MCCNet](https://pan.baidu.com/s/10JIKL2Q48RvBGeT2pmPfDA) (code: 3pvq) on ORSSD and EORSSD datasets.
   
   In addition, we also provide [saliency maps of our MCCNet](https://pan.baidu.com/s/1dz-GeELIqMdzKlPvzETixA) (code: 413m) on the recently published [ORSI-4199](https://github.com/wchao1213/ORSI-SOD) dataset.
   
   ![Image](https://github.com/MathLee/CorrNet/blob/main/images/table.png)
   
# Training

We get the ground truth of edge using [sal2edge.m](https://github.com/JXingZhao/EGNet/blob/master/sal2edge.m) in [EGNet](https://github.com/JXingZhao/EGNet)，and use data_aug.m for data augmentation.

Modify pathes of [VGG backbone](https://pan.baidu.com/s/1YQxKZ-y2C4EsqrgKNI7qrw) (code: ego5) and datasets, then run train_MCCNet.py.


# Pre-trained model and testing
Download the following pre-trained model, and modify pathes of pre-trained model and datasets, then run test_MCCNet.py.

[ORSSD](https://pan.baidu.com/s/1LdUE8F11r61r8wk3Y9wPLA) (code: awqr)

[EORSSD](https://pan.baidu.com/s/14LrEt1LW5QmZvkhsgbKgfg) (code: wm3p)

[ORSI-4199](https://pan.baidu.com/s/1hmANQp9cslyPuDE-3NlqAg) (code: 336a)

   
# Evaluation Tool
   You can use the [evaluation tool (MATLAB version)](http://dpfan.net/d3netbenchmark/) to evaluate the above saliency maps.


# [ORSI-SOD_Summary](https://github.com/MathLee/ORSI-SOD_Summary)
   
# Citation
        @ARTICLE{Li_2021_MCCNet,
                author = {Gongyang Li and Zhi Liu and Zhen Bai and Weisi Lin and Haibin Ling},
                title = {Lightweight Salient Object Detection in Optical Remote Sensing Images via Feature Correlation},
                journal = {IEEE Transactions on Geoscience and Remote Sensing},
                year = {2021},
                }
                
                
If you encounter any problems with the code, want to report bugs, etc.

Please contact me at lllmiemie@163.com or ligongyang@shu.edu.cn.
