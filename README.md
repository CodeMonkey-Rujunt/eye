# Ocular Disease Recognition with SwAV using PyTorch

## Introduction

Early diagnosis and treatment of ophthalmic diseases such as glaucoma, catarct and AMD are important because they significantly reduce quality of life (QoL).
Eye fundus images are useful in identifying such ophthalmologic diseases.
Here I will introduce ocular disease recognition model that uses eye fundus images as input and trains multi-labels of ocular diseases in a supervised learning.
The results of inference were evaluated by area under the curve (AUC) of reciever operating characteristics (ROC).

## Dataset

The `ODIR-5K` dataset includes both eyes of more than 4000 patients.
These eye fundus images are annotated with 8 labels indicating normal, 6 major ocular diseases and other ocuar diseases.
Since labels are attached to both eyes, the ocular disease recognition model uses the fundus images of both eyes as input.

<img src="figure/input.png" alt="input" width="500px" />

### Number of Labels

```
             train   test  train+test        %
Normal        1015    125        1140     27.7
Diabetes      1012    116        1128     27.4
Glaucoma       196     19         215      5.2
Cataract       192     20         212      5.2
AMD            145     19         164      4.0
Hypertension    94      9         103      2.5
Myopia         155     19         174      4.2
Others         891     88         979     23.8
Total         3700    415        4115    100.0
```

## Model

`SwAV`, Swappnig Assignments between Views with ResNet-50.


## Results

```
epoch   1/300 batch 197/197 loss 1.5172 472.393sec average AUC 0.576 (0.537 0.556 0.552 0.792 0.512 0.502 0.659 0.500)
epoch   2/300 batch 197/197 loss 1.4231 470.192sec average AUC 0.663 (0.571 0.629 0.616 0.961 0.561 0.462 0.951 0.551)
epoch   3/300 batch 197/197 loss 1.4081 469.354sec average AUC 0.668 (0.551 0.648 0.634 0.962 0.556 0.462 0.972 0.556)
epoch   4/300 batch 197/197 loss 1.4014 472.703sec average AUC 0.685 (0.561 0.667 0.646 0.967 0.563 0.531 0.977 0.568)
epoch   5/300 batch 197/197 loss 1.3958 474.009sec average AUC 0.677 (0.562 0.681 0.580 0.973 0.563 0.500 0.980 0.577)
...
epoch 296/300 batch  99/ 99 loss 1.7840 235.859sec average AUC 0.892 (0.946 0.938 0.967 0.993 0.530 0.818 0.999 0.941)
epoch 297/300 batch  99/ 99 loss 1.7882 234.929sec average AUC 0.888 (0.947 0.930 0.962 0.994 0.531 0.805 0.999 0.939)
epoch 298/300 batch  99/ 99 loss 1.7858 233.380sec average AUC 0.894 (0.949 0.932 0.966 0.991 0.572 0.809 0.999 0.936)
epoch 299/300 batch  99/ 99 loss 1.7865 235.773sec average AUC 0.891 (0.949 0.936 0.974 0.995 0.557 0.780 0.999 0.940)
epoch 300/300 batch  99/ 99 loss 1.7881 234.934sec average AUC 0.899 (0.944 0.934 0.978 0.995 0.546 0.861 0.998 0.939)
```

## Refrences

- [SwAV](https://github.com/facebookresearch/swav)
- [ODIR-5K](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)
