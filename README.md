# Ocular Disease Recognition with SwAV using PyTorch

## Introduction

Early diagnosis and treatment of ophthalmic diseases such as glaucoma, catarct and AMD are important because they significantly reduce quality of life.
Eye fundus images are useful in identifying such ophthalmologic diseases.
Here I will introduce ocular disease recognition model that uses eye fundus images as input and trains multi-labels of ocular diseases in a supervised learning.
The results of inference were evaluated by area under the curve (AUC) of reciever operating characteristics (ROC).

## Dataset

The `ODIR-5K` dataset includes both eye fundus images of 3500 patients for training.
These eye fundus images are annotated with 8 labels indicating normal, 6 major ocular diseases and other ocuar diseases.

<img src="figure/input.png" alt="input" width="500px" />

### Count of Eye Fundus Images for each Disease Labels

```
               train   test  train+test      %
Normal        2541.0  275.0      2816.0   40.0
Diabetes      1598.0  180.0      1778.0   25.3
Glaucoma       274.0   39.0       313.0    4.4
Cataract       251.0   24.0       275.0    3.9
AMD            250.0   30.0       280.0    4.0
Hypertension   174.0   18.0       192.0    2.7
Myopia         240.0   22.0       262.0    3.7
Others         995.0  129.0      1124.0   16.0
Total         6323.0  717.0      7040.0  100.0
```

## Model

`SwAV`, Swappnig Assignments between Views with ResNet-50.


## Results

```bash
python main.py

epoch   1/300 batch  99/ 99 loss 2.2418 235.7sec mean AUC 0.542 (0.533 0.522 0.555 0.713 0.505 0.438 0.566 0.502)
epoch   2/300 batch  99/ 99 loss 2.0943 238.3sec mean AUC 0.653 (0.584 0.599 0.684 0.943 0.513 0.467 0.921 0.510)
epoch   3/300 batch  99/ 99 loss 2.0550 239.3sec mean AUC 0.655 (0.590 0.613 0.702 0.957 0.544 0.373 0.965 0.500)
epoch   4/300 batch  99/ 99 loss 2.0427 237.9sec mean AUC 0.664 (0.590 0.609 0.751 0.963 0.539 0.385 0.975 0.505)
epoch   5/300 batch  99/ 99 loss 2.0321 238.5sec mean AUC 0.672 (0.598 0.619 0.748 0.955 0.534 0.427 0.983 0.513)
...
epoch 296/300 batch  99/ 99 loss 1.7840 235.8sec mean AUC 0.892 (0.946 0.938 0.967 0.993 0.530 0.818 0.999 0.941)
epoch 297/300 batch  99/ 99 loss 1.7882 234.9sec mean AUC 0.888 (0.947 0.930 0.962 0.994 0.531 0.805 0.999 0.939)
epoch 298/300 batch  99/ 99 loss 1.7858 233.3sec mean AUC 0.894 (0.949 0.932 0.966 0.991 0.572 0.809 0.999 0.936)
epoch 299/300 batch  99/ 99 loss 1.7865 235.7sec mean AUC 0.891 (0.949 0.936 0.974 0.995 0.557 0.780 0.999 0.940)
epoch 300/300 batch  99/ 99 loss 1.7881 234.9sec mean AUC 0.899 (0.944 0.934 0.978 0.995 0.546 0.861 0.998 0.939)

test batch  11/ 11 loss 1.8079 20.4sec mean AUC 0.895 (0.972 0.928 0.970 0.992 0.531 0.833 0.998 0.932)
```

## Refrences

- [ODIR-5K](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)
  Eye fundus images.
- [ODIR-2019](https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning)
  Disease labels for each eye fundus image.
- [SwAV](https://github.com/facebookresearch/swav)
