# Ocular Disease Recognition with PyTorch

## Introduction

Early diagnosis and treatment of ophthalmic diseases such as glaucoma and AMD are important because they significantly reduce quality of life.
Eye fundus images are useful in identifying such ophthalmologic diseases.
Here, eye fundus images and annotations are used as input of deep learning, and inference is performed using multi-label classification.

## Dataset

The `ODIR-5K` dataset consists of over 7000 eye fundus images.
These are annotated with 8 labels, which are 6 major ocular disease labels and normal and other labels.
The labels are annotatted on each of the left and right eye fundus images as multi-label data.

<img src="figure/input.png" alt="input" width="300px" />

### Number of Labels

```
               train   test  train+test      %
Normal          2791    307        3098   42.9
Diabetes        1603    192        1795   24.9
Glaucoma         288     38         326    4.5
Cataract         288     25         313    4.3
AMD              259     21         280    3.9
Hypertension     171     21         192    2.7
Myopia           236     29         265    3.7
Others           849    103         952   13.2
Total           6485    736        7221  100.0
```

## Model

`SwAV`, Swappnig Assignments between Views with ResNet-50.


## Results

```
epoch   1/150 batch 197/197 loss 1.8741 241.446sec average AUC 0.582 (0.549 0.551 0.540 0.790 0.529 0.485 0.689 0.524)
epoch   2/150 batch 197/197 loss 1.7675 237.068sec average AUC 0.670 (0.601 0.596 0.694 0.949 0.567 0.432 0.953 0.564)
epoch   3/150 batch 197/197 loss 1.7469 237.833sec average AUC 0.679 (0.582 0.619 0.708 0.950 0.566 0.466 0.956 0.589)
epoch   4/150 batch 197/197 loss 1.7401 237.183sec average AUC 0.676 (0.569 0.634 0.699 0.948 0.536 0.439 0.962 0.619)
epoch   5/150 batch 197/197 loss 1.7295 236.221sec average AUC 0.681 (0.573 0.644 0.683 0.954 0.539 0.446 0.969 0.637)
...
epoch 145/150 batch 197/197 loss 1.5418 237.228sec average AUC 0.879 (0.856 0.881 0.933 0.983 0.909 0.569 0.992 0.906)
epoch 146/150 batch 197/197 loss 1.5449 235.480sec average AUC 0.879 (0.853 0.881 0.928 0.982 0.912 0.580 0.997 0.896)
epoch 147/150 batch 197/197 loss 1.5356 236.611sec average AUC 0.883 (0.856 0.887 0.934 0.981 0.914 0.590 0.996 0.909)
epoch 148/150 batch 197/197 loss 1.5434 235.995sec average AUC 0.880 (0.842 0.875 0.943 0.980 0.907 0.600 0.995 0.901)
epoch 149/150 batch 197/197 loss 1.5454 235.399sec average AUC 0.879 (0.841 0.873 0.942 0.977 0.913 0.584 0.997 0.903)
epoch 150/150 batch 197/197 loss 1.5371 233.698sec average AUC 0.878 (0.848 0.885 0.928 0.980 0.898 0.582 0.997 0.906)
```

## Refrences

- [SwAV](https://github.com/facebookresearch/swav)
- [ODIR-5K](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)
