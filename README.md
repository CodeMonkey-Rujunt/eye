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
              train  test  train+test    %
Normal         2789   309        3098   42
Diabetes       1619   176        1795   24
Glaucoma        296    30         326    4
Cataract        288    25         313    4
AMD             252    28         280    3
Hypertension    170    22         192    2
Myopia          235    30         265    3
Others          854    98         952   13
Total          6503   718        7221  100
```

## Model

Swappnig Assignments between Views (SwAV) with ResNet-50.

## Results

```
epoch   1/150 batch 180/180 loss 1.8155 229.013sec average AUC 0.580 (0.548 0.551 0.564 0.827 0.521 0.434 0.675 0.519)
epoch   2/150 batch 180/180 loss 1.7033 213.026sec average AUC 0.670 (0.596 0.621 0.650 0.953 0.623 0.453 0.924 0.541)
epoch   3/150 batch 180/180 loss 1.6810 215.956sec average AUC 0.692 (0.593 0.635 0.704 0.956 0.653 0.439 0.965 0.587)
epoch   4/150 batch 180/180 loss 1.6712 252.009sec average AUC 0.700 (0.589 0.644 0.690 0.960 0.674 0.469 0.969 0.604)
epoch   5/150 batch 180/180 loss 1.6636 356.533sec average AUC 0.699 (0.585 0.653 0.670 0.955 0.665 0.462 0.968 0.631)
...
epoch 146/150 batch 180/180 loss 1.4670 216.017sec average AUC 0.888 (0.863 0.892 0.945 0.985 0.924 0.577 0.997 0.918)
epoch 147/150 batch 180/180 loss 1.4749 215.701sec average AUC 0.882 (0.850 0.888 0.942 0.983 0.928 0.554 0.996 0.917)
epoch 148/150 batch 180/180 loss 1.4767 215.474sec average AUC 0.887 (0.851 0.890 0.937 0.986 0.940 0.592 0.997 0.902)
epoch 149/150 batch 180/180 loss 1.4753 218.688sec average AUC 0.884 (0.854 0.890 0.943 0.989 0.935 0.568 0.994 0.903)
epoch 150/150 batch 180/180 loss 1.4695 221.733sec average AUC 0.890 (0.856 0.892 0.948 0.986 0.936 0.585 0.998 0.918)
```
