import numpy as np
import torch
from torch.nn import functional as F
from Datasets import Modes, ODIR5K
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import pytorch_lightning as pl

import model

model_path = 'exps/2-final/swav-r50-epoch=27-odir_score_val=0.837-val_loss=0.929-auc_score_val=0.884.ckpt'
cnn = model.SwavFinetuning.load_from_checkpoint(model_path, classes=8)

device = 'cuda'
cnn = cnn.eval().to(device)

INPUT_IMG = 224

mean = [0.485, 0.456, 0.406]
std = [0.228, 0.224, 0.225]

#build the validation augmentations
trans = []
randomresizedcrop = transforms.RandomResizedCrop(INPUT_IMG)
trans = transforms.Compose([
    randomresizedcrop,
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])

test_t = trans

odir_test = ODIR5K(Modes.test, test_t)

test_dl = DataLoader(odir_test, batch_size=32) 
trainer = pl.Trainer(gpus=[1])
trainer.test(cnn, test_dl)

test_dl = DataLoader(odir_test, batch_size=32) 

batch = next(iter(test_dl))
