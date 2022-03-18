import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from sklearn import metrics
from sklearn import preprocessing
import cv2
from copy import copy

import datasets

batch_size = 32
learning_rate = 5e-5
epochs = 20

class RandomGaussianBlur(object):
    def __call__(self, img):
        do_it = np.random.rand() > 0.5
        if not do_it:
            return img
        sigma = np.random.rand() * 1.9 + 0.1
        return cv2.GaussianBlur(np.asarray(img), (23, 23), sigma)

def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

def odir_metric(self, gt_data, pr_data):
    th = 0.5
    gt = gt_data.flatten()
    pr = pr_data.flatten()
    kappa = metrics.cohen_kappa_score(gt, pr>th)
    f1 = metrics.f1_score(gt, pr>th, average='micro')
    auc = metrics.roc_auc_score(gt, pr)
    final_score = (kappa+f1+auc)/3.0
    
    return final_score

INPUT_IMG = 224
color_transform = [get_color_distortion(), RandomGaussianBlur()]
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])
randomresizedcrop = transforms.RandomResizedCrop(INPUT_IMG)
transform = transforms.Compose([
    randomresizedcrop,
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Compose(color_transform),
    transforms.ToTensor(),
    normalize,
    ])

# init the dataset withou any augmentation
full_train = datasets.ODIR5K('train', None)

# calculate the validation size
train_size = int(0.8 * len(full_train))
val_size = len(full_train) - train_size

# split the datasts
odir_train, odir_val = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(42))

# trick to disantangle the agumentations variable from train to validation
odir_train.dataset = copy(full_train)

# set the train augmentations
odir_train.dataset.aug = transform

# build the validation augmentations
randomresizedcrop = transforms.RandomResizedCrop(INPUT_IMG)
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.228, 0.224, 0.225])
transform = transforms.Compose([
    randomresizedcrop,
    transforms.ToTensor(),
    normalize,
    ])

odir_val.dataset.aug = transform

odir_test = datasets.ODIR5K('test', transform)

train_loader = DataLoader(odir_train, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(odir_val, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(odir_test, batch_size=batch_size, shuffle=False, num_workers=4)

# training
net = torch.hub.load('facebookresearch/swav', 'resnet50')

d_dim = net.fc.in_features
net.fc = lambda x: x
    
if freeze:
    for p in net.parameters():
        p.requires_grad = False

classes = 8
linear_clf = nn.Linear(d_dim, classes)

optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

prob_preds = []
labels = []

for images, labels in data_loader:
    images = images.to(device)
    labels = labels.to(device)

    representations = net(images).detach()
    logits = linear_clf(representations)
    loss = F.cross_entropy(logits, labels)

    probs = F.softmax(logits, dim=1)
    prob_preds.append(probs.cpu().numpy())
    labels.extend(y.cpu().numpy())

prob_preds = np.concatenate(prob_preds)
auc = metrics.roc_auc_score(labels, prob_preds, average='weighted', multi_class='ovo')
#log('auc_score_{step}', auc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

labels_onehot = preprocessing.OneHotEncoder(sparse=False).fit_transform(np.array(labels).reshape(len(labels), 1))
final_score = odir_metric(labels_onehot, prob_preds)
#log('odir_score_{step}', final_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    
#trainer = pl.Trainer(gpus=[2], max_epochs=epochs)
#trainer.fit(model, train_loader, val_loader)

# test
model_path = 'model/swav.pt'
cnn = model.SwavFinetuning.load_from_checkpoint(model_path, classes=8)

device = torch.device('cuda')
cnn = cnn.to(device)


odir_test = datasets.ODIR5K('test', transform)

test_dataloader = DataLoader(odir_test, batch_size=32) 

cnn.eval()
for index, batch in test_dataloader:
    print(index, batch)
