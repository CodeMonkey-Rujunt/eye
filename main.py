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

def odir_metric(gt_data, pr_data):
    threshold = 0.5
    gt = gt_data.flatten()
    pr = pr_data.flatten()
    kappa = metrics.cohen_kappa_score(gt, pr > threshold)
    f1 = metrics.f1_score(gt, pr > threshold, average='micro')
    auc = metrics.roc_auc_score(gt, pr)
    final_score = (kappa+f1+auc)/3.0
    
    return final_score

def data_load(batch_size=32):
    INPUT_IMG = 224

    color_transform = [get_color_distortion(), RandomGaussianBlur()]
    randomresizedcrop = transforms.RandomResizedCrop(INPUT_IMG)
    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.228, 0.224, 0.225])
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
    odir_train.dataset.augmentations = transform

    # build the validation augmentations
    transform = transforms.Compose([
        randomresizedcrop,
        transforms.ToTensor(),
        normalize,
        ])

    odir_val.dataset.augmentations = transform
    odir_test = datasets.ODIR5K('test', transform)

    train_loader = DataLoader(odir_train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(odir_val, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(odir_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader

def test():
    prob_preds = []
    y_true = []
    y_pred = []
    net.eval()
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            representations = net(images).detach()
            logits = linear_clf(representations)

        probs = F.softmax(logits, dim=1)
        prob_preds.append(probs.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(torch.argmax(probs, dim=1).cpu().numpy())
    
    prob_preds = np.concatenate(prob_preds)
    auc = metrics.roc_auc_score(y_true, prob_preds, average='weighted', multi_class='ovo')
    print(auc)
    
    labels_onehot = preprocessing.OneHotEncoder(sparse=False).fit_transform(np.array(labels).reshape(len(labels), 1))
    final_score = odir_metric(labels_onehot, prob_preds)
    print(final_score)
    
    if step == 'test':
        cm = metrics.confusion_matrix(labels, y_pred)
        target_names = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
        plot_confusion_matrix(cm, target_names=target_names, auc=auc, normalize=False)
        print(cm)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

def main(epochs=20, classes=8, learning_rate=5e-5, freeze=False):
    device = torch.device('cuda:1')

    train_loader = data_load()

    net = torch.hub.load('facebookresearch/swav', 'resnet50')
    net = net.to(device)

    d_dim = net.fc.in_features
    net.fc = Identity()
        
    if freeze:
        for p in net.parameters():
            p.requires_grad = False

    linear_clf = nn.Linear(d_dim, classes)
    linear_clf = linear_clf.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    y_pred = []
    y_true = []
    net.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            representations = net(images).detach()
            logits = linear_clf(representations)
            loss = F.cross_entropy(logits, labels)

            probs = F.softmax(logits, dim=1)
            y_pred += probs.detach().cpu()
            y_true += labels.detach().cpu()

        auc = metrics.roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovo')
        print(auc)

        labels_onehot = preprocessing.OneHotEncoder(sparse=False).fit_transform(np.array(y_true).reshape(len(y_true), 1))
        final_score = odir_metric(labels_onehot, y_pred)
        print(final_score)

if __name__ == '__main__':
    main()
