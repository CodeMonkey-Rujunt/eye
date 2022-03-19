import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from sklearn import metrics
from sklearn import preprocessing
import cv2

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

    # init the dataset and augmentations
    train_dataset = datasets.ODIR5K('train', transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # build the test augmentations
    transform = transforms.Compose([
        randomresizedcrop,
        transforms.ToTensor(),
        normalize,
        ])

    test_dataset = datasets.ODIR5K('test', transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader

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

    train_loader, test_loader = data_load()

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

    #y_true = []
    #y_pred = []
    net.train()
    for epoch in range(epochs):
        train_loss = 0
        for index, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device)
            labels = labels.to(device)

            representations = net(images) #.detach()
            logits = linear_clf(representations)
            loss = F.cross_entropy(logits, labels)
            train_loss += loss.item()

            probs = F.softmax(logits, dim=1)
            #y_true.append(labels.detach().cpu().numpy())
            #y_pred.append(probs.detach().cpu())

            print('\repoch %2d batch %2d/%2d loss %5.3f' % (epoch+1, index, len(train_loader), train_loss / index), end='')

        print('')

        '''
        auc = metrics.roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovo')
        print(auc)

        labels_onehot = preprocessing.OneHotEncoder(sparse=False).fit_transform(np.array(y_true).reshape(len(y_true), 1))
        final_score = odir_metric(labels_onehot, y_pred)
        print(final_score)
        '''

if __name__ == '__main__':
    main()
