import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from sklearn import metrics
from sklearn import preprocessing
import argparse
import timeit
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

def metric(gt_data, pr_data):
    threshold = 0.5
    gt = gt_data.flatten()
    pr = pr_data.flatten()
    kappa = metrics.cohen_kappa_score(gt, pr > threshold)
    f1 = metrics.f1_score(gt, pr > threshold, average='micro')
    auc = metrics.roc_auc_score(gt, pr)
    final_score = (kappa+f1+auc)/3.0
    
    return final_score

def data_load(args):
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # build the test augmentations
    transform = transforms.Compose([
        randomresizedcrop,
        transforms.ToTensor(),
        normalize,
        ])

    test_dataset = datasets.ODIR5K('test', transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

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
    final_score = metric(labels_onehot, prob_preds)
    print(final_score)
    
    cm = metrics.confusion_matrix(labels, y_pred)
    target_names = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    plot_confusion_matrix(cm, target_names=target_names, auc=auc, normalize=False)
    print(cm)

def main(args):
    device = torch.device('cuda:1')

    train_loader, test_loader = data_load(args)

    net = torch.hub.load('facebookresearch/swav', 'resnet50')

    num_features = net.fc.in_features
    net.fc = nn.Sequential(
            nn.Linear(num_features, args.classes),
            nn.Sigmoid())

    net = net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    #optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = torch.nn.CrossEntropyLoss()

    net.train()
    for epoch in range(args.epochs):
        start_time = timeit.default_timer()
        y_true = torch.FloatTensor()
        y_pred = torch.FloatTensor()
        train_loss = 0
        for index, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_true = torch.cat((y_true, labels.cpu()))
            y_pred = torch.cat((y_pred, outputs.detach().cpu()))

            print('\repoch %2d/%2d batch %3d/%3d' % (epoch+1, args.epochs, index, len(train_loader)), end='')
            print(' loss %6.4f' % (train_loss / index), end='')
            print(' %6.3fsec' % (timeit.default_timer() - start_time), end='')

        print('')
        aucs = [metrics.roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(args.classes)]
        auc_classes = ' '.join(['%5.3f' % (aucs[i]) for i in range(args.classes)])
        print('The average AUC is %5.3f (%s)' % (np.mean(aucs), auc_classes))
        torch.save(net.state_dict(), 'model/checkpoint.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--classes', default=8, type=int)
    parser.add_argument('--lr', default=1e-4, type=float) # 5e-5
    parser.add_argument('--momentum', default=0.9, type=float)
    args = parser.parse_args()
    print(vars(args))

    main(args)
