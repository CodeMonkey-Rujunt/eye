import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os

class ODIR5K(Dataset):
    def __init__(self, mode, augmentations=None):
        super(ODIR5K, self).__init__()
        self.mode = mode
        self.csv_path = 'labels/pytorch_fake%s.csv' % (self.mode)
        self.df = pd.read_csv(self.csv_path)
        self.augmentations = augmentations
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        csv_line = self.df.iloc[index]
        img_path = csv_line['path']
        img_path = 'data/images/%s/%s' % (self.mode, img_path.split('/')[-1])
        label = torch.tensor(int(csv_line['label']))
        
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
        else:
            print('%s not found' % (img_path))
        
        if self.augmentations:
            image = self.augmentations(image)
        
        return image, label

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
        ])
    
    dataset = ODIR5K('train', transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for image, label in dataloader:
        image = image.view(3, 500, 500).permute(1, 2, 0)
        plt.imshow(image)
        plt.savefig('figure/input.png')

        break
