import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os

class ODIR5K(Dataset):
    def __init__(self, mode, transform=None):
        super(ODIR5K, self).__init__()
        self.mode = mode
        self.csvfile = 'labels/%s.csv' % (self.mode)
        self.df = pd.read_csv(self.csvfile)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def get_image(self, row, eye):
        image = row['%s-Fundus' % (eye)]
        image = 'data/images/%s/%s' % (self.mode, image)
        if os.path.exists(image):
            image = Image.open(image).convert('RGB')
        else:
            print('%s not found' % (image))

        return image

    def __getitem__(self, index):
        row = self.df.iloc[index]

        left = self.get_image(row, 'Left')
        right = self.get_image(row, 'Right')

        label = torch.FloatTensor([int(i) for i in row['N':'O']])
        
        if self.transform:
            left = self.transform(left)
            right = self.transform(right)
        
        return left, right, label

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
        ])
    
    dataset = ODIR5K('train', transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for left_image, right_image, label in dataloader:
        print(left_image.shape, right_image.shape, label)
        left_image = left_image.view(3, 500, 500).permute(1, 2, 0)
        right_image = right_image.view(3, 500, 500).permute(1, 2, 0)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(left_image)
        plt.subplot(1, 2, 2)
        plt.imshow(right_image)
        plt.tight_layout()
        plt.savefig('figure/input.png')

        break
