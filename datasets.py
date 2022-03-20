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
        self.csvfile = 'labels/eye_labels_%s.csv' % (self.mode)
        self.df = pd.read_csv(self.csvfile)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        imagefile = row['ID']
        imagefile = 'data/images/%s/%s' % (self.mode, imagefile.split('/')[-1])
        label = torch.FloatTensor([int(i) for i in row['N':'O']])
        
        if os.path.exists(imagefile):
            image = Image.open(imagefile).convert('RGB')
        else:
            print('%s not found' % (imagefile))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
        ])
    
    dataset = ODIR5K('train', transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for image, label in dataloader:
        print(image, label)
        image = image.view(3, 500, 500).permute(1, 2, 0)
        plt.imshow(image)
        plt.savefig('figure/input.png')

        break
