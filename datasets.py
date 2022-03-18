import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class ODIR5K(Dataset):
    def __init__(self, mode, augmentations):
        super(ODIR5K, self).__init__()
        self.mode = mode
        self.csv_path = 'data/pytorch_fake{self.mode}.csv'
        self.csv = pd.read_csv(self.csv_path)
        self.aug = augmentations
    
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        csv_line = self.csv.iloc[index]
        img_path = csv_line['path']
        label = torch.tensor(int(csv_line['label']))
        
        image = Image.open(img_path).convert('RGB')
        
        if self.aug is not None:
            image = self.aug(image)
        
        return image, label

if __name__ == '__main__':
    train, test = train_test_split(list(range(len(odir_train))), test_size=0.20, random_state=42)
    transform = transforms.Compose([transforms.Resize((500, 500)), transforms.ToTensor()])
    
    dataset = ODIR5K('train', transform, train)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))

    for image, label in dataloader:
        image = np.transpose(image.numpy(), (1, 2, 0))
        plt.figure()
        plt.imshow(image)
        plt.show()
