import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from itertools import islice
import csv
from os

import model

csvtrain = 'exps/train_gt.csv'
csvval = 'exps/val_gt.csv'

image_base = 'ODIR-5K/Training Images/'

fttrain = 'exps/2-final/train_proba.ft'
ftval = 'exps/2-final/val_proba.ft'

model_path = 'model/swav.pt'
cnn = model.SwavFinetuning.load_from_checkpoint(model_path, classes=8)

device = torch.device('cuda')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])
randomresizedcrop = transforms.RandomResizedCrop(299)
transform = transforms.Compose([
    randomresizedcrop,
    transforms.ToTensor(),
    normalize,
    ])

cnn = cnn.to(device).eval()

for csvfile, ftfile in ((csvtrain, fttrain), (csvval, ftval)):
    with open(csvfile, 'r') as fp:
        csvreader = csv.reader(fp)
        lines = len(list(islice(csvreader,1,None)))

        feature_matrix = np.zeros((lines, 16), np.float)
        y_true = np.zeros((lines, 8), np.int)
        patient_id = np.zeros([lines], np.int)
        count = 0

        fp.seek(0, 0)
        
        csvreader = csv.reader(fp)
        
        for l in islice(csvreader,1,None):
            for side in ('left', 'right'):
                image_path = '%s/%s_%s.jpg' % (image_base, l[0], side))
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert('RGB')
                    torch_img = transform(image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        representations = cnn(torch_img).detach()
                    
                        probs = torch.softmax(cnn.linear_clf(representations), axis=1)
                    
                    if side == 'left':
                        feature_matrix[count][0:8] = probs.cpu().numpy().flatten()
                    if side == 'right':
                        feature_matrix[count][8:16] = probs.cpu().numpy().flatten()
                
            patient_id[count] = int(l[0])
            y_true[count] = [int(label) for label in l[1:9]]
            count+= 1
        
        print('Converting DataFrame...')

        df = pd.DataFrame({'id':patient_id,
                'eyes_feature':[f for f in feature_matrix],
                'y_true':[y for y in y_true]})

        print('Saving DataFrame')

        df.to_feather(ftfile)
