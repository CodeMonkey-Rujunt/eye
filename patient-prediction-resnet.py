import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from itertools import islice
import csv
from os.path import isfile, join

import model

csvtrain = 'exps/train_gt.csv'
csvval = 'exps/val_gt.csv'

image_base = 'ODIR-5K/Training Images/'

fttrain = 'exps/2-final/train_proba.ft'
ftval = 'exps/2-final/val_proba.ft'

model_path = 'exps/2-final/swav-r50-epoch=27-odir_score_val=0.837-val_loss=0.929-auc_score_val=0.884.ckpt'
cnn = model.SwavFinetuning.load_from_checkpoint(model_path, classes=8)

device = 'cuda'

mean = [0.485, 0.456, 0.406]
std = [0.228, 0.224, 0.225]
trans = []
randomresizedcrop = transforms.RandomResizedCrop(299)
trans = transforms.Compose([
    randomresizedcrop,
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])

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
                image_path = join(image_base, l[0] + '_{}.jpg'.format(side))
                if isfile(image_path):
                    image = Image.open(image_path).convert('RGB')
                    torch_img = trans(image).unsqueeze(0).to(device)
                    
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
