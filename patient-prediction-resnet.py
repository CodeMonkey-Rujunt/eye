import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import os

image_base = 'data/images'

csvtrain = 'labels/train_gt.csv'
csvval = 'labels/val_gt.csv'

device = torch.device('cuda')

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.228, 0.224, 0.225])
randomresizedcrop = transforms.RandomResizedCrop(299)
transform = transforms.Compose([
    randomresizedcrop,
    transforms.ToTensor(),
    normalize,
    ])

model_path = 'model/swav.pt'
#net = model.SwavFinetuning.load_from_checkpoint(model_path, classes=8)
#net = net.to(device)

#net.eval()
for csvfile in (csvtrain, csvval):
    df = pd.read_csv(csvfile)
    print(df)

    feature_matrix = np.zeros((df.shape[0], 16), np.float)
    y_true = np.zeros((df.shape[0], 8), np.int)
    patient_id = np.zeros([df.shape[0]], np.int)
    count = 0

    for index, row in df.iterrows():
        for side in ('left', 'right'):
            image_path = '%s/%s_%s.jpg' % (image_base, row['ID'], side)

            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                image = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    representations = net(image).detach()
                    probs = torch.softmax(net.linear_clf(representations), axis=1)

                if side == 'left':
                    feature_matrix[count][0:8] = probs.cpu().numpy().flatten()

                if side == 'right':
                    feature_matrix[count][8:16] = probs.cpu().numpy().flatten()
            else:
                print('%s not found' % (image_path))

        patient_id[count] = int(row['ID'])
        y_true[count] = [int(label) for label in row[1:9]]
        count += 1

    df = pd.DataFrame({
        'id': patient_id,
        'eyes_feature': [f for f in feature_matrix],
        'y_true': [y for y in y_true]})

    print(df)
