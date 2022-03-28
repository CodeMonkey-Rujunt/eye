import pandas as pd
import numpy as np
from sklearn import metrics

EYE_TRAIN_GT = 'labels/eye_labels_train.csv'
EYE_VAL_GT = 'labels/eye_labels_val.csv'

def evaluate(y_val_res, resultfile):
    df_dict = dict()
    df_dict[GT_HEADER[0]] = X_val_id
    for i in range(1,len(GT_HEADER)):
        df_dict[GT_HEADER[i]] = y_val_res[:,i-1]

    df = pd.DataFrame(df_dict)
    #df.to_csv(resultfile, index=False)

    gt_data = import_gt(VAL_GT_XLSX)
    pr_data, wrong_col_order, wrong_row_order, missing_results = import_pred(gt_data, resultfile)

    # calculate kappa, F-1 score and AUC value
    threshold = 0.5
    gt = gt_data[:, 1:].flatten()
    pr = pr_data[:, 1:].flatten()

    kappa = metrics.cohen_kappa_score(gt, pr > threshold)
    f1 = metrics.f1_score(gt, pr > threshold, average='micro')
    auc = metrics.roc_auc_score(gt, pr)
    final_score = (kappa + f1 + auc) / 3.0

    print('kappa score:', kappa, 'f-1 score:', f1, 'AUC vlaue:', auc, 'Final Score:', final_score)

def preprocess():
    # read file with labels from each eye
    csvfile = pd.read_csv(CSVFILE)
    labels_dict = pd.Series(csvfile.target.values, index=csvfile.filename).to_dict()

    # read files containing target for train and val patients
    df_train_gt = pd.read_csv(TRAIN_GT)
    df_val_gt = pd.read_csv(VAL_GT)

    train_set = set(df_train_gt['ID'].to_numpy())
    val_set = set(df_val_gt['ID'].to_numpy())

    X_train = np.zeros([len(train_set)*2], np.object)
    y_train = np.zeros([len(train_set)*2, 8], np.int)
    X_train_id = np.zeros([len(train_set)], np.object)
    X_val = np.zeros([len(val_set)*2], np.object)
    X_val_id = np.zeros([len(val_set)], np.object)

    df = pd.read_excel(XLSXFILE)

    i_train, i_test = 0, 0
    for index, row in df.iterrows():
        if row['ID'] in train_set:
            X_train[2*i_train] = row['Left-Fundus']
            y_train[2*i_train] = [int(i) for i in labels_dict[row['Left-Fundus']][1:-1].split(', ')]
            X_train[2*i_train + 1] = row['Right-Fundus']
            y_train[2*i_train + 1] = [int(i) for i in labels_dict[row['Right-Fundus']][1:-1].split(', ')]
            X_train_id[i_train] = row['ID']
            i_train += 1

        elif row['ID'] in val_set:
            X_val[2*i_test] = row['Left-Fundus']
            X_val[2*i_test + 1] = row['Right-Fundus']
            X_val_id[i_test] = row['ID']
            i_test += 1
    
    eye_train = pd.read_csv(EYE_TRAIN_GT)
    X_eye_train = eye_train['ID'].to_numpy()

    y_eye_train = np.zeros([len(X_eye_train), 8], dtype=np.int)
    for index, row in eye_train.iterrows():
        for (i,j) in (('N',0), ('D',1), ('G',2), ('C',3), ('A',4), ('H',5), ('M',6), ('O',7)):
            if row[i] == 1:
                y_eye_train[index][j] = 1

    eye_val = pd.read_csv(EYE_VAL_GT)
    X_eye_val = eye_val['ID'].to_numpy()

if __name__ == '__main__':
    main()
