import pandas as pd
import numpy as np
from sklearn import metrics
import sys

CSVFILE = '/work/ocular-dataset/full_df.csv'
XLSXFILE = 'data/labels/data.xlsx'

TRAIN_GT = 'labels/train_gt.csv'
VAL_GT = 'labels/val_gt.csv'
EYE_TRAIN_GT = 'labels/eye_labels_train.csv'
EYE_VAL_GT = 'labels/eye_labels_val.csv'

VAL_GT_XLSX = 'labels/val_gt.xlsx'

GT_HEADER = ['ID', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

# read the ground truth from xlsx file and output case id and eight labels 
def import_gt(filepath):
    data = pd.ExcelFile(filepath)
    table = book.parse(book.sheet_names[0])
    data = [[int(table.row_values(i, 0, 1)[0])] + table.row_values(i, -8) for i in range(1, table.nrows)]

    return np.array(data)

# read the submitted predictions in csv format and output case id and eight labels 
def import_pred(gt_data, filepath):
    df = pd.read_csv(filepath)
    pr_data = [[int(row[0])] + list(map(float, row[1:])) for i, row in df.iterrows()]
    pr_data = np.array(pr_data)
    
    # Sort columns if they are not in predefined order
    order = ['ID','N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    order_index = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    order_dict = { item: ind for ind, item in enumerate(order) }
    sort_index = [order_dict[item] for ind, item in enumerate(header) if item in order_dict]
    wrong_col_order = 0
    if(sort_index != order_index):
        wrong_col_order = 1
        pr_data[:, order_index] = pr_data[:, sort_index] 
    
    # Sort rows if they are not in predefined order
    wrong_row_order = 0
    order_dict = { item: ind for ind, item in enumerate(gt_data[:, 0]) }
    order_index = [ v for v in order_dict.values() ]
    sort_index = [order_dict[item] for ind, item in enumerate(pr_data[:, 0]) if item in order_dict]
    if(sort_index != order_index):
        wrong_row_order = 1
        pr_data[order_index, :] = pr_data[sort_index, :]
        
    # If have missing results
    missing_results = 0
    if (gt_data.shape != pr_data.shape):
        missing_results = 1

    return pr_data, wrong_col_order, wrong_row_order, missing_results

# calculate kappa, F-1 socre and AUC value
def ODIR_Metrics(gt_data, pr_data):
    th = 0.5
    gt = gt_data.flatten()
    pr = pr_data.flatten()
    kappa = metrics.cohen_kappa_score(gt, pr > th)
    f1 = metrics.f1_score(gt, pr > th, average='micro')
    auc = metrics.roc_auc_score(gt, pr)
    final_score = (kappa + f1 + auc) / 3.0

    return kappa, f1, auc, final_score

class ODIR_Dataset:
    def __init__(self):
        # read file with labels from each eye
        csvfile = pd.read_csv(CSVFILE)
        labels_dict = pd.Series(csvfile.target.values, index=csvfile.filename).to_dict()

        # read files containing target for train and val patients
        df_train_gt = pd.read_csv(TRAIN_GT)
        df_val_gt = pd.read_csv(VAL_GT)

        train_set = set(df_train_gt['ID'].to_numpy())
        val_set = set(df_val_gt['ID'].to_numpy())

        self.X_train = np.zeros([len(train_set)*2], np.object)
        self.X_train_id = np.zeros([len(train_set)], np.object)
        self.y_train = np.zeros([len(train_set)*2, 8], np.int)
        self.X_val = np.zeros([len(val_set)*2], np.object)
        self.X_val_id = np.zeros([len(val_set)], np.object)

        df = pd.read_excel(XLSXFILE)

        i_train, i_test = 0, 0
        for index, row in df.iterrows():
            if row['ID'] in train_set:
                self.X_train[2*i_train] = row['Left-Fundus']
                self.y_train[2*i_train] = [int(i) for i in labels_dict[row['Left-Fundus']][1:-1].split(', ')]
                self.X_train[2*i_train + 1] = row['Right-Fundus']
                self.y_train[2*i_train + 1] = [int(i) for i in labels_dict[row['Right-Fundus']][1:-1].split(', ')]

                self.X_train_id[i_train] = row['ID']

                i_train += 1
            elif row['ID'] in val_set:
                self.X_val[2*i_test] = row['Left-Fundus']
                self.X_val[2*i_test + 1] = row['Right-Fundus']

                self.X_val_id[i_test] = row['ID']

                i_test += 1
        
        eye_train = pd.read_csv(EYE_TRAIN_GT)
        self.X_eye_train = eye_train['ID'].to_numpy()

        self.y_eye_train = np.zeros([len(self.X_eye_train), 8], dtype=np.int)
        for index, row in eye_train.iterrows():
            for (i,j) in (('N',0), ('D',1), ('G',2), ('C',3), ('A',4), ('H',5), ('M',6), ('O',7)):
                if row[i] == 1:
                    self.y_eye_train[index][j] = 1

        eye_val = pd.read_csv(EYE_VAL_GT)
        self.X_eye_val = eye_val['ID'].to_numpy()

    def evaluate(self, y_val_res, resultfile):
        if resultfile in (TRAIN_GT, VAL_GT, VAL_GT_XLSX):
            raise Exception('resultfile with same names as gt files')

        df_dict = dict()
        df_dict[GT_HEADER[0]] = self.X_val_id
        for i in range(1,len(GT_HEADER)):
            df_dict[GT_HEADER[i]] = y_val_res[:,i-1]

        df = pd.DataFrame(df_dict)

        #df.to_csv(resultfile, index=False)

        gt_data = import_gt(VAL_GT_XLSX)
        pr_data, wrong_col_order, wrong_row_order, missing_results = import_pred(gt_data, resultfile)

        if wrong_col_order:
            sys.exit(sys.argv[0], 'Error: Submission with disordered columns.')
            
        if wrong_row_order:
            sys.exit(sys.argv[0], 'Error: Submission with disordered rows.')
            
        if missing_results:
            sys.exit(sys.argv[0], 'Error: Incomplete submission with missing data.')
            
        kappa, f1, auc, final_score = ODIR_Metrics(gt_data[:, 1:], pr_data[:, 1:])
        print('kappa score:', kappa, 'f-1 score:', f1, 'AUC vlaue:', auc, 'Final Score:', final_score)
