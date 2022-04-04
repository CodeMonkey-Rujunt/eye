import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    disease_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others']

    book = pd.ExcelFile('labels/ODIR-5K_Training_Annotations(Updated)_V2.xlsx')
    df = book.parse(book.sheet_names[0], index_col=0)

    for eye in ['Left', 'Right']:
        df['%s-Diagnostic Keywords' % (eye)] = df['%s-Diagnostic Keywords' % (eye)].replace('，', ',')

    print('Age', df['Patient Age'].describe().to_dict())
    print('Sex', df.groupby('Patient Sex').size().to_dict())

    df = pd.concat([df.loc[:, ['Left-Fundus', 'Right-Fundus']], df.loc[:, 'N':'O']], axis=1)

    train, test = train_test_split(df, train_size=0.9)

    train.to_csv('labels/train.csv', sep=',', index=False)
    test.to_csv('labels/test.csv', sep=',', index=False)

    stats = pd.concat([
        train.loc[:, 'N':'O'].sum(axis=0),
        test.loc[:, 'N':'O'].sum(axis=0),
        ], axis=1)
    stats.index = disease_names
    stats.columns = ['train', 'test']
    stats['train+test'] = stats['train'] + stats['test']
    stats['%'] = stats['train+test'] / stats['train+test'].sum(axis=0) * 100
    stats.loc['Total', :] = stats.sum(axis=0)

    print(stats.round(1))

if __name__ == '__main__':
    main()
