import pandas as pd
from sklearn.model_selection import train_test_split
import re
import os

def main():
    disease_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others']

    disease_labels = pd.read_csv('labels/disease_labels.txt', sep='\t', header=None)
    keyword_dict = dict([(row[0], row[1]) for index, row in disease_labels.iterrows()])

    book = pd.ExcelFile('data/labels/ODIR-5K_Training_Annotations(Updated)_V2.xlsx')
    df = book.parse(book.sheet_names[0], index_col=0)

    print('Age', df['Patient Age'].describe().to_dict())
    print('Sex', df.groupby('Patient Sex').size().to_dict())

    keyword_count = dict()
    for index, row in df.iterrows():
        for eye in ['Left', 'Right']:
            keywords = row['%s-Diagnostic Keywords' % (eye)]
            for keyword in re.split('，|,', keywords):
                keyword_count[keyword] = keyword_count.get(keyword, 0) + 1
    print(keyword_count, len(keyword_count))

    data = []
    for index, row in df.iterrows():
        for eye in ['Left', 'Right']:
            image = row['%s-Fundus' % (eye)]
            keywords = row['%s-Diagnostic Keywords' % (eye)]

            labels = [0] * 8
            for keyword in re.split('，|,', keywords):
                if keyword in keyword_dict.keys():
                    labels[keyword_dict[keyword]] = 1

            data.append([image] + labels)

    data = pd.DataFrame(data, columns=['ID', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'])

    train, test = train_test_split(data, train_size=0.9)

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
