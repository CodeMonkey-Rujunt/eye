import pandas as pd

def main():
    # Normal 0.8127 0.8465 0.8618 0.8699
    # Diabetes 0.8309 0.8505 0.8668 0.8735
    # Glaucoma 0.9776 0.9791 0.9831 0.9874
    # Cataract 0.9854 0.9863 0.9888 0.9906
    # AMD 0.9603 0.9731 0.9780 0.9826
    # Hypertension 0.9637 0.9751 0.9746 0.9788
    # Myopia 0.9923 0.9946 0.9938 0.9942
    # Others

    df = pd.read_csv('labels/eye_labels_train.csv', sep=',')
    print(df)
    print(df.loc[:, 'N':'O'].sum(axis=0))

    book = pd.ExcelFile('data/labels/ODIR-5K_Training_Annotations(Updated)_V2.xlsx')
    df = book.parse(book.sheet_names[0], index_col=0)
    print(df)
    print(df.loc[:, 'N':'O'].sum(axis=0))

    print('Age stats:', df['Patient Age'].describe().to_dict())
    print('Sex stats:', df.groupby('Patient Sex').size().to_dict())

    keyword_dict = dict()
    for index, row in df.iterrows():
        for keyword in row['Left-Diagnostic Keywords'].split('，'):
            for label, value in zip(df.columns[6:], row['N':]):
                if value:
                    #print(keyword, label, value)
                    pass
            #keyword_dict[keyword] = keyword_dict.get(keyword, 0) + 1 

        '''
        for keyword in row['Right-Diagnostic Keywords'].split('，'):
            print(keyword, row['N':].tolist())
            #keyword_dict[keyword] = keyword_dict.get(keyword, 0) + 1 
        '''

    '''
    keywords = sorted(keyword_dict.items(), key=lambda x: x[1], reverse=True)
    print('There are %d keywords.' % (len(keywords)))
    print(pd.DataFrame(keywords, columns=['keyword', 'count']).head(20))
    '''

if __name__ == '__main__':
    main()
