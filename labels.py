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

    book = pd.ExcelFile('data/labels/ODIR-5K_Training_Annotations(Updated)_V2.xlsx')
    df = book.parse(book.sheet_names[0], index_col=0)

    print(df)

    keyword_dict = dict()
    for keywords in df['Left-Diagnostic Keywords'].tolist() + df['Right-Diagnostic Keywords'].tolist():
        for keyword in keywords.split('，'):
            keyword_dict[keyword] = keyword_dict.get(keyword, 0) + 1 

    keywords = sorted(keyword_dict.items(), key=lambda x: x[1], reverse=True)
    print('\nThere are %d keywords.' % (len(keywords)))
    print(pd.DataFrame(keywords, columns=['keyword', 'count']))

    print('\nAge stats:', df['Patient Age'].describe().to_dict())
    print('Sex stats:', df.groupby('Patient Sex').size().to_dict())

if __name__ == '__main__':
    main()
