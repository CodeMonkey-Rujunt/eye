import pandas as pd
import glob
import os 

def create_csv(split):
    ROOT_PATH = '/odir5k/ocular-dataset/ODIR-5K-Flow/fake-{split}/*'

    g = glob.glob(ROOT_PATH)
    records = []

    for folder in g:
        label = int(folder.split('/')[-1][0])
        print(label, folder)

        files = os.listdir(folder)
        for file in files:
            records.append([os.path.join(folder, file), label])
    
    assert len(records) > 0, 'There is no record.'

    df = pd.DataFrame(records, columns=['path', 'label'])
    print(df)

    filename = 'pytorch_fake{split}.csv'
    #df.to_csv(filename, index=False)

if __name__ == '__main__':
    create_csv('train')
    create_csv('test')
