import pandas as pd
import numpy as np
import splitfolders


def annotation_to_npy(csvfile, root):
    df = pd.read_csv(csvfile, header=None)
    df=df.iloc[:,[1,2,3]]
    df['coord'] = df[[1,2]].values.tolist()
    df = df.groupby(3)['coord'].apply(list).reset_index(name='pts')
    for idx, row in df.iterrows():
        filename = root + str(int(row[3].split('.')[0]) + 2000) + '.npy'
        arr = np.array(row['pts'])
        with open(filename, 'wb') as f:
            np.save(f, arr)


def split_data(input_folder, output_folder, splitratio):
    splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=splitratio, group_prefix=None, move=True)