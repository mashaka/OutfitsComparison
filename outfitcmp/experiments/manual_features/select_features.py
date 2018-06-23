"""
Description: Select best 100 HOG features
"""
import os
import pandas as pd
from tqdm import tqdm
from sklearn.feature_selection import SelectKBest, f_regression

WORKING_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(WORKING_DIR, '..', '..', '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
INPUT_DIR = os.path.join(ROOT_DIR, 'data', 'manual_features_fixed')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'data', 'manual_features_k_best')

NUMBER_OF_FEATURES = 100

def execute():
    """ Execute script """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    df = pd.read_csv(os.path.join(INPUT_DIR, 'train_0.csv'))
    labels = df['label']
    for name in tqdm(list(df)):
        if not name.startswith('hog'):
            df.drop(columns=[name], inplace=True)
    selector = SelectKBest(k=NUMBER_OF_FEATURES, score_func=f_regression).fit(df.values, labels.values)
    selected_features = selector.get_support(indices=True)
    checked_files = os.listdir(OUTPUT_DIR)
    for filename in os.listdir(INPUT_DIR):
        if filename in checked_files:
            continue
        print('Start working on {}'.format(filename))
        df = pd.read_csv(os.path.join(INPUT_DIR, filename))
        for name in tqdm(list(df)):
            if name.startswith('hog'):
                if int(name[4:]) not in selected_features:
                    df.drop(columns=[name], inplace=True)
        df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)    


if __name__ == '__main__':
    execute()