"""
Description: Select best 100 HOG features
"""
import os
import pandas as pd
from tqdm import tqdm

WORKING_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(WORKING_DIR, '..', '..', '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
INPUT_DIR = os.path.join(ROOT_DIR, 'data', 'manual_features_k_best')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'data', 'manual_features_final')

def execute():
    """ Execute script """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    df_test = pd.read_csv(os.path.join(INPUT_DIR, 'test_0.csv'))
    df_train = pd.read_csv(os.path.join(INPUT_DIR, 'train_0.csv'))
    for filename in os.listdir(INPUT_DIR):
        if filename in ['test_0.csv', 'train_0.csv']:
            continue
        df = pd.read_csv(os.path.join(INPUT_DIR, filename))
        if filename.startswith('test'):
            df_test = pd.concat([df_test, df])
        else:
            df_train = pd.concat([df_train, df])
    df_test.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)
    df_train.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
        

if __name__ == '__main__':
    execute()