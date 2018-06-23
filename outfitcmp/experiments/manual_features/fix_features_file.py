"""
Description: Fix incorrect format of features file
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from outfitcmp.experiments.manual_features.features_generation import get_header

WORKING_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(WORKING_DIR, '..', '..', '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
INPUT_DIR = os.path.join(ROOT_DIR, 'data', 'manual_features')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'data', 'manual_features_fixed')

FILENAMES = ['train'] #, 'test']

def fix(value):  
    value = value[1:-1]
    value = float(value)
    return value

def execute():
    """ Execute script  """
    header = get_header()
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for filename in FILENAMES:
        print('Reading {}'.format(filename))
        df = pd.read_csv(os.path.join(INPUT_DIR, filename + '.csv'), skiprows=80000, nrows=10000)
        for name in tqdm(list(df)):
            if name.startswith('hog'):
                df[name] = df[name].apply(fix)
        df.to_csv(os.path.join(OUTPUT_DIR, filename + '_8.csv'), index=False, header=header)


if __name__ == '__main__':
    execute()