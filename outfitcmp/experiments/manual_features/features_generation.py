"""
Description: Manual generation of features using segmented photo of outfit
"""
import os
import numpy as np
import csv
import h5py
from tqdm import tqdm

from outfitcmp.scripts.data_preprocessing_flow import read_splits, load_data_desc, SPLITS_NAMES

WORKING_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(WORKING_DIR, '..', '..', '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'data', 'manual_features')

SEGMENTATION_FILE = os.path.join(DATA_DIR, 'out_seg.h5')
IMAGES_DIR = os.path.join(DATA_DIR, '144k', 'images')

SEG_TYPES_COUNT = 25

def get_header():
    """ Get header for for features """
    return ['label', 'image_name'] + ['seg_type_{}'.format(t) for t in range(SEG_TYPES_COUNT)]

def generate_features(image_name, seg_info, label):
    """ Generate features for one photo """
    image_path = os.path.join(IMAGES_DIR, image_name)
    features = [label, image_name]
    for t in range(SEG_TYPES_COUNT):
        features.append((seg_info == t).sum())
    return features

def generate_features_group(name, ids, ids_dict, data_desc, seg_prob):
    """ Generate features for one group in SPLITS_NAMES """
    features = [get_header()]
    # i = 0
    for id in tqdm(ids):
        # i += 1
        # if i > 1000: 
        #     break
        image_name = data_desc.at[ids_dict[id], 'image_name']
        label = data_desc.at[ids_dict[id], 'label']
        features.append(generate_features(image_name, seg_prob[ids_dict[id]], label))
    
    with open(os.path.join(OUTPUT_DIR, name + '.csv'), 'w', encoding='utf8') as output_file:
        writer = csv.writer(output_file, lineterminator='\n')
        writer.writerows(features)

def execute():
    """ Execute script that tests features generation on one photo """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    with h5py.File(SEGMENTATION_FILE, 'r') as seg_file:
        seg_prob = seg_file['seg_prob']
        data_desc = load_data_desc()
        splits_dict = read_splits()
        ids_dict = {id: i for i, id in enumerate(data_desc['id'].tolist())}
        for name in ['test', 'train']:
            generate_features_group(name, splits_dict[name], ids_dict, data_desc, seg_prob)


if __name__ == '__main__':
    execute()