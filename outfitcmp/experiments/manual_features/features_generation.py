"""
Description: Manual generation of features using segmented photo of outfit
"""
import os
import numpy as np
import csv
import cv2
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
SEG_IMG_SIZE = (400, 600)
HOG_FEATURES_COUNT = 3780
HOG_IMG_SIZE = (64, 128)

HOG_DESCRIPTOR = cv2.HOGDescriptor()

def get_header():
    """ Get header for for features """
    header = ['label', 'image_name']
    for t in range(SEG_TYPES_COUNT):
        header.append('seg_type_{}'.format(t))
    for t in range(SEG_TYPES_COUNT):
        header.append('mean_B_{}'.format(t))
        header.append('mean_G_{}'.format(t))
        header.append('mean_R_{}'.format(t))
        header.append('mean_L_{}'.format(t))
        header.append('mean_A_{}'.format(t))
        header.append('mean_B_{}'.format(t))
    for t in range(HOG_FEATURES_COUNT):
        header.append('hog_{}'.format(t))
    return header

def generate_features(image_name, seg_info, label):
    """ Generate features for one photo """
    image_path = os.path.join(IMAGES_DIR, image_name)
    features = [label, image_name]
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, SEG_IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    for t in range(SEG_TYPES_COUNT):
        features.append((seg_info == t).sum())
    means_bgr = [(0,0,0) for _ in range(SEG_TYPES_COUNT)]
    means_lab = [(0,0,0) for _ in range(SEG_TYPES_COUNT)]
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            means_bgr[seg_info[y][x]] += img[y][x]
            means_lab[seg_info[y][x]] += img_lab[y][x]
    for t in range(SEG_TYPES_COUNT):
        for i in range(3):
            features.append(means_bgr[t][i]/SEG_IMG_SIZE[0]/SEG_IMG_SIZE[1])
        for i in range(3):
            features.append(means_lab[t][i]/SEG_IMG_SIZE[0]/SEG_IMG_SIZE[1])
    img_hog = cv2.resize(img, HOG_IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    hog_features = HOG_DESCRIPTOR.compute(img_hog)
    features.extend(list(hog_features))
    return features

def generate_features_group(name, ids, ids_dict, data_desc, seg_prob):
    """ Generate features for one group in SPLITS_NAMES """
    features = [get_header()]
    cannot_read = 0
    for id in tqdm(ids):
        image_name = data_desc.at[ids_dict[id], 'image_name']
        label = data_desc.at[ids_dict[id], 'label']
        img_features = generate_features(image_name, seg_prob[ids_dict[id]], label)
        if img_features != None:
            features.append(img_features)
        else:
            cannot_read += 1
    print('Can\'t open {} for {}'.format(cannot_read, name))
    
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