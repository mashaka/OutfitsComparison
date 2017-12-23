"""
Author: Maria Sandrikova
Description: Script for data preprocessing for preprocessing 
    for particular pretrained model on which network will be based

Data expected to be presented in following way:
    images/ - folder with images
    labels.csv - map images to ids and their labels,
        has 'id', 'label' and 'image_name' columns
    testids.csv/trainids.csv/validids.csv - list of test/train/validation set ids
        (from 'id' column in labels.csv), has 'testids' header

Supported models:
    - Xception
"""
import os
import yaml
import csv
import pandas as pd
from tqdm import tqdm
import numpy as np

from keras.preprocessing import image

WORKING_DIR = os.path.dirname(__file__)
CONFIG_NAME = 'data_preprocessing.yaml'
CONFIG_FILE = os.path.join(WORKING_DIR, CONFIG_NAME)

DATA_DIR = os.path.join(WORKING_DIR, '..', 'data')

SUPPORTED_MODELS = ['Xception']

# Load params from config
with open(CONFIG_FILE, encoding='utf8') as yaml_file:
    config = yaml.load(yaml_file)

# Import correct model for preprocessing
if config['model_name'] == 'Xception':
    from keras.applications.xception import preprocess_input
    print('Import modules for {} model'.format(config['model_name']))
else:
    raise ValueError('{} model is not supported. Use one of these models: {}'.format(
        config['model_name'], SUPPORTED_MODELS))

def check_config():
    """ Sanity checks for config """
    if os.path.exists(os.path.join(DATA_DIR, config['root_dir'], config['model_name'])):
        raise ValueError(
            'Output directory for {} model already exists inside {} data folder'.format(
                config['model_name'],
                config['root_dir']
            ))

def load_data():
    """ Load data description CSV in pandas DataFrame"""
    data = pd.read_csv(os.path.join(DATA_DIR, config['root_dir'], config['labels_file']))
    return data

def get_image_size_for_model(model_name):
    """ Return size of an image accepted by given model """
    if model_name == 'Xception':
        return (299, 299)
    else:
        raise ValueError('{} model is not supported. Use one of these models: {}'.format(
            model_name, SUPPORTED_MODELS))

def preprocess_data(data):
    """ Preprocess data for particular pretrained model """
    images_dir = os.path.join(DATA_DIR, config['root_dir'], config['images_dir'])
    features = []
    print('Start processing data')
    for image_name in tqdm(data['image_name'].tolist()):
        image_path = os.path.join(images_dir, image_name)
        image_size = get_image_size_for_model(config['model_name'])
        img = image.load_img(image_path, target_size=image_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features.append(x[0])
    return features

def split_data(features, data):
    """ Split data to test, train and validation sets """
    ids_dict = {id: i for i, id in enumerate(data['id'].tolist())}
    test, validation, train = ({'x': [], 'y': []} for _ in range(3))
    sets_dict = {
        'test': test,
        'validation': validation,
        'train': train
    }
    for name in sets_dict:
        dataset = sets_dict[name]
        with open(os.path.join(DATA_DIR, config['root_dir'], config[name + '_ids'])) as ids_file:
            ids = ids_file.readlines()[1:]
            ids = list(map(int, ids))
        for id in ids:
            dataset['y'].append(data.at[ids_dict[id], 'label'])
            dataset['x'].append(features[ids_dict[id]])
    for name in sets_dict:
        dataset = sets_dict[name]
        print('Length of a {} set is {}'.format(name, len(dataset['y'])))
    return test, train, validation

def save_one_set(filename, data):
    """ Save one set of features to NPZ archive """
    np.savez(filename, y=data['y'], x=data['x'])

def save_data(test, train, validation):
    """ Save data to NPZ archives """
    output_dir = os.path.join(DATA_DIR, config['root_dir'], config['model_name'])
    os.makedirs(output_dir)
    save_one_set(os.path.join(output_dir, 'train.npz'), train)
    save_one_set(os.path.join(output_dir, 'test.npz'), test)
    save_one_set(os.path.join(output_dir, 'validation.npz'), validation)

def execute():
    """ Execute script """
    check_config()
    data = load_data()
    features = preprocess_data(data)
    test, train, validation = split_data(features, data)
    save_data(test, train, validation)


if __name__ == '__main__':
    execute()
