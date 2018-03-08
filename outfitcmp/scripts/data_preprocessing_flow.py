"""
Description: Preprocess dataset to use it with keras flow_from_directory()
"""
import os
import yaml
import shutil
import pandas as pd
from tqdm import tqdm

WORKING_DIR = os.path.dirname(__file__)
CONFIG_NAME = 'data_preprocessing.yaml'
CONFIG_FILE = os.path.join(WORKING_DIR, CONFIG_NAME)

DATA_DIR = os.path.join(WORKING_DIR, '..', 'data')

FLOW_SUFFIX = '_flow'

SUPPORTED_MODELS = ['Xception']

SPLITS_NAMES = ['test', 'train', 'validation']

# Load params from config
with open(CONFIG_FILE, encoding='utf8') as yaml_file:
    config = yaml.load(yaml_file)

def check_config():
    """ Sanity checks for config """
    if os.path.exists(os.path.join(DATA_DIR, config['root_dir'] + FLOW_SUFFIX)):
        raise ValueError(
            'Output directory for {} already exists inside data folder'.format(
                config['root_dir'] + FLOW_SUFFIX
            ))
    if not os.path.exists(os.path.join(DATA_DIR, config['root_dir'])):
        raise ValueError(
            'Input directory {} does not exist'.format(
                config['root_dir']
            ))

def load_data_desc():
    """ Load data description CSV in pandas DataFrame"""
    data = pd.read_csv(os.path.join(DATA_DIR, config['root_dir'], config['labels_file']))
    return data

def read_splits():
    """ Read list of ids defining data splits """
    splits_dict = dict()
    for name in SPLITS_NAMES:
        with open(os.path.join(DATA_DIR, config['root_dir'], config[name + '_ids'])) as ids_file:
            ids = ids_file.readlines()[1:]
            ids = list(map(int, ids))
            splits_dict[name] = ids
    return splits_dict

def init_one_split(name, ids, data_desc):
    """ Create subdirectory for particular split"""
    print("Start processing {}".format(name))
    data_dir = os.path.join(DATA_DIR, config['root_dir'], config['images_dir'])
    split_dir = os.path.join(DATA_DIR, config['root_dir'] + FLOW_SUFFIX, name)
    os.makedirs(split_dir)
    labels = data_desc['label'].unique().tolist()
    labels = list(map(str, labels))
    for label in labels:
        os.makedirs(os.path.join(split_dir, label))
    ids_dict = {id: i for i, id in enumerate(data_desc['id'].tolist())}
    for id in tqdm(ids):
        image_name = data_desc.at[ids_dict[id], 'image_name']
        label = str(data_desc.at[ids_dict[id], 'label'])
        shutil.copy(
            os.path.join(data_dir, image_name),
            os.path.join(split_dir, label, image_name)
        )

def init_folder():
    """ Create and initialize new data directory """
    output_dir = os.path.join(DATA_DIR, config['root_dir'] + FLOW_SUFFIX)
    os.makedirs(output_dir)

    data_desc = load_data_desc()
    splits_dict = read_splits()

    for name in splits_dict:
        init_one_split(name, splits_dict[name], data_desc)

def execute():
    """ Launch script """
    check_config()
    init_folder()

if __name__ == "__main__":
    execute()
