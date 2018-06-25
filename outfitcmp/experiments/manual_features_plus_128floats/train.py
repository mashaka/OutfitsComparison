"""
Description: Train using manualy generated features
"""
import os
import yaml
import pandas as pd
import numpy as np
import shutil
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

WORKING_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(WORKING_DIR, '..', '..', '..')
FEATURES_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'trained_models', 'manual_features_plus_128floats')

LABEL_COLUMN = 'label'
IMAGE_PATH_COLUMN = 'image_name'
CONFIG_NAME = 'experiment_config.yaml'
CONFIG_FILE = os.path.join(WORKING_DIR, CONFIG_NAME)

def read_features(features_type):
    """ Read features from csv file """
    features = pd.read_csv(os.path.join(FEATURES_DIR, features_type + '.csv'))
    labels = features[LABEL_COLUMN].values
    filenames = features[IMAGE_PATH_COLUMN].values
    features.drop([LABEL_COLUMN, IMAGE_PATH_COLUMN], 1, inplace=True)
    return features.values, labels, filenames

def init_experiment_folder(config):
    """ Create and initialize directory for experiment """
    experiment_dir = os.path.join(RESULTS_DIR, config['experiment_name'])
    os.makedirs(experiment_dir)
    # Copy an experiment config for logs and possibility to reproduce result in future
    shutil.copy(CONFIG_FILE, os.path.join(experiment_dir, CONFIG_NAME))

def train(config):
    """ Train model using manualy generated features """
    experiment_dir = os.path.join(RESULTS_DIR, config['experiment_name'])
    print('Reading train features')
    X_train, y_train, _ = read_features('train')
    clf = DecisionTreeRegressor()
    print('Training')
    clf.fit(X_train, y_train)
    print('Reading test features')
    X_test, y_test, test_filenames = read_features('test')
    print('Predicting')
    y_pred = clf.predict(X_test)
    print('Saving')
    np.savez(
        os.path.join(experiment_dir, config['predicted_file']),
        pred=y_pred,
        y_true=y_test,
        filenames=test_filenames
    )

def execute():
    """ Execute script that tests features generation on one photo """
    # Load params from config
    with open(CONFIG_FILE, encoding='utf8') as yaml_file:
        config = yaml.load(yaml_file)
    init_experiment_folder(config)
    train(config)

if __name__ == '__main__':
    execute()
