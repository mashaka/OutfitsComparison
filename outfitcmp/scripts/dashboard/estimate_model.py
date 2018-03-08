"""
Description: Estimate trained model
"""
import os
import yaml
import numpy as np

WORKING_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(WORKING_DIR, '..', '..', '..')
EXPERIMENT_DIR = os.path.join(ROOT_DIR, 'trained_models', 'baseline', 'Xception_3')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

CONFIG_NAME = 'network_config.yaml'

def load_predictions(experiment_dir, config):
    """ Load predictions """
    predictions_npz = np.load(os.path.join(experiment_dir, config['predicted_file']))
    return predictions_npz['pred']

def estimate(experiment_dir):
    """
    Estimate model
    """
    with open(os.path.join(experiment_dir, CONFIG_NAME), encoding='utf8') as yaml_file:
        config = yaml.load(yaml_file)
    predictions = load_predictions(experiment_dir, config)
    print('Loaded predictions')

def execute():
    """ Launch script """
    predict(EXPERIMENT_DIR)

if __name__ == '__main__':
    execute()