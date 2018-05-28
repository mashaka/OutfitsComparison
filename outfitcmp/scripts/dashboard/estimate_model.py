"""
Description: Estimate trained model
"""
import os
import yaml
import numpy as np
from scipy import stats

from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error, mean_squared_error

from outfitcmp.scripts.dashboard.estimate_pairs import estimate_people_pairs
from outfitcmp.scripts.dashboard.utils import load_predictions

WORKING_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(WORKING_DIR, '..', '..', '..')
EXPERIMENT_DIR = os.path.join(ROOT_DIR, 'trained_models', 'baseline', 'Xception_3')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

CONFIG_NAME = 'network_config.yaml'

def accuracy_with_gap(y_true, y_pred, gap):
    """ Classification accuracy allowing error in gap classes """
    true_predictions = 0
    for i in range(len(y_pred)):
        if abs(y_pred[i] - y_true[i]) <= gap:
            true_predictions += 1
    return true_predictions/len(y_true)

def estimate(experiment_dir, config=None):
    """
    Estimate model
    """
    if config is None:
        with open(os.path.join(experiment_dir, CONFIG_NAME), encoding='utf8') as yaml_file:
            config = yaml.load(yaml_file)
    results = load_predictions(experiment_dir, config)
    print('Loaded predictions for {}'.format(config['experiment_name']))
    pairs = estimate_people_pairs(results['y_true'], results['y_pred'], results['filenames'])
    results = {
        "precision": precision_score(results['y_true'], results['y_pred_class'], average='macro'),
        "recall": recall_score(results['y_true'], results['y_pred_class'], average='macro'),
        "acc_0": accuracy_score(results['y_true'], results['y_pred_class']),
        "acc_1": accuracy_with_gap(results['y_true'], results['y_pred_class'], 1),
        "acc_2": accuracy_with_gap(results['y_true'], results['y_pred_class'], 2),
        "MAE": mean_absolute_error(results['y_true'], results['y_pred']),
        "MSE": mean_squared_error(results['y_true'], results['y_pred']),
        "pairs": 0 if pairs['total'] == 0 else pairs['correct'] / pairs['total']
    }
    for key, value in results.items():
        if isinstance(value, float):
            results[key] = round(value, 5)
    return results

def execute():
    """ Launch script """
    print(estimate(EXPERIMENT_DIR))

if __name__ == '__main__':
    execute()