"""
Description: Estimate trained model
"""
import os
import yaml
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score

WORKING_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(WORKING_DIR, '..', '..', '..')
EXPERIMENT_DIR = os.path.join(ROOT_DIR, 'trained_models', 'baseline', 'Xception_3')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

CONFIG_NAME = 'network_config.yaml'

def load_predictions(experiment_dir, config):
    """ Load predictions """
    predictions_npz = np.load(os.path.join(experiment_dir, config['predicted_file']))
    return predictions_npz['pred'], predictions_npz['y_true']

def accuracy_with_gap(y_true, y_pred, gap):
    """ Classification accuracy allowing error in gap classes """
    true_predictions = 0
    for i in range(len(y_pred)):
        if abs(y_pred[i] - y_true[i]) <= gap:
             true_predictions += 1
    return true_predictions/len(y_true)

def estimate(experiment_dir):
    """
    Estimate model
    """
    with open(os.path.join(experiment_dir, CONFIG_NAME), encoding='utf8') as yaml_file:
        config = yaml.load(yaml_file)
    pred, y_true = load_predictions(experiment_dir, config)
    y_pred = pred.argmax(axis=-1)
    print('Loaded predictions')
    results = {
        "precision": precision_score(y_true, y_pred, average='macro'),
        "recall": recall_score(y_true, y_pred, average='macro'),
        "acc_0": accuracy_score(y_true, y_pred),
        "acc_1": accuracy_with_gap(y_true, y_pred, 1),
        "acc_2": accuracy_with_gap(y_true, y_pred, 2),
        "MAE": "TODO",
        "pairs": "TODO"
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