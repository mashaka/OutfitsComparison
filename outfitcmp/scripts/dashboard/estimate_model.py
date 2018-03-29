"""
Description: Estimate trained model
"""
import os
import yaml
import numpy as np
from scipy import stats

from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error, mean_squared_error

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

def class_from_regression(y_pred):
    """ Classification accuracy allowing error in gap classes """
    y_pred_class = []
    for y in y_pred:
        y_pred_class.append(int(round(min(max(1, y), 10))))
    return np.array(y_pred_class)

def estimate(experiment_dir):
    """
    Estimate model
    """
    with open(os.path.join(experiment_dir, CONFIG_NAME), encoding='utf8') as yaml_file:
        config = yaml.load(yaml_file)
    pred, y_true = load_predictions(experiment_dir, config)
    y_pred = pred.argmax(axis=-1)
    print('Loaded predictions for {}'.format(config['experiment_name']))
    y_pred_class = y_pred
    if config['is_regression']:
        y_pred_class = class_from_regression(y_pred)
        print(y_pred_class.shape, y_pred_class[:20])
        print(y_pred.shape, y_pred[:20])
        print(y_true.shape, y_pred[:20])
        print(stats.describe(y_pred_class))
    results = {
        "precision": precision_score(y_true, y_pred_class, average='macro'),
        "recall": recall_score(y_true, y_pred_class, average='macro'),
        "acc_0": accuracy_score(y_true, y_pred_class),
        "acc_1": accuracy_with_gap(y_true, y_pred_class, 1),
        "acc_2": accuracy_with_gap(y_true, y_pred_class, 2),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
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