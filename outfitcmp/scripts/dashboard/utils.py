'''
Description: Utils for dashboard scripts
'''
import os
import numpy as np

def class_from_regression(y_pred):
    """ Classification accuracy allowing error in gap classes """
    y_pred_class = []
    for y in y_pred:
        y_pred_class.append(int(round(min(max(1, y), 10))))
    return np.array(y_pred_class)

def load_predictions(experiment_dir, config):
    """ Load predictions """
    predictions_npz = np.load(os.path.join(experiment_dir, config['predicted_file']))
    y_true = predictions_npz['y_true']
    y_pred = predictions_npz['pred']
    y_pred_class = y_pred
    if config['is_regression']:
        y_pred = y_pred.flatten()
        y_pred_class = class_from_regression(y_pred)
    filenames = predictions_npz['filenames'] if 'filenames' in predictions_npz else []
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_class': y_pred_class, 
        'filenames': filenames
    }
        
    