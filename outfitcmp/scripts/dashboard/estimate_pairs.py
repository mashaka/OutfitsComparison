"""
Description: Count amount of people pairs that are estimated correctly by model
"""
import os
import yaml
import pickle
from tqdm import tqdm
from outfitcmp.scripts.dashboard.utils import load_predictions

WORKING_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(WORKING_DIR, '..', '..', '..')
CLUSTERS_FILE = os.path.join(ROOT_DIR, 'data', 'same_people_clusters.pickle')

TEST_PREDICTIONS_DIR = os.path.join(ROOT_DIR, 'trained_models', 'regression', '1 - Regression')
CONFIG_NAME = 'network_config.yaml'

def clusterise_predictions(y_true, y_pred, filenames, clusters):
    '''
    Find predictions corresponding to clusters
    '''
    predictions_dict = {}
    for i, filename in enumerate(filenames):
        short_filename = filename[(filename.rfind('\\') + 1):]
        predictions_dict[short_filename] = (y_true[i], y_pred[i])
    prediction_clusters = []
    exception_counter = 0
    for cluster in tqdm(clusters):
        prediction_cluster = []
        for filename in cluster:
            try:
                prediction_cluster.append(predictions_dict[filename])
            except KeyError:
                exception_counter += 1
        prediction_clusters.append(prediction_cluster)
    return prediction_clusters


def estimate_people_pairs(y_true, y_pred, filenames):
    ''' 
    Count amount of people pairs that are estimated correctly by model 
    Returns: {'total': int, 'correct': int}
    '''
    clusters = pickle.load(open(CLUSTERS_FILE, 'rb'))
    prediction_clusters = clusterise_predictions(y_true, y_pred, filenames, clusters)
    total = 0
    correct = 0
    for cluster in tqdm(prediction_clusters):
        for i in range(len(cluster)):
            for j in range(i+1, len(cluster)):
                total += 1
                if (cluster[i][0] < cluster[j][0]) == (cluster[i][1] < cluster[j][1]):
                    correct += 1
    return {'total': total, 'correct': correct}

def execute():
    """ Launch script """
    with open(os.path.join(TEST_PREDICTIONS_DIR, CONFIG_NAME), encoding='utf8') as yaml_file:
        config = yaml.load(yaml_file)
    results = load_predictions(TEST_PREDICTIONS_DIR, config)
    print(estimate_people_pairs(results['y_true'], results['y_pred'], results['filenames']))

if __name__ == '__main__':
    execute()