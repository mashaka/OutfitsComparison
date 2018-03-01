"""
Author: Maria Sandrikova
Description: Estimate trained model
"""
import os
import yaml
import numpy as np
from numpy import argmax
from keras.models import model_from_json

from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

WORKING_DIR = os.path.dirname(__file__)
EXPERIMENT_DIR = os.path.join(WORKING_DIR, '..', 'experiments', 'baseline', 'results', 'Xception')
DATA_DIR = os.path.join(WORKING_DIR, '..', 'data')

CONFIG_NAME = 'network_config.yaml'
CONFIG_FILE = os.path.join(EXPERIMENT_DIR, CONFIG_NAME)

# Load params from config
with open(CONFIG_FILE, encoding='utf8') as yaml_file:
    config = yaml.load(yaml_file)

def load_dataset():
    """ Load one dataset """
    data_dir = os.path.join(DATA_DIR, config['data_dir'], config['model_name'])
    dataset_npz = np.load(os.path.join(data_dir, 'test.npz'))
    # Convert them to real dicts to have possibility to add new keys in it in future
    dataset = {'x': dataset_npz['x'], 'y': dataset_npz['y']}
    # One-hot-encoding
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(dataset['y'])
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    dataset['one_hot_encoded'] = onehot_encoder.fit_transform(integer_encoded)
    return dataset, label_encoder

def load_model():
    """
    Load serialized model
    """
    with open(os.path.join(EXPERIMENT_DIR, config['model_file'])) as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    # Load model's weights
    loaded_model.load_weights(os.path.join(EXPERIMENT_DIR, config['weights_file']))
    return loaded_model

def invert_one_hot_encoding(predicted, label_encoder):
    """ Reverse one hot encoding """
    res = []
    for item in predicted:
        res.append(label_encoder.inverse_transform([argmax(item)]))
    return res

def predict():
    """
    Estimate model
    """
    # Load dataset and one-hot-encode it
    test, label_encoder = load_dataset()
    print('Loaded model')
    model = load_model()

    predicted = model.predict(test['x'])
    print('Predicted')
    predicted_labels = invert_one_hot_encoding(predicted, label_encoder)
    accuracy = accuracy_score(test['y'], predicted_labels)
    recall = recall_score(test['y'], predicted_labels, average='micro')
    precision = precision_score(test['y'], predicted_labels, average='micro')
    print('Accuracy: {}, Recall: {}, Precision: {}'.format(accuracy, recall, precision))

def execute():
    """ Launch script """
    predict()

if __name__ == '__main__':
    execute()