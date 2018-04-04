"""
Description: Predict using trained model
"""
import os
import yaml
import numpy as np
from keras.models import model_from_json

from outfitcmp.scripts.utils import prepare_data_generator

WORKING_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(WORKING_DIR, '..', '..')
EXPERIMENT_DIR = os.path.join(ROOT_DIR, 'trained_models', 'regression', '2 - More custom layers')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

CONFIG_NAME = 'network_config.yaml'

def load_model(experiment_dir, config):
    """
    Load serialized model
    """
    loaded_model.load(os.path.join(experiment_dir, config['model_file']))
    loaded_model.compile(
        loss=config['loss'],
        optimizer=config['optimizer'],
        metrics=config['metrics']
    )
    return loaded_model

def convert_to_original_values(values, generator):
    """ Convert labels changed by flow_from_directory to original one """
    dict_of_values = {value: int(key) for key, value in generator.class_indices.items()}
    return np.array([dict_of_values[y_i] for y_i in values])

def predict_using_model(experiment_dir, config, model, _isRegression=False):
    """ Predict using a trained model save predictions along with true values """
    test_generator = prepare_data_generator(config, 'test', needShuffle=False, isRegression=_isRegression)
    predicted = model.predict_generator(
        generator=test_generator.getGenerator(),
        steps=len(test_generator)
    )
    y_true = test_generator.classes
    y_true = convert_to_original_values(y_true, test_generator)
    if not config['is_regression']:
        predicted = predicted.argmax(axis=-1)
        predicted = predicted.flatten()
        predicted = convert_to_original_values(predicted, test_generator)
    else:
        predicted = predicted.flatten()
    np.savez(
        os.path.join(experiment_dir, config['predicted_file']),
        pred=predicted,
        y_true=y_true,
        filenames=test_generator.filenames
    )
    print('Saved predictions')
    # Estimate
    if not _isRegression:
        loss, metric = model.evaluate_generator(
            generator=test_generator.getGenerator(),
            steps=len(test_generator)
        )
    else:
        loss, mse, mae = model.evaluate_generator(
            generator=test_generator.getGenerator(),
            steps=len(test_generator)
        )
    if not _isRegression:
        print('\nTesting loss: {}, acc: {}\n'.format(loss, metric))
    else:
        print('\nTesting loss: {}, mse: {}, mae: {}\n'.format(loss, mse, mae))

def predict_using_saved_model(experiment_dir):
    """ Predict using a trained model in a given folder and save predictions along with true values """
    with open(os.path.join(experiment_dir, CONFIG_NAME), encoding='utf8') as yaml_file:
        config = yaml.load(yaml_file)
    model = load_model(experiment_dir, config)
    print("Loaded model")
    predict_using_model(experiment_dir, config, model, config['is_regression'])

def execute():
    """ Execute script """
    predict_using_saved_model(EXPERIMENT_DIR)

if __name__ == "__main__":
    execute()
    