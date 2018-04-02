'''
Description: Convert weight.h5 and model.h5 to one model.h5 file
'''
import os
import yaml
from keras.models import model_from_json
from outfitcmp.scripts.predict_using_model import load_model

WORKING_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(WORKING_DIR, '..', '..')
EXPERIMENT_DIR = os.path.join(ROOT_DIR, 'trained_models', 'regression', '2 - More custom layers')
CONFIG_NAME = 'network_config.yaml'

OUTPUT_FILE = os.path.join(EXPERIMENT_DIR, 'model_new.h5')

def convert(experiment_dir):
    ''' Convert weight.h5 and model.h5 to one model.h5 file '''
    with open(os.path.join(experiment_dir, CONFIG_NAME), encoding='utf8') as yaml_file:
        config = yaml.load(yaml_file)
    model = load_model(experiment_dir, config)
    model.save(OUTPUT_FILE)

def execute():
    """ Execute script """
    convert(EXPERIMENT_DIR)

if __name__ == "__main__":
    execute()
    