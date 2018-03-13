"""
Description: Utility functions
"""
import os
from keras.preprocessing.image import ImageDataGenerator

WORKING_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(WORKING_DIR, '..', '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

SUPPORTED_MODELS = ['Xception']

def get_image_size_for_model(config):
    """ Return size of an image accepted by given model """
    if config['model_name'] == 'Xception':
        return (299, 299)
    else:
        raise ValueError('{} model is not supported. Use one of these models: {}'.format(
            config['model_name'], SUPPORTED_MODELS))


def regression_flow_from_directory(flow_from_directory_gen, list_of_values):
    for x, y in flow_from_directory_gen:
        yield x, list_of_values[y]


def prepare_data_generator(config, split_name, needShuffle=True, isRegression=False):
    """ Create data generator for particular split """
    # Import correct model for preprocessing
    if config['model_name'] == 'Xception':
        from keras.applications.xception import preprocess_input
        print('Import modules for {} model'.format(config['model_name']))
    else:
        raise ValueError('{} model is not supported. Use one of these models: {}'.format(
            config['model_name'], SUPPORTED_MODELS))
    data_dir = os.path.join(DATA_DIR, config['data_dir'], split_name)
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    if isRegression:
        flow_from_directory_gen = datagen.flow_from_directory(
            data_dir,
            target_size=get_image_size_for_model(config),
            batch_size=config['batch_size'],
            shuffle=needShuffle,
            class_mode='sparse')
        list_of_values = map(int, os.listdir(data_dir))
        generator = regression_flow_from_directory(flow_from_directory_gen, list_of_values)
    else:
        generator = datagen.flow_from_directory(
            data_dir,
            target_size=get_image_size_for_model(config),
            batch_size=config['batch_size'],
            shuffle=needShuffle,
            class_mode='categorical')
    return generator