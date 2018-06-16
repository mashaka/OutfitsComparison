"""
Description: Utility functions
"""
import os
import numpy as np
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


def regression_flow_from_directory(flow_from_directory_gen, dict_of_values):
    for x, y in flow_from_directory_gen:
        y = np.array([dict_of_values[y_i] for y_i in y])
        yield x, y


class GeneratorLen(object):
    def __init__(self, gen, length, classes, class_indices, filenames):
        self.gen = gen
        self.length = length
        self.classes = classes
        self.class_indices = class_indices
        self.filenames = filenames

    def __len__(self): 
        return self.length

    def getGenerator(self):
        return self.gen


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
    if split_name == 'train':
        datagen = ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            preprocessing_function=preprocess_input) 
    else:
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    if isRegression:
        flow_from_directory_gen = datagen.flow_from_directory(
            data_dir,
            target_size=get_image_size_for_model(config),
            batch_size=config['batch_size'],
            shuffle=needShuffle,
            class_mode='sparse')
        dict_of_values = {value: int(key) for key, value in flow_from_directory_gen.class_indices.items()}
        generator = regression_flow_from_directory(flow_from_directory_gen, dict_of_values)
        generator = GeneratorLen(
            generator, 
            len(flow_from_directory_gen), 
            list(map(int, flow_from_directory_gen.classes)),
            flow_from_directory_gen.class_indices,
            flow_from_directory_gen.filenames
        )
    else:
        generator = datagen.flow_from_directory(
            data_dir,
            target_size=get_image_size_for_model(config),
            batch_size=config['batch_size'],
            shuffle=needShuffle,
            class_mode='categorical')
        generator = GeneratorLen(
            generator, 
            len(generator), 
            list(map(int, generator.classes)),
            generator.class_indices,
            generator.filenames
        )
    return generator