"""
Description: Train baseline model adding several Dense layers to 
    pretrained model, like Xception
"""
import os
import time
import yaml
import shutil
import numpy as np
from numpy import argmax

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

WORKING_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(WORKING_DIR, '..', '..', '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'trained_models', 'regression')
CHECKPOINTS_DIR_NAME = 'checkpoints'

CONFIG_NAME = 'network_config.yaml'
CONFIG_FILE = os.path.join(WORKING_DIR, CONFIG_NAME)

# Load params from config
with open(CONFIG_FILE, encoding='utf8') as yaml_file:
    config = yaml.load(yaml_file)

###################################
# For reproducable results
np.random.seed(config['numpy_seed']) 
import tensorflow as tf
tf.set_random_seed(config['tensorflow_seed'])
###################################
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
import keras.callbacks

from outfitcmp.scripts.utils import SUPPORTED_MODELS, prepare_data_generator
from outfitcmp.scripts.predict_using_model import predict_using_model

def check_config():
    """ Sanity checks for a config """
    if os.path.exists(os.path.join(RESULTS_DIR, config['experiment_name'])):
        raise ValueError(
            'Output directory for {} experiment already exists'.format(
                config['experiment_name']
            ))
    if config['model_name'] not in SUPPORTED_MODELS:
        raise ValueError('{} is not supported. Use one of following: {}'.format(
            config['model_name'],
            SUPPORTED_MODELS
        ))
    if not os.path.exists(os.path.join(DATA_DIR, config['data_dir'])):
        raise ValueError(
            'Data directory {}  does not exists'.format(
                config['data_dir']
            ))

def init_experiment_folder():
    """ Create and initialize directory for experiment """
    experiment_dir = os.path.join(RESULTS_DIR, config['experiment_name'])
    os.makedirs(experiment_dir)
    # Copy network config for logs and possibility to reproduce result in future
    shutil.copy(CONFIG_FILE, os.path.join(experiment_dir, CONFIG_NAME))
    os.makedirs(os.path.join(experiment_dir, CHECKPOINTS_DIR_NAME))

def import_base_model():
    """ Import particular pretrained model """
    if config['model_name'] == 'Xception':
        from keras.applications.xception import Xception
        return Xception(weights='imagenet', include_top=False)
    else:
        raise ValueError('{} model is not supported. Use one of these models: {}'.format(
            config['model_name'], SUPPORTED_MODELS))

def train_model():
    """ Train model, estimate results and save logs """
    print('Start loading data')
    train_generator = prepare_data_generator(config, 'train', isRegression=config['is_regression'])
    validation_generator = prepare_data_generator(config, 'validation', isRegression=config['is_regression'])

    print('Start training')
    start = time.time()
    # Init pretrained model
    base_model = import_base_model()
    # Add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # Add a custom layers
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    # And a logistic layer
    predictions = Dense(1)(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        loss=config['loss'],
        optimizer=config['optimizer'],
        metrics=config['metrics']
    )

    # Save intermediate results
    experiment_dir = os.path.join(RESULTS_DIR, config['experiment_name'])
    check_cb = keras.callbacks.ModelCheckpoint(
        os.path.join(experiment_dir, CHECKPOINTS_DIR_NAME, '{epoch:02d}-{val_loss:.2f}.hdf5'),
        monitor='val_loss', verbose=0, save_best_only=True, mode='min', period=10
    )

    # Stop if we stop learning
    earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, min_delta=0.00001)

    model_json = model.to_json()
    with open(os.path.join(experiment_dir, config['architecture_file']), 'w') as json_file:
        json_file.write(model_json)

    # Train model
    ret = model.fit_generator(
        generator=train_generator.getGenerator(),
        steps_per_epoch=len(train_generator),
        epochs=config['num_epoches'],
        verbose=1,
        validation_data=validation_generator.getGenerator(),
        validation_steps=len(validation_generator),
        callbacks=[check_cb, earlystop_cb])

    # Serialize weights in HDF5
    model.save(os.path.join(experiment_dir, config['model_file']))
    # Save logs
    with open(os.path.join(experiment_dir, config['logs_file']), 'w') as logs_file:
        logs_file.write(str(ret.history))
    print('Saved model to disk')
    predict_using_model(experiment_dir, config, model, _isRegression=config['is_regression'])
    print('Finish working in {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start))))
    
def execute():
    """ Execute script """
    check_config()
    init_experiment_folder()
    train_model()

if __name__ == "__main__":
    execute()
