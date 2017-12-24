"""
Author: Maria Sandrikova
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
DATA_DIR = os.path.join(WORKING_DIR, '..', 'data')
RESULTS_DIR = os.path.join(WORKING_DIR, 'results')
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
from keras.layers import Dense, GlobalAveragePooling2D
import keras.callbacks
from keras.preprocessing.image import ImageDataGenerator

SUPPORTED_MODELS = ['Xception']

# Import correct model for preprocessing
if config['model_name'] == 'Xception':
    from keras.applications.xception import preprocess_input
    print('Import modules for {} model'.format(config['model_name']))
else:
    raise ValueError('{} model is not supported. Use one of these models: {}'.format(
        config['model_name'], SUPPORTED_MODELS))

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

def get_image_size_for_model():
    """ Return size of an image accepted by given model """
    if config['model_name'] == 'Xception':
        return (299, 299)
    else:
        raise ValueError('{} model is not supported. Use one of these models: {}'.format(
            config['model_name'], SUPPORTED_MODELS))

def prepare_data_generator(split_name):
    """ Create data generator for particular split """
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    generator = datagen.flow_from_directory(
        os.path.join(DATA_DIR, config['data_dir'], split_name),
        target_size=get_image_size_for_model(),
        batch_size=config['batch_size'],
        class_mode='categorical')
    return generator

def train_model():
    """ Train model, estimate results and save logs """
    print('Start loading data')
    train_generator = prepare_data_generator('train')
    validation_generator = prepare_data_generator('validation')

    print('Start training')
    start = time.time()
    # Init pretrained model
    base_model = import_base_model()
    # Add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # Add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # And a logistic layer
    predictions = Dense(config['number_of_classes'], activation='softmax')(x)

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
    
    # Serialize model in JSON
    model_json = model.to_json()
    with open(os.path.join(experiment_dir, config['model_file']), 'w') as json_file:
        json_file.write(model_json)

    # Train model
    ret = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator),
        epochs=config['num_epoches'],
        verbose=1,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[check_cb, earlystop_cb])

    # Serialize weights in HDF5
    model.save_weights(os.path.join(experiment_dir, config['weights_file']))
    # Save logs
    with open(os.path.join(experiment_dir, config['logs_file']), 'w') as logs_file:
        logs_file.write(str(ret.history))
    print('Saved model to disk')
    test_generator = prepare_data_generator('test')
    # Predict results on test dataset
    predicted = model.predict_generator(
        generator=test_generator,
        steps=len(test_generator)
    )
    np.savez(
        os.path.join(experiment_dir, 'predicted.npz'),
        pred=predicted
    )
    print('Saved predictions')
    # Estimate
    loss, acc = model.evaluate_generator(
        generator=test_generator,
        steps=len(test_generator)
    )
    print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
    print('Finish working in {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start))))
    
def execute():
    """ Execute script """
    check_config()
    init_experiment_folder()
    train_model()

if __name__ == "__main__":
    execute()
