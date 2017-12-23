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

SUPPORTED_MODELS = ['Xception']

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
    if not os.path.exists(os.path.join(DATA_DIR, config['data_dir'], config['model_name'])):
        raise ValueError(
            'Data directory for {} model and {} initial does not exists'.format(
                config['model_name'],
                config['data_dir']
            ))

def init_experiment_folder():
    """ Create and initialize directory for experiment """
    experiment_dir = os.path.join(RESULTS_DIR, config['experiment_name'])
    os.makedirs(experiment_dir)
    # Copy network config for logs and possibility to reproduce result in future
    shutil.copy(CONFIG_FILE, os.path.join(experiment_dir, CONFIG_NAME))
    os.makedirs(os.path.join(experiment_dir, CHECKPOINTS_DIR_NAME))

def load_dataset(name):
    """ Load one dataset """
    data_dir = os.path.join(DATA_DIR, config['data_dir'], config['model_name'])
    dataset_npz = np.load(os.path.join(data_dir, name))
    # Convert them to real dicts to have possibility to add new keys in it in future
    return {'x': dataset_npz['x'], 'y': dataset_npz['y']}

def load_data():
    """
    Load training, test and validation datasests from NPZ archives
    """
    print('Start loading data {} for {} model'.format(config['data_dir'], config['model_name']))
    start = time.time()
    test = load_dataset('test.npz')
    train = load_dataset('train.npz')
    validation = load_dataset('validation.npz')
    print('Finish loading data in {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start))))
    return test, train, validation

def describe_data(test, train, validation):
    """
    Print information about datasets
    """
    print('Train shape: {}'.format(train['x'].shape))
    print('Test shape: {}'.format(test['x'].shape))
    print('Validation shape: {}'.format(validation['x'].shape))
    print('--------------------------------------------')

def import_base_model():
    """ Import particular pretrained model """
    if config['model_name'] == 'Xception':
        from keras.applications.xception import Xception
        return Xception(weights='imagenet', include_top=False)
    else:
        raise ValueError('{} model is not supported. Use one of these models: {}'.format(
            config['model_name'], SUPPORTED_MODELS))

def one_hot_encode_dataset(label_encoder, dataset):
    integer_encoded = label_encoder.fit_transform(dataset['y'])
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    return onehot_encoder.fit_transform(integer_encoded)

def one_hot_encode(test, train, validation):
    """ Transform categorical data to its one-hot-encoded version """
    label_encoder = LabelEncoder()
    for dataset in [test, train, validation]:
        dataset['one_hot_encoded'] = one_hot_encode_dataset(label_encoder, dataset)
    return label_encoder

def train_model(test, train, validation):
    """ Train model, estimate results and save logs """
    start = time.time()
    # Transform categorical data to its one-hot-encoded version
    label_encoder = one_hot_encode(test, train, validation)

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
    ret = model.fit(
        x=train['x'], y=train['one_hot_encoded'],
        batch_size=config['batch_size'],
        epochs=config['num_epoches'],
        shuffle=False, # we want reproducable results
        verbose=1,
        validation_data=(validation['x'], validation['one_hot_encoded']),
        callbacks=[check_cb, earlystop_cb])

    # Serialize weights in HDF5
    model.save_weights(os.path.join(experiment_dir, config['weights_file']))
    # Save logs
    with open(os.path.join(experiment_dir, config['logs_file']), 'w') as logs_file:
        logs_file.write(str(ret.history))
    print('Saved model to disk')
    # Predict results on test dataset
    predicted = model.predict(test['x'])
    np.savez(
        os.path.join(experiment_dir, 'predicted.npz'),
        pred=predicted
    )
    print('Saved predictions')
    # Estimate
    loss, acc = model.evaluate(test['x'], test['one_hot_encoded'], verbose=1)
    print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
    print('Finish working in {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start))))
    

def execute():
    """ Execute script """
    check_config()
    init_experiment_folder()
    test, train, validation = load_data()
    describe_data(test, train, validation)
    train_model(test, train, validation)

if __name__ == "__main__":
    execute()
