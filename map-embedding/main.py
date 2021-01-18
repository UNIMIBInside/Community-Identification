from model import *
from input_generator import *
import pickle
import numpy as np
import pandas as pd
import time
import os
import json
import argparse
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
from keras.optimizers import Adam
from keras.applications.resnet50 import preprocess_input
from keras.utils.vis_utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


parser = argparse.ArgumentParser()
parser.add_argument('--multitasking', help="Use multitask neural network", \
                                                dest='multitasking', action='store_true')
parser.set_defaults(multitasking=False)
parser.add_argument('--binarization', help="Use dataframe column with binarization", \
                                                dest='binarization', action='store_true')
parser.set_defaults(binarization=False)
parser.add_argument('--load_input', help="Load input preprocessed", \
                                                dest='load_input', action='store_true')
parser.set_defaults(load_input=False)
args = parser.parse_args()

if args.load_input and not args.multitasking:
    print('Error on use --load_input without --multitasking')
    exit(1)
if args.binarization and not args.multitasking:
    print('Error on use --binarization without --multitasking')
    exit(1)
multitask = args.multitasking
binarization = args.binarization
load_input = args.load_input


if __name__ == '__main__':
    # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:  # Currently, memory growth needs to be the same across GPUs
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)  # Memory growth must be set before GPUs have been initialized

    np.random.seed(1234)
    tf.random.set_seed(1234)
    os.chdir(os.getcwd())

    folder_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    nb_epoch =   # number of epoch at training stage
    batch_size = 64 # number of batch at training stage
    vector_shape = 128 # dimension of the embedding vector
    target_size_1 = 224
    target_size_2 = 224
    learning_rate = 
    #CACHEDATA = True  # cache data or NOT

    # Folder creation

    path_ret = 'RET'
    path_model = 'MODEL'
    path_result = 'results'
    path_weight = 'weight'
    if os.path.isdir(folder_path + '/map-embedding/' + path_ret) is False:
        os.mkdir(folder_path + '/map-embedding/' + path_ret)
    if os.path.isdir(folder_path + '/map-embedding/' + path_model) is False:
        os.mkdir(folder_path + '/map-embedding/' + path_model)
    if os.path.isdir(folder_path + '/map-embedding/' + path_result) is False:
        os.mkdir(folder_path + '/map-embedding/' + path_result)
    if os.path.isdir(folder_path + '/map-embedding/' + path_weight) is False:
        os.mkdir(folder_path + '/map-embedding/' + path_weight)

    # Image Creation

    if not load_input:
        train_data, validation_data = creation_input_model(folder_path, multitask, binarization, \
                                                        batch_size, target_size_1, target_size_2)
        if isinstance(train_data, tuple):
            train_x, train_targets = train_data
            validation_x, validation_targets = validation_data
            with open(os.path.join('results', "train_x.pickle"),'wb') as f: pickle.dump(train_x, f)
            with open(os.path.join('results', "train_targets.pickle"),'wb') as f: pickle.dump(train_targets, f)
            with open(os.path.join('results', "validation_x.pickle"),'wb') as f: pickle.dump(validation_x, f)
            with open(os.path.join('results', "validation_targets.pickle"),'wb') as f: pickle.dump(validation_targets, f)
    
    else:
        train_x = pickle.load(open(os.path.join('results', "train_x.pickle"),'rb'))
        train_targets = pickle.load(open(os.path.join('results', "train_targets.pickle"),'rb'))
        validation_x = pickle.load(open(os.path.join('results', "validation_x.pickle"),'rb'))
        validation_targets = pickle.load(open(os.path.join('results', "validation_targets.pickle"),'rb'))


    # Model creation
    if not multitask:
        build = Map_Embedding(weights=True, include_top=False, input_shape= (target_size_1, target_size_2, 3))
    else:
        build = Map_Embedding2(weights=True, include_top=False, input_shape= (target_size_1, target_size_2, 3), binarization=binarization)

    #build.summary()
    #plot_model(build, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    # compile model
    metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    if not multitask:
        #build.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=metrics)
        build.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics=metrics)
    else:
        loss_dict = {}
        loss_weight_dict = {}
        for i in range(19):
            loss_dict[f'map_embedding_output{i}'] = 'binary_crossentropy'
            loss_weight_dict[f'map_embedding_output{i}'] = 1.0
        for i in range(19,21):
            loss_dict[f'map_embedding_output{i}'] = 'categorical_crossentropy'
            loss_weight_dict[f'map_embedding_output{i}'] = 1.0
        build.compile(optimizer=Adam(lr=learning_rate), loss=loss_dict, metrics=metrics)

    # Model training

    hyperparams_name = 'Map_embedding_{}'.format(vector_shape)
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))

    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_loss', verbose=0, save_best_only=True, mode='min')

    print("training model...")
    ts = time.time()

    if not multitask:
        history = build.fit(train_data,
                            epochs=nb_epoch,
                            batch_size=batch_size,
                            validation_data=validation_data,
                            callbacks=[model_checkpoint],
                            verbose=1)
    else:
        history = build.fit(train_x, train_targets,
                            epochs=nb_epoch,
                            batch_size=batch_size,
                            validation_data=(validation_x, validation_targets),
                            callbacks=[model_checkpoint],
                            verbose=1)

    build.save_weights(os.path.join(
        'MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
    print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))

    # Save model
    hyperparams_name = 'Map_Embedding'
    build.save_weights(os.path.join('weight', '{}.h5'.format(hyperparams_name)), overwrite=True)
    json.dump(history.history, open('results/history.json', 'w'))
    pd.DataFrame.from_dict(history.history, orient="index").to_csv('results/history.csv')

    # Prediction
    datagenerator=ImageDataGenerator(preprocessing_function=preprocess_input)
    test_metadata_dataframe = pd.read_csv(folder_path + '/grid-creation/data/milano_merged/marker_metadata_binarized.csv')
    test_image_path = folder_path + '/grid-creation/data/milano_merged/map_tiles/'

    test_generator = datagenerator.flow_from_dataframe(
        dataframe=test_metadata_dataframe,
        directory=test_image_path,
        x_col="map_tiles",
        y_col=None,
        batch_size=1,
        seed=42,
        shuffle=False,
        class_mode=None,
        target_size=(target_size_1, target_size_2))

    embedding = prediction(build, test_generator, path= f'weight/{hyperparams_name}.best.h5', load = True, multitask=multitask)
    
    # Save results in pickle format
    with open(os.path.join('results', f"{hyperparams_name}.pickle"),'wb') as f: pickle.dump(embedding, f)

    #build.load_weights('weight/Map_embedding_multi_128.best.h5')
    #build.evaluate(validation_x, validation_targets, batch_size)
    