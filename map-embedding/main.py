from model import *
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

def generate_data_from_directory(datagenerator, metadata_dataframe, image_path, columns, \
                                                    batch_size, target_size_1, target_size_2):
    data_generator = datagenerator.flow_from_dataframe(
        dataframe=metadata_dataframe,
        directory=image_path,
        x_col="map_tiles",
        y_col=columns,
        subset="training",
        batch_size=batch_size,
        seed=42,
        shuffle=True,
        class_mode="raw",
        target_size=(target_size_1, target_size_2))
    total_images = data_generator.n
    steps = (total_images//batch_size) + 1
    x , y = [], []
    for i in range(steps):
        a , b = data_generator.next()
        x.extend(a)
        y.extend(b)
    return np.array(x), np.array(y)


parser = argparse.ArgumentParser()
parser.add_argument('--multitasking', help="Use multitask neural network", \
                                                dest='multitasking', action='store_true')
parser.set_defaults(multitasking=False)
parser.add_argument('--binarization', help="Use dataframe column with binarization", \
                                                dest='binarization', action='store_true')
parser.set_defaults(binarization=False)
args = parser.parse_args()

if args.binarization and not args.multitasking:
    print('Error on use --binarization without --multitasking')
    exit(1)
multitask = args.multitasking
binarization = args.binarization


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
    nb_epoch = 20  # number of epoch at training stage
    batch_size = 64 # number of batch at training stage
    vector_shape = 128 # dimension of the embedding vector
    target_size_1 = 224
    target_size_2 = 224
    learning_rate = 0.001
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

    datagenerator=ImageDataGenerator(preprocessing_function=preprocess_input)

    if not binarization:
        columns = ['peak', 'playground', 'train_station', 'metro_station', 'tram_stop', 'bus_stop', \
            'university', 'parking_car', 'parking_bicycle', 'parking_motorcycle', 'water_natural', \
            'water_artificial', 'park', 'grassland', 'farmland', 'aerodrome', 'highway_residential', \
            'highway_cycleway', 'highway_pedestrian', 'highway_less', 'highway_some', 'highway_more', \
            'building_less', 'building_some', 'building_more']
        n_col = 1
        offset = 19
    else:
        columns = ['peak_yes', 'peak_no', 'playground_yes', 'playground_no', 'train_station_yes', \
            'train_station_no', 'metro_station_yes', 'metro_station_no', 'tram_stop_yes', \
            'tram_stop_no', 'bus_stop_yes', 'bus_stop_no', 'university_yes', 'university_no', \
            'parking_car_yes', 'parking_car_no', 'parking_bicycle_yes', 'parking_bicycle_no', \
            'parking_motorcycle_yes', 'parking_motorcycle_no', 'water_natural_yes', 'water_natural_no', \
            'water_artificial_yes', 'water_artificial_no', 'park_yes', 'park_no', 'grassland_yes', \
            'grassland_no', 'farmland_yes', 'farmland_no', 'aerodrome_yes', 'aerodrome_no', \
            'highway_residential_yes', 'highway_residential_no', 'highway_cycleway_yes', \
            'highway_cycleway_no', 'highway_pedestrian_yes', 'highway_pedestrian_no', \
            'highway_less', 'highway_some', 'highway_more', \
            'building_less', 'building_some', 'building_more']
        n_col = 2
        offset = 38

    train_metadata_dataframe = pd.read_csv(folder_path + '/grid-creation/data/city_merged/marker_metadata_binarized.csv')
    train_image_path = folder_path + '/grid-creation/data/city_merged/map_tiles/'
    
    train_x, train_y = generate_data_from_directory(datagenerator, \
                                                    train_metadata_dataframe, train_image_path, \
                                                    columns, batch_size, target_size_1, target_size_2)
    if not multitask:
        train_targets = train_y
    else:
        print(train_x.shape)
        print(train_y.shape)
        train_targets = []
        for i in range(train_y.shape[0]):
            if not train_targets:
                for c in range(19):
                    #print(c*n_col, (c+1)*n_col)
                    train_targets.append([train_y[i][c*n_col:(c+1)*n_col]])
                for c in range(2):
                    #print(offset+3*c, offset+3*(c+1))
                    train_targets.append([train_y[i][offset+c*3:offset+(c+1)*3]])
            else:
                for c in range(19):
                    train_targets[c].append(train_y[i][c*n_col:(c+1)*n_col])
                for c in range(2):
                    train_targets[19+c].append(train_y[i][offset+c*3:offset+(c+1)*3])
        train_targets = [np.array(e) for e in train_targets]
        #for prova in train_targets:
        #    print(prova)
        #    print(prova.shape)

    
    validation_metadata_dataframe = pd.read_csv(folder_path + '/grid-creation/data/milano_merged/marker_metadata_binarized.csv')
    validation_image_path = folder_path + '/grid-creation/data/milano_merged/map_tiles/'
    
    validation_x, validation_y = generate_data_from_directory(datagenerator, \
                                                validation_metadata_dataframe, validation_image_path, \
                                                columns, batch_size, target_size_1, target_size_2)
    if not multitask:
        validation_targets = validation_y
    else:
        print(validation_x.shape)
        print(validation_y.shape)
        validation_targets = []
        for i in range(validation_y.shape[0]):
            if not validation_targets:
                for c in range(19):
                    validation_targets.append([validation_y[i][c*n_col:(c+1)*n_col]])
                for c in range(2):
                    validation_targets.append([validation_y[i][offset+c*3:offset+(c+1)*3]])
            else:
                for c in range(19):
                    validation_targets[c].append(validation_y[i][c*n_col:(c+1)*n_col])
                for c in range(2):
                    validation_targets[19+c].append(validation_y[i][offset+c*3:offset+(c+1)*3])
        validation_targets = [np.array(e) for e in validation_targets]


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
        build.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=metrics)
    else:
        loss_dict = {}
        for i in range(19):
            loss_dict[f'map_embedding_output{i}'] = 'binary_crossentropy'
        for i in range(19,21):
            loss_dict[f'map_embedding_output{i}'] = 'categorical_crossentropy'
        build.compile(optimizer=Adam(lr=learning_rate), loss=loss_dict, metrics=metrics)

    # Model training

    hyperparams_name = 'Map_embedding_{}'.format(vector_shape)
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))

    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_loss', verbose=0, save_best_only=True, mode='min')

    print("training model...")
    ts = time.time()
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


    # Prediction
    test_metadata_dataframe = pd.read_csv(folder_path + '/grid-creation/data/milano_merged/marker_metadata_binarized.csv')
    test_image_path = folder_path + '/grid-creation/data/milano_merged/map_tiles/'

    test_generator = datagenerator.flow_from_dataframe(
        dataframe=test_metadata_dataframe,
        directory=test_image_path,
        x_col="map_tiles",
        y_col=None,
        batch_size=1,
        seed=42,
        shuffle=True,
        class_mode=None,
        target_size=(target_size_1, target_size_2))


    embedding = prediction(build, test_generator, path= 'weight/Map_Embedding.h5', load = True, multitask=multitask)

    # Save results in pickle format
    with open(os.path.join('results', hyperparams_name + ".pickle"),'wb') as f: pickle.dump(embedding, f)
