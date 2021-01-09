from model import *
import pickle
import numpy as np
import pandas as pd
import time
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
from keras.optimizers import Adam
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

    columns = ['peak', 'playground', 'train_station', 'metro_station', 'tram_stop', 'bus_stop', \
            'university', 'parking_car', 'parking_bicycle', 'parking_motorcycle', 'water_natural', \
            'water_artificial', 'park', 'grassland', 'farmland', 'aerodrome', 'highway_residential', \
            'highway_cycleway', 'highway_pedestrian', 'highway_less', 'highway_some', 'highway_more', \
            'building_less', 'building_some', 'building_more']

    train_metadata_dataframe = pd.read_csv(folder_path + '/grid-creation/data/city_merged/marker_metadata_binarized.csv')
    train_image_path = folder_path + '/grid-creation/data/city_merged/map_tiles/'
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_metadata_dataframe,
        directory=train_image_path,
        x_col="map_tiles",
        y_col=columns,
        subset="training",
        batch_size=batch_size,
        seed=42,
        shuffle=True,
        class_mode="raw",
        target_size=(target_size_1, target_size_2))

    validation_metadata_dataframe = pd.read_csv(folder_path + '/grid-creation/data/milano_merged/marker_metadata_binarized.csv')
    validation_image_path = folder_path + '/grid-creation/data/milano_merged/map_tiles/'
    validation_generator = train_datagen.flow_from_dataframe(
        dataframe=validation_metadata_dataframe,
        directory=validation_image_path,
        x_col="map_tiles",
        y_col=columns,
        subset="training",
        batch_size=batch_size,
        seed=42,
        shuffle=True,
        class_mode="raw",
        target_size=(target_size_1, target_size_2))


    # Model creation

    build = Map_Embedding(weights=True, include_top=False, input_shape= (target_size_1, target_size_2, 3))

    # compile model
    build.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    # Model training

    hyperparams_name = 'Map_embedding_{}'.format(vector_shape)
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))

    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_accuracy', verbose=0, save_best_only=True, mode='min')

    print("training model...")
    ts = time.time()
    history = build.fit(train_generator,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        validation_data=validation_generator,
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

    # Prediction
    test_metadata_dataframe = pd.read_csv(folder_path + '/grid-creation/data/milano_merged/marker_metadata_binarized.csv')
    test_image_path = folder_path + '/grid-creation/data/milano_merged/map_tiles/'
    """test_generator = train_datagen.flow_from_dataframe(
        directory=test_image_path,
        batch_size=1,
        shuffle=False,
        target_size=(target_size_1, target_size_2))"""

    test_generator = train_datagen.flow_from_dataframe(
        dataframe=test_metadata_dataframe,
        directory=test_image_path,
        x_col="map_tiles",
        y_col=None,
        batch_size=1,
        seed=42,
        shuffle=True,
        class_mode=None,
        target_size=(target_size_1, target_size_2))


    embedding = prediction(build, test_generator, path= 'weight/Map_Embedding.h5', load = True)

    # Save results in pickle format
    with open(os.path.join('results', hyperparams_name + ".pickle"),'wb') as f: pickle.dump(embedding, f)
