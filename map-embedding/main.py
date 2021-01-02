from model import *
import pickle
import numpy as np
import time
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import tensorflow as tf

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

    DATAPATH = '../grid-creation/data'
    nb_epoch = 150  # number of epoch at training stage
    batch_size = 32 # number of batch at training stage
    vector_shape = 1024 # dimension of the embedding vector
    #CACHEDATA = True  # cache data or NOT

    # Folder creation

    path_result = 'RET'
    path_model = 'MODEL'
    path_weight = 'weight'
    if os.path.isdir(path_result) is False:
        os.mkdir(path_result)
    if os.path.isdir(path_model) is False:
        os.mkdir(path_model)
    if os.path.isdir('results') is False:
        os.mkdir('results')
    if os.path.isdir(path_weight) is False:
        os.mkdir(path_weight)

    # Model creation
    build = Map_Embedding(weights= None, include_top=False, input_shape= (32,32,3))

    # Model training

    hyperparams_name = 'Map_embedding_{}'.format(vector_shape)
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))

    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

    print("training model...")
    ts = time.time()
    history = build.fit(X_train, Y_train,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        validation_data=(X_test, Y_test),
                        callbacks=[model_checkpoint],
                        verbose=0)
    build.save_weights(os.path.join(
        'MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
    print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))

    # Save model
    hyperparams_name = 'Map_Embedding'
    build.save_weights(os.path.join('weight', '{}.h5'.format(hyperparams_name)), overwrite=True)

    # Prediction
    embedding = prediction(build, input, path= 'weight/Map_Embedding.h5', load = False)

    # Save results in pickle format
    embedding.to_pickle(
        os.path.join(
            'results', hyperparams_name + ".pickle"
        )
    )