from model import *
import os
import pickle

if __name__ == '__main__':
    os.chdir(os.getcwd())
    path_weight = 'weight'
    path_result = 'results'
    if os.path.isdir(path_result) is False:
        os.mkdir(path_result)
    if os.path.isdir(path_weight) is False:
        os.mkdir(path_weight)
    build = Map_Embedding(weights= None, include_top=False, input_shape= (32,32,3))


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