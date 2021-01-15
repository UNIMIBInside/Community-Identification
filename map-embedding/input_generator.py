import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input


def generate_data(datagenerator, metadata_dataframe, image_path, columns, \
                                            batch_size, target_size_1, target_size_2, shuffle=True):
    data_generator = datagenerator.flow_from_dataframe(
        dataframe=metadata_dataframe,
        directory=image_path,
        x_col="map_tiles",
        y_col=columns,
        subset="training",
        batch_size=batch_size,
        seed=42,
        shuffle=shuffle,
        class_mode="raw",
        target_size=(target_size_1, target_size_2))
    return data_generator

def generate_data_multitasking(datagenerator, metadata_dataframe, image_path, columns, \
                                    batch_size, target_size_1, target_size_2, shuffle=True):
    data_generator = datagenerator.flow_from_dataframe(
        dataframe=metadata_dataframe,
        directory=image_path,
        x_col="map_tiles",
        y_col=columns,
        subset="training",
        batch_size=batch_size,
        seed=42,
        shuffle=shuffle,
        class_mode="raw",
        target_size=(target_size_1, target_size_2))
    total_images = data_generator.n
    steps = int(np.ceil(total_images/batch_size))

    x , y = [], []
    for _ in range(steps):
        a , b = data_generator.next()

        #a = tf.convert_to_tensor(a)
        #b = tf.convert_to_tensor(b)
        a = np.array(a, dtype=np.uint8)
        b = np.array(b, dtype=np.uint8)
        
        x.extend(a)
        y.extend(b)

    #return tf.convert_to_tensor(x), tf.convert_to_tensor(y)
    return np.array(x), np.array(y)


# Image Creation
def creation_input_model(folder_path, multitask, binarization, batch_size, target_size_1, target_size_2):

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
        columns = ['peak_no', 'peak_yes', 'playground_no', 'playground_yes', 'train_station_no', \
            'train_station_yes', 'metro_station_no', 'metro_station_yes', 'tram_stop_no', \
            'tram_stop_yes', 'bus_stop_no', 'bus_stop_yes', 'university_no', 'university_yes', \
            'parking_car_no', 'parking_car_yes', 'parking_bicycle_no', 'parking_bicycle_yes', \
            'parking_motorcycle_no', 'parking_motorcycle_yes', 'water_natural_no', 'water_natural_yes', \
            'water_artificial_no', 'water_artificial_yes', 'park_no', 'park_yes', 'grassland_no', \
            'grassland_yes', 'farmland_no', 'farmland_yes', 'aerodrome_no', 'aerodrome_yes', \
            'highway_residential_no', 'highway_residential_yes', 'highway_cycleway_no', \
            'highway_cycleway_yes', 'highway_pedestrian_no', 'highway_pedestrian_yes', \
            'highway_less', 'highway_some', 'highway_more', \
            'building_less', 'building_some', 'building_more']
        n_col = 2
        offset = 38

    train_metadata_dataframe = pd.read_csv(folder_path + '/grid-creation/data/city_merged/marker_metadata_binarized.csv')
    train_image_path = folder_path + '/grid-creation/data/city_merged/map_tiles/'

    if not multitask:
        train_data = generate_data(datagenerator, train_metadata_dataframe, train_image_path, \
                                                    columns, batch_size, target_size_1, target_size_2)
    else:
        train_x, train_y = generate_data_multitasking(datagenerator, \
                                                    train_metadata_dataframe, train_image_path, \
                                                    columns, batch_size, target_size_1, target_size_2)
        print(train_x.shape)
        print(train_y.shape)

        train_targets = [[train_y[i][c*n_col:(c+1)*n_col] for i in range(train_y.shape[0])] for c in range(19)]
        train_targets.append([train_y[i][offset+0*3:offset+(0+1)*3] for i in range(train_y.shape[0])])
        train_targets.append([train_y[i][offset+1*3:offset+(1+1)*3] for i in range(train_y.shape[0])])
        train_targets = [np.array(e) for e in train_targets]
        #train_targets = [tf.convert_to_tensor(e) for e in train_targets]

        #print(len(train_targets))
        #for prova in train_targets:
        #    print(prova)
        #    print(prova.shape)

        train_data = (train_x, train_targets)


    validation_metadata_dataframe = pd.read_csv(folder_path + '/grid-creation/data/milano_merged/marker_metadata_binarized.csv')
    validation_image_path = folder_path + '/grid-creation/data/milano_merged/map_tiles/'
    
    if not multitask:
        validation_data = generate_data(datagenerator, validation_metadata_dataframe, validation_image_path, \
                                                    columns, batch_size, target_size_1, target_size_2, shuffle=False)
    else:
        validation_x, validation_y = generate_data_multitasking(datagenerator, \
                                                    validation_metadata_dataframe, validation_image_path, \
                                                    columns, batch_size, target_size_1, target_size_2, shuffle=False)
        print(validation_x.shape)
        print(validation_y.shape)

        validation_targets = [[validation_y[i][c*n_col:(c+1)*n_col] for i in range(validation_y.shape[0])] for c in range(19)]
        validation_targets.append([validation_y[i][offset+0*3:offset+(0+1)*3] for i in range(validation_y.shape[0])])
        validation_targets.append([validation_y[i][offset+1*3:offset+(1+1)*3] for i in range(validation_y.shape[0])])
        validation_targets = [np.array(e) for e in validation_targets]
        #validation_targets = [tf.convert_to_tensor(e) for e in validation_targets]
    
        validation_data = (validation_x, validation_targets)

    return train_data, validation_data
