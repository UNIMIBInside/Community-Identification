from imports import *
import utils

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train_file", help="Insert path to train marker metadata csv", type=str)
parser.add_argument("-v", "--validation_file", help="Insert path to validation marker metadata csv", type=str)
args = parser.parse_args()
train_file_path = args.train_file
validation_file_path = args.validation_file

folder_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
train_dataframe = pd.read_csv(f'{folder_path}/{train_file_path}')
validation_dataframe = pd.read_csv(f'{folder_path}/{validation_file_path}')

columns = ['peak', 'playground', 'train_station', 'metro_station', 'tram_stop', \
            'bus_stop', 'university', 'parking_car', 'parking_bicycle', 'parking_motorcycle', \
            'water_natural', 'water_artificial', 'park', 'grassland', 'farmland', 'aerodrome', \
            'highway_residential', 'highway_cycleway', 'highway_pedestrian']
not_binary_columns = ['highway', 'building']


for field in columns:
    #print(field)
    train_dataframe[f'{field}_no'] = 0
    train_dataframe[f'{field}_yes'] = 0
    train_dataframe[field] = train_dataframe.apply(lambda x: utils.binarization(x[field]), axis=1)
    train_dataframe[f'{field}_yes'] = train_dataframe.apply(lambda x: utils.binarization(x[field]), axis=1)
    train_dataframe[f'{field}_no'] = (~train_dataframe[f'{field}_yes'].astype(bool)).astype(int)
    #train_dataframe = train_dataframe.drop(columns=[field])

    validation_dataframe[f'{field}_no'] = 0
    validation_dataframe[f'{field}_yes'] = 0
    validation_dataframe[field] = validation_dataframe.apply(lambda x: utils.binarization(x[field]), axis=1)
    validation_dataframe[f'{field}_yes'] = validation_dataframe.apply(lambda x: utils.binarization(x[field]), axis=1)
    validation_dataframe[f'{field}_no'] = (~validation_dataframe[f'{field}_yes'].astype(bool)).astype(int)
    #validation_dataframe = validation_dataframe.drop(columns=[field])


for field in not_binary_columns:
    #print(field)
    first_third_quantile = train_dataframe[field].quantile(1/3)
    second_third_quantile = train_dataframe[field].quantile(2/3)
    third_third_quantile = train_dataframe[field].quantile(3/3)

    train_dataframe[f'{field}_less'] = train_dataframe.apply(lambda x: \
        utils.binarization_quantile(x[field], -1, first_third_quantile), axis=1)
    train_dataframe[f'{field}_some'] = train_dataframe.apply(lambda x: \
        utils.binarization_quantile(x[field], first_third_quantile, second_third_quantile), axis=1)
    train_dataframe[f'{field}_more'] = train_dataframe.apply(lambda x: \
        utils.binarization_quantile(x[field], second_third_quantile, third_third_quantile), axis=1)
    train_dataframe = train_dataframe.drop(columns=[field])

    validation_dataframe[f'{field}_less'] = validation_dataframe.apply(lambda x: \
        utils.binarization_quantile(x[field], -1, first_third_quantile), axis=1)
    validation_dataframe[f'{field}_some'] = validation_dataframe.apply(lambda x: \
        utils.binarization_quantile(x[field], first_third_quantile, second_third_quantile), axis=1)
    validation_dataframe[f'{field}_more'] = validation_dataframe.apply(lambda x: \
        utils.binarization_quantile(x[field], second_third_quantile, third_third_quantile), axis=1)
    validation_dataframe = validation_dataframe.drop(columns=[field])


train_file_path = train_file_path.replace('marker_metadata', 'marker_metadata_binarized')
train_dataframe.to_csv(f'{folder_path}/{train_file_path}', index=False)
validation_file_path = validation_file_path.replace('marker_metadata', 'marker_metadata_binarized')
validation_dataframe.to_csv(f'{folder_path}/{validation_file_path}', index=False)
