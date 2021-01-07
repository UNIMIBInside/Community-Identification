from imports import *
import utils

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Insert dataset path", type=str)
args = parser.parse_args()
file_path = args.file

folder_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

metadata_dataframe = pd.read_csv(f'{folder_path}/{file_path}')


not_binary_columns = ['highway', 'building']
columns = ['peak', 'playground', 'train_station', 'metro_station', 'tram_stop', \
            'bus_stop', 'university', 'parking_car', 'parking_bicycle', 'parking_motorcycle', \
            'water_natural', 'water_artificial', 'park', 'grassland', 'farmland', 'aerodrome', \
            'highway_residential', 'highway_cycleway', 'highway_pedestrian']

for field in not_binary_columns:
    first_third_quantile = metadata_dataframe[field].quantile(1/3)
    second_third_quantile = metadata_dataframe[field].quantile(2/3)
    third_third_quantile = metadata_dataframe[field].quantile(3/3)
    #print(field, first_third_quantile, second_third_quantile, third_third_quantile)
    metadata_dataframe[f'{field}_less'] = metadata_dataframe.apply(lambda x: \
        utils.binarization_quantile(x[field], -1, first_third_quantile), axis=1)
    metadata_dataframe[f'{field}_some'] = metadata_dataframe.apply(lambda x: \
        utils.binarization_quantile(x[field], first_third_quantile, second_third_quantile), axis=1)
    metadata_dataframe[f'{field}_more'] = metadata_dataframe.apply(lambda x: \
        utils.binarization_quantile(x[field], second_third_quantile, third_third_quantile), axis=1)
    metadata_dataframe = metadata_dataframe.drop(columns=[field])

for field in columns:
    metadata_dataframe[field] = metadata_dataframe.apply(lambda x: utils.binarization(x[field]), axis=1)

file_path = file_path.replace('marker_metadata', 'marker_metadata_binarized')
metadata_dataframe.to_csv(f'{folder_path}/{file_path}', index=False)
