from imports import *

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Insert dataset path", type=str)
args = parser.parse_args()
file_path = args.file

folder_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

for c in range(3):
    try:
        os.mkdir(f'{folder_path}/{file_path}_{c}_merged')
    except FileExistsError:
        pass
    try:
        os.mkdir(f'{folder_path}/{file_path}_{c}_merged/map_tiles')
    except FileExistsError:
        pass

columns = ['map_tiles', 'peak', 'playground', 'train_station', 'metro_station', 'tram_stop', \
            'bus_stop', 'university', 'parking_car', 'parking_bicycle', 'parking_motorcycle', \
            'water_natural', 'water_artificial', 'park', 'grassland', 'farmland', 'aerodrome', \
            'highway', 'highway_residential', 'highway_cycleway', 'highway_pedestrian', 'building']
columns.pop(0)

city_list = os.listdir(f'{folder_path}/{file_path}')
#city_list = sorted(city_list)
random.shuffle(city_list)

n_city = len(city_list)

i = 0
c = 0
metadata_dataframe = pd.DataFrame(columns=columns)
for city in city_list[:int(n_city/3)]:
    #i += 1
    #if i > 1:
    #    break

    map_tiles_files = os.listdir(f'{folder_path}/{file_path}/{city}/map_tiles')
    new_map_tiles_files = [f'{city}_{map_tiles}' for map_tiles in map_tiles_files]
    city_metadata_dataframe = pd.read_csv(f'{folder_path}/{file_path}/{city}/marker_metadata.csv')
    city_metadata_dataframe['map_tiles'] = city_metadata_dataframe.apply(lambda x: \
                                            f'{x["city"]}_{int(x["marker_label"])}.png', axis=1)
    city_metadata_dataframe = city_metadata_dataframe.drop(columns=['longitude', 'latitude', \
                            'lon_min', 'lat_min', 'lon_max', 'lat_max', 'marker_label', 'city'])

    if len(map_tiles_files) == city_metadata_dataframe.shape[0]:
        print(city)
    else:
        print(f'{city} - Missing data')
        match_map_tiles = [bool(f'{city}_{m}.png' in new_map_tiles_files) \
                                for m in range(city_metadata_dataframe.shape[0])]
        missing_map_tiles = []
        i = 0
        while i < city_metadata_dataframe.shape[0]:
            try:
                missing_map_tiles.append(match_map_tiles.index(False, i))
                i = match_map_tiles.index(False, i) + 1
            except ValueError:
                i = city_metadata_dataframe.shape[0]
        city_metadata_dataframe = city_metadata_dataframe.drop(missing_map_tiles)
    
    city_metadata_dataframe['metadata_sum'] = city_metadata_dataframe.apply(lambda x: \
                                                            int(x[columns].sum()), axis=1)
    empty_map_tiles = list(city_metadata_dataframe.loc[city_metadata_dataframe.metadata_sum == 0].map_tiles)
    city_metadata_dataframe = city_metadata_dataframe.loc[city_metadata_dataframe.metadata_sum != 0]
    city_metadata_dataframe = city_metadata_dataframe.drop(columns=['metadata_sum'])
    metadata_dataframe = metadata_dataframe.append(city_metadata_dataframe)

    for old_map_tiles_name, new_map_tiles_name in zip(map_tiles_files, new_map_tiles_files):
        if not new_map_tiles_name in empty_map_tiles:
            old_path = f'{folder_path}/{file_path}/{city}/map_tiles/{old_map_tiles_name}'
            new_path = f'{folder_path}/{file_path}_{c}_merged/map_tiles/{new_map_tiles_name}'
            shutil.copyfile(old_path, new_path)


metadata_dataframe.to_csv(f'{folder_path}/{file_path}_{c}_merged/marker_metadata.csv', index=False)


i = 0
c = 1
metadata_dataframe = pd.DataFrame(columns=columns)
for city in city_list[int(n_city/3):int(2*n_city/3)]:
    #i += 1
    #if i > 1:
    #    break

    map_tiles_files = os.listdir(f'{folder_path}/{file_path}/{city}/map_tiles')
    new_map_tiles_files = [f'{city}_{map_tiles}' for map_tiles in map_tiles_files]
    city_metadata_dataframe = pd.read_csv(f'{folder_path}/{file_path}/{city}/marker_metadata.csv')
    city_metadata_dataframe['map_tiles'] = city_metadata_dataframe.apply(lambda x: \
                                            f'{x["city"]}_{int(x["marker_label"])}.png', axis=1)
    city_metadata_dataframe = city_metadata_dataframe.drop(columns=['longitude', 'latitude', \
                            'lon_min', 'lat_min', 'lon_max', 'lat_max', 'marker_label', 'city'])

    if len(map_tiles_files) == city_metadata_dataframe.shape[0]:
        print(city)
    else:
        print(f'{city} - Missing data')
        match_map_tiles = [bool(f'{city}_{m}.png' in new_map_tiles_files) \
                                for m in range(city_metadata_dataframe.shape[0])]
        missing_map_tiles = []
        i = 0
        while i < city_metadata_dataframe.shape[0]:
            try:
                missing_map_tiles.append(match_map_tiles.index(False, i))
                i = match_map_tiles.index(False, i) + 1
            except ValueError:
                i = city_metadata_dataframe.shape[0]
        city_metadata_dataframe = city_metadata_dataframe.drop(missing_map_tiles)
    
    city_metadata_dataframe['metadata_sum'] = city_metadata_dataframe.apply(lambda x: \
                                                            int(x[columns].sum()), axis=1)
    empty_map_tiles = list(city_metadata_dataframe.loc[city_metadata_dataframe.metadata_sum == 0].map_tiles)
    city_metadata_dataframe = city_metadata_dataframe.loc[city_metadata_dataframe.metadata_sum != 0]
    city_metadata_dataframe = city_metadata_dataframe.drop(columns=['metadata_sum'])
    metadata_dataframe = metadata_dataframe.append(city_metadata_dataframe)

    for old_map_tiles_name, new_map_tiles_name in zip(map_tiles_files, new_map_tiles_files):
        if not new_map_tiles_name in empty_map_tiles:
            old_path = f'{folder_path}/{file_path}/{city}/map_tiles/{old_map_tiles_name}'
            new_path = f'{folder_path}/{file_path}_{c}_merged/map_tiles/{new_map_tiles_name}'
            shutil.copyfile(old_path, new_path)


metadata_dataframe.to_csv(f'{folder_path}/{file_path}_{c}_merged/marker_metadata.csv', index=False)


i = 0
c = 2
metadata_dataframe = pd.DataFrame(columns=columns)
for city in city_list[int(2*n_city/3):]:
    #i += 1
    #if i > 1:
    #    break

    map_tiles_files = os.listdir(f'{folder_path}/{file_path}/{city}/map_tiles')
    new_map_tiles_files = [f'{city}_{map_tiles}' for map_tiles in map_tiles_files]
    city_metadata_dataframe = pd.read_csv(f'{folder_path}/{file_path}/{city}/marker_metadata.csv')
    city_metadata_dataframe['map_tiles'] = city_metadata_dataframe.apply(lambda x: \
                                            f'{x["city"]}_{int(x["marker_label"])}.png', axis=1)
    city_metadata_dataframe = city_metadata_dataframe.drop(columns=['longitude', 'latitude', \
                            'lon_min', 'lat_min', 'lon_max', 'lat_max', 'marker_label', 'city'])

    if len(map_tiles_files) == city_metadata_dataframe.shape[0]:
        print(city)
    else:
        print(f'{city} - Missing data')
        match_map_tiles = [bool(f'{city}_{m}.png' in new_map_tiles_files) \
                                for m in range(city_metadata_dataframe.shape[0])]
        missing_map_tiles = []
        i = 0
        while i < city_metadata_dataframe.shape[0]:
            try:
                missing_map_tiles.append(match_map_tiles.index(False, i))
                i = match_map_tiles.index(False, i) + 1
            except ValueError:
                i = city_metadata_dataframe.shape[0]
        city_metadata_dataframe = city_metadata_dataframe.drop(missing_map_tiles)
    
    city_metadata_dataframe['metadata_sum'] = city_metadata_dataframe.apply(lambda x: \
                                                            int(x[columns].sum()), axis=1)
    empty_map_tiles = list(city_metadata_dataframe.loc[city_metadata_dataframe.metadata_sum == 0].map_tiles)
    city_metadata_dataframe = city_metadata_dataframe.loc[city_metadata_dataframe.metadata_sum != 0]
    city_metadata_dataframe = city_metadata_dataframe.drop(columns=['metadata_sum'])
    metadata_dataframe = metadata_dataframe.append(city_metadata_dataframe)

    for old_map_tiles_name, new_map_tiles_name in zip(map_tiles_files, new_map_tiles_files):
        if not new_map_tiles_name in empty_map_tiles:
            old_path = f'{folder_path}/{file_path}/{city}/map_tiles/{old_map_tiles_name}'
            new_path = f'{folder_path}/{file_path}_{c}_merged/map_tiles/{new_map_tiles_name}'
            shutil.copyfile(old_path, new_path)


metadata_dataframe.to_csv(f'{folder_path}/{file_path}_{c}_merged/marker_metadata.csv', index=False)
