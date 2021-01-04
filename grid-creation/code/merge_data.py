from imports import *


folder_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

try:
    os.mkdir(f'{folder_path}/data/city_merged')
except FileExistsError:
    pass
try:
    os.mkdir(f'{folder_path}/data/city_merged/map_tiles')
except FileExistsError:
    pass
columns = ['map_tiles', 'peak', 'playground', 'train_station', 'metro_station', 'tram_stop', \
            'bus_stop', 'university', 'parking_car', 'parking_bicycle', 'parking_motorcycle', \
            'water_natural', 'water_artificial', 'park', 'grassland', 'farmland', 'aerodrome', \
            'highway', 'highway_residential', 'highway_cycleway', 'highway_pedestrian', 'building']
metadata_dataframe = pd.DataFrame(columns=columns)

city_list = os.listdir(f'{folder_path}/data/City')

for city in city_list:

    map_tiles_files = os.listdir(f'{folder_path}/data/City/{city}/map_tiles')
    new_map_tiles_files = [f'{city}_{map_tiles}' for map_tiles in map_tiles_files]
    city_metadata_dataframe = pd.read_csv(f'{folder_path}/data/City/{city}/marker_metadata.csv')
    city_metadata_dataframe['map_tiles'] = city_metadata_dataframe.apply(lambda x: \
                                            f'{x["city"]}_{x["marker_label"]}.png', axis=1)
    city_metadata_dataframe = city_metadata_dataframe.drop(columns=['longitude', 'latitude', \
                            'lon_min', 'lat_min', 'lon_max', 'lat_max', 'marker_label', 'city'])

    for old_map_tiles_name, new_map_tiles_name in zip(map_tiles_files, new_map_tiles_files):
        old_path = f'{folder_path}/data/City/{city}/map_tiles/{old_map_tiles_name}'
        new_path = f'{folder_path}/data/city_merged/map_tiles/{new_map_tiles_name}'
        shutil.copyfile(old_path, new_path)

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


    metadata_dataframe = metadata_dataframe.append(city_metadata_dataframe)


metadata_dataframe.to_csv(f'{folder_path}/data/city_merged/marker_metadata.csv', index=False)
