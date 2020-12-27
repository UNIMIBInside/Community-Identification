from imports import *

import utils

folder_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

points_dataframe = utils.load_milan_data(f'{folder_path}/data/motoscooter_movimento_new.csv')
start = time.time()
print(f'Start Map Tiles - {points_dataframe.iloc[0].city}')
utils.extract_map_tiles(folder_path, points_dataframe)
end = time.time()
print(f'End Map Tiles - {points_dataframe.iloc[0].city}', end-start)

start = time.time()
print(f'Start OSM Metadata - {points_dataframe.iloc[0].city}')
utils.compute_nodes_and_ways(folder_path, points_dataframe)
end = time.time()
print(f'End OSM Metadata - {points_dataframe.iloc[0].city}', end-start)
print()

city_italy_selected = utils.load_json(f'{folder_path}/data/city_italy.json')
city_italy_geojson = utils.load_geojson(f'{folder_path}/data/limits_IT_municipalities.geojson')
city_italy_dataframe = city_italy_geojson.join(city_italy_selected, \
                                    on ='city', how='inner').reset_index()

for i in range(city_italy_dataframe.shape[0]):
    grid_dataframe = utils.get_city_grid(city_italy_dataframe.iloc[i], 400)

    start = time.time()
    print(f'Start Map Tiles - {grid_dataframe.iloc[0].city}')
    utils.extract_map_tiles(folder_path, grid_dataframe)
    end = time.time()
    print(f'End Map Tiles - {grid_dataframe.iloc[0].city}', end-start)

    start = time.time()
    print(f'Start OSM Metadata - {grid_dataframe.iloc[0].city}')
    utils.compute_nodes_and_ways(folder_path, grid_dataframe)
    end = time.time()
    print(f'End OSM Metadata - {grid_dataframe.iloc[0].city}', end-start)
