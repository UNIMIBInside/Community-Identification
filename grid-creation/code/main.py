#!/usr/local/bin/python

from utils import *
from imports import *

city = pd.DataFrame(load_json('data/it.json')).rename(columns={'city': 'name'}).set_index('name') #caricamento città italiane
italy = load_geojson('geojson-italy/geojson/limits_IT_municipalities.geojson') #caricamento geojson italia
city_italy = italy.join(city, on ='name', how='inner').reset_index() # matching tra le città italiane principali e tutte le città presenti nel dataset (limits_IT_municipalities.geojson)
get_city_grid_as_gdf(city_italy.iloc[0], 400) # 400 metri le dimensioni in x e y (teoriche)
