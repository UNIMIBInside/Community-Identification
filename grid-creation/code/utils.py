from imports import *

def load_milan_data(path):
    """Funzione di caricamento del dataset di Milano per ottenere il dataframe dei marker"""
    motoscooter_dataframe = pd.read_csv(path)
    points_dataframe = motoscooter_dataframe[['placelat', 'placelon', 'placename']]
    points_dataframe = points_dataframe.rename( \
        columns={'placelat': 'latitude', 'placelon': 'longitude', 'placename': 'marker_label'})
    points_dataframe = points_dataframe.drop_duplicates()
    delta = 0.00235
    points_dataframe['lat_min'] = points_dataframe.latitude - delta
    points_dataframe['lat_max'] = points_dataframe.latitude + delta
    points_dataframe['lon_min'] = points_dataframe.longitude - delta
    points_dataframe['lon_max'] = points_dataframe.longitude + delta
    points_dataframe['city'] = 'Milano'
    points_dataframe = points_dataframe.sort_values('marker_label')
    return points_dataframe

def load_json(path):
    """Funzione di caricamento json delle città italiane selezionate"""
    dataframe = pd.read_json(path)
    dataframe.city = dataframe.city.apply(lambda x: x.title())
    dataframe = dataframe[['city']]
    dataframe = dataframe.set_index('city')
    return dataframe

def load_geojson(path):
    """Funzione di caricamento dati geojson delle città italiane"""
    dataframe = gpd.read_file(path)
    dataframe.name = dataframe.name.apply(lambda x: x.title())
    dataframe = dataframe[['name', 'geometry']]
    dataframe = dataframe.rename(columns={'name': 'city'})
    dataframe = dataframe.set_index('city', inplace = False)
    return dataframe

def get_city_grid(row, bin_side_length):
    """Funzione per la costruzione del dataframe con la griglia della città"""

    city_label = row.city
    lon_min, lat_min, lon_max, lat_max = row.geometry.bounds
    #width = bin_side_length / 111320 * 1.2
    #height = bin_side_length  / 111320 * 1.2
    #Length in meters of 1° of latitude = always 111.32 km
    #Length in meters of 1° of longitude = 40075 km * cos( latitude ) / 360
    height = 0.00470
    width = 0.00470
    grid_n_rows = int(np.ceil((lat_max-lat_min) / height))
    grid_n_cols = int(np.ceil((lon_max-lon_min) / width))

    polygons = []
    y_bottom = lat_min
    y_top = lat_min + height
    for _ in range(grid_n_rows):
        x_left = lon_min
        x_right = lon_min + width
        for _ in range(grid_n_cols):
            polygons.append(Polygon([(x_left, y_top), (x_right, y_top), \
                                    (x_right, y_bottom), (x_left, y_bottom)]))
            x_left = x_left + width
            x_right = x_right + width
        y_bottom = y_bottom + height
        y_top = y_top + height

    grid_dataframe = gpd.GeoDataFrame({"geometry": polygons})
    grid_dataframe['centroid'] = grid_dataframe.centroid.apply(lambda pt: (pt.x, pt.y))
    grid_dataframe['longitude'] = grid_dataframe['centroid'].apply(lambda x: x[0])
    grid_dataframe['latitude'] = grid_dataframe['centroid'].apply(lambda x: x[1])
    grid_dataframe = grid_dataframe.join(grid_dataframe.geometry.bounds)
    grid_dataframe['marker_label'] = range(grid_dataframe.shape[0])
    grid_dataframe['city'] = city_label

    grid_dataframe = grid_dataframe.drop(columns=['geometry', 'centroid'])
    grid_dataframe = grid_dataframe.rename(columns={'minx': 'lon_min', 'miny': 'lat_min', \
                                                        'maxx': 'lon_max', 'maxy': 'lat_max'})

    return grid_dataframe


def extract_map_tiles(folder_path, points_dataframe):
    """Funzione di estrazione dei map tiles a partire dai marker"""

    city_label = points_dataframe.iloc[0].city
    try:
        os.mkdir(f'{folder_path}/data/{city_label}')
    except FileExistsError:
        pass
    try:
        os.mkdir(f'{folder_path}/data/{city_label}/map_tiles')
    except FileExistsError:
        pass

    for i in range(0, points_dataframe.shape[0]):
        #latitude_min = points_dataframe.iloc[i].lat_min
        #latitude_max = points_dataframe.iloc[i].lat_max
        #longitude_min = points_dataframe.iloc[i].lon_min
        #longitude_max = points_dataframe.iloc[i].lon_max
        #delta_lat_meter = geopy.distance.geodesic((latitude_min, longitude_min), \
        #                                            (latitude_max, longitude_min)).m
        #delta_lon_meter = geopy.distance.geodesic((latitude_min, longitude_min), \
        #                                            (latitude_min, longitude_max)).m
        #delta_lat_pixel = int(np.ceil(delta_lat_meter/1.6))
        #delta_lon_pixel = int(np.ceil(delta_lon_meter/1.6))
        delta_lat_pixel = 325
        delta_lon_pixel = 230
        map_tiles = False
        while not map_tiles:
            try:
                map_tiles = StaticMap(delta_lon_pixel, delta_lat_pixel, \
                                      url_template='https://a.tile.openstreetmap.org/{z}/{x}/{y}.png')
                map_tiles = True
            except Exception as e:
                print(e)
                time.sleep(10)

        coords = (points_dataframe.iloc[i].longitude, points_dataframe.iloc[i].latitude)
        marker_outline = CircleMarker(coords, 'white', 0.1)
        map_tiles.add_marker(marker_outline)
        image = map_tiles.render(zoom=16)
        image_path = folder_path + '/data/' + city_label + '/map_tiles/' + \
                        str(int(points_dataframe.iloc[i].marker_label)) + '.png'
        image.save(image_path)


def compute_node_count(points_dataframe, marker_label, nodes):
    """Funzione di conteggio metadata nodi"""

    nodes = [node.tags for node in nodes]

    nodes = ['peak' if isinstance(node, dict) and node.get('natural') \
                and node['natural'] == 'peak' else node for node in nodes]
    nodes = ['playground' if isinstance(node, dict) and node.get('leisure') \
                and node['leisure'] == 'playground' else node for node in nodes]
    nodes = ['train_station' if isinstance(node, dict) and node.get('railway') \
                and node['railway'] == 'station' and (not node.get('station') or \
                node.get('station') and node['station'] != 'subway') else node for node in nodes]
    nodes = ['metro_station' if isinstance(node, dict) and node.get('railway') \
                and node['railway'] == 'station' and node.get('station') \
                and node['station'] == 'subway' else node for node in nodes]
    nodes = ['train_station' if isinstance(node, dict) and node.get('railway') \
                and node['railway'] == 'halt' else node for node in nodes]
    nodes = ['tram_stop' if isinstance(node, dict) and node.get('railway') \
                and node['railway'] == 'tram_stop' else node for node in nodes]
    nodes = ['bus_stop' if isinstance(node, dict) and node.get('highway') \
                and node['highway'] == 'bus_stop' and (not node.get('railway') or \
                node.get('railway') and node['railway'] != 'tram_stop') else node for node in nodes]
    nodes = ['university' if isinstance(node, dict) and node.get('amenity') \
                and node['amenity'] == 'university' else node for node in nodes]
    nodes = ['aerodrome' if isinstance(node, dict) and node.get('aeroway') \
                and node['aeroway'] == 'aerodrome' else node for node in nodes]
    nodes = ['parking_car' if isinstance(node, dict) and node.get('amenity') \
                and node['amenity'] == 'parking' else node for node in nodes]
    nodes = ['parking_bicycle' if isinstance(node, dict) and node.get('amenity') \
                and node['amenity'] == 'bicycle_parking' else node for node in nodes]
    nodes = ['parking_motorcycle' if isinstance(node, dict) and node.get('amenity') \
                and node['amenity'] == 'motorcycle_parking' else node for node in nodes]

    counter_node = Counter(nodes)
    for type_node, count_type in counter_node.most_common():
        points_dataframe.loc[points_dataframe.marker_label == marker_label, type_node] += count_type
    return points_dataframe

def compute_way_count(points_dataframe, marker_label, ways):
    """Funzione di conteggio metadata way"""

    ways = [way.tags for way in ways]

    ways = ['playground' if isinstance(way, dict) and way.get('leisure') \
                and way['leisure'] == 'playground' else way for way in ways]
    ways = ['university' if isinstance(way, dict) and way.get('amenity') \
                and way['amenity'] == 'university' else way for way in ways]
    ways = ['water_natural' if isinstance(way, dict) and way.get('natural') \
                and way['natural'] == 'water' and way.get('water') and way['water'] \
                in ['lake', 'river', 'oxbow', 'lagoon', 'stream_pool'] else way for way in ways]
    ways = ['water_artificial' if isinstance(way, dict) and way.get('natural') \
                and way['natural'] == 'water' else way for way in ways]
    ways = ['park' if isinstance(way, dict) and way.get('leisure') \
                and way['leisure'] == 'park' else way for way in ways]
    ways = ['grassland' if isinstance(way, dict) and way.get('natural') \
                and way['natural'] == 'grassland' else way for way in ways]
    ways = ['farmland' if isinstance(way, dict) and way.get('landuse') \
                and way['landuse'] == 'farmland' else way for way in ways]
    ways = ['aerodrome' if isinstance(way, dict) and way.get('aeroway') \
                and way['aeroway'] == 'aerodrome' else way for way in ways]
    ways = ['highway_residential' if isinstance(way, dict) and way.get('highway') \
                and way['highway'] == 'residential' else way for way in ways]
    ways = ['highway_cycleway' if isinstance(way, dict) and way.get('highway') \
                and way['highway'] == 'cycleway' else way for way in ways]
    ways = ['highway_pedestrian' if isinstance(way, dict) and way.get('highway') \
                and way['highway'] == 'pedestrian' else way for way in ways]
    ways = ['highway' if isinstance(way, dict) and way.get('highway') else way for way in ways]
    ways = ['building' if isinstance(way, dict) and way.get('building') else way for way in ways]
    ways = ['parking_car' if isinstance(way, dict) and way.get('amenity') \
                and way['amenity'] == 'parking' else way for way in ways]
    ways = ['parking_bicycle' if isinstance(way, dict) and way.get('amenity') \
                and way['amenity'] == 'bicycle_parking' else way for way in ways]
    ways = ['parking_motorcycle' if isinstance(way, dict) and way.get('amenity') \
                and way['amenity'] == 'motorcycle_parking' else way for way in ways]

    counter_ways = Counter(ways)
    for type_way, count_type in counter_ways.most_common():
        points_dataframe.loc[points_dataframe.marker_label == marker_label, type_way] += count_type
    return points_dataframe

def compute_nodes_and_ways(folder_path, points_dataframe):
    """Funzione per il calcolo dei nodes e ways per i riquadri"""

    overpass_api = overpy.Overpass()
    city_label = points_dataframe.iloc[0].city

    points_dataframe['peak'] = 0
    points_dataframe['playground'] = 0
    points_dataframe['train_station'] = 0
    points_dataframe['metro_station'] = 0
    points_dataframe['tram_stop'] = 0
    points_dataframe['bus_stop'] = 0
    points_dataframe['university'] = 0
    points_dataframe['parking_car'] = 0
    points_dataframe['parking_bicycle'] = 0
    points_dataframe['parking_motorcycle'] = 0

    points_dataframe['water_natural'] = 0
    points_dataframe['water_artificial'] = 0
    points_dataframe['park'] = 0
    points_dataframe['grassland'] = 0
    points_dataframe['farmland'] = 0
    points_dataframe['aerodrome'] = 0
    points_dataframe['highway'] = 0
    points_dataframe['highway_residential'] = 0
    points_dataframe['highway_cycleway'] = 0
    points_dataframe['highway_pedestrian'] = 0
    points_dataframe['building'] = 0


    for i in range(points_dataframe.shape[0]):
        iterative_borders = (points_dataframe.iloc[i].lat_min, points_dataframe.iloc[i].lon_min, \
                                points_dataframe.iloc[i].lat_max, points_dataframe.iloc[i].lon_max)
        query_executed = False
        while not query_executed:
            try:
                query = f"""(
                            node[natural=peak]{iterative_borders};
                            node[leisure=playground]{iterative_borders};
                            way[leisure=playground]{iterative_borders};
                            node[railway=station]{iterative_borders};
                            node[railway=halt]{iterative_borders};
                            node[railway=tram_stop]{iterative_borders};
                            node[highway=bus_stop]{iterative_borders};
                            node[amenity=university]{iterative_borders};
                            way[amenity=university]{iterative_borders};
                            node[amenity=parking]{iterative_borders};
                            way[amenity=parking]{iterative_borders};
                            node[amenity=bicycle_parking]{iterative_borders};
                            way[amenity=bicycle_parking]{iterative_borders};
                            node[amenity=motorcycle_parking]{iterative_borders};
                            way[amenity=motorcycle_parking]{iterative_borders};
                            way[natural=water]{iterative_borders};
                            way[leisure=park]{iterative_borders};
                            way[natural=grassland]{iterative_borders};
                            way[landuse=farmland]{iterative_borders};
                            way[aeroway=aerodrome]{iterative_borders};
                            node[aeroway=aerodrome]{iterative_borders};
                            way[highway]{iterative_borders};
                            way[building]{iterative_borders};
                        );
                        out center;"""
                result = overpass_api.query(query)
                points_dataframe = compute_node_count(points_dataframe, \
                                    points_dataframe.iloc[i].marker_label, result.nodes)
                points_dataframe = compute_way_count(points_dataframe, \
                                    points_dataframe.iloc[i].marker_label, result.ways)
                query_executed = True
            except (overpy.exception.OverpassTooManyRequests, \
                        overpy.exception.OverpassGatewayTimeout):
                print(i)
                print('Start sleep')
                time.sleep(60)
                print('End sleep')

    points_dataframe.to_csv(f'{folder_path}/data/{city_label}/marker_metadata.csv', index=False)
