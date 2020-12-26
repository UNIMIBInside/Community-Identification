from imports import *

#class City:
#    def __init__(self, name, poligono):
#        self.name = name
#        self.Poligono = poligono
#
#    def get_centroid(self):
#        return self.centroid
#
    
def getXY(pt): #tupla dei centroidi
    return (pt.x, pt.y)

def getXXYY(miny, minx, maxy, maxx):
    return (miny, minx, maxy, maxx)


# primo elemento corrisponde alla riga del dataset delle città, mentre la seconda è la "larghezz" di ogni quadrato
def get_city_grid_as_gdf (total_bounds, bin_side_length): #crs
    x_min, y_min, x_max, y_max = [ele for ele in total_bounds.iloc[1].bounds]
    width = bin_side_length / 111320 * 1.2
    height = bin_side_length  / 111320 * 1.2
    # width = bin_side_length / 0.706
    # height = bin_side_length / 0.706
    rows = int(np.ceil((y_max-y_min) / height))
    cols = int(np.ceil((x_max-x_min) / width))
    x_left = x_min
    x_right = x_min + width
    polygons = []
    centroid = []
    for i in range(cols):
        y_top = y_max
        y_bottom = y_max - height
        for j in range(rows):
            polygons.append(Polygon([(x_left, y_top), (x_right, y_top), (x_right, y_bottom), (x_left, y_bottom)]))
            y_top = y_top - height
            y_bottom = y_bottom - height
        x_left = x_left + width
        x_right = x_right + width
    grid = gpd.GeoDataFrame({"geometry": polygons})
    coord = grid.bounds
    grid['centroid_Point'] = grid.centroid
    grid['centroid'] = grid['centroid_Point'].apply(lambda p: getXY(p))
    grid['longitude'] = grid['centroid'].apply(lambda x: x[0])
    grid['latitude'] = grid['centroid'].apply(lambda x: x[1])
    grid = grid.join(coord)
    grid['poligono'] = grid.apply(lambda x: getXXYY(x.miny, x.minx, x.maxy, x.maxx), axis = 1)
    grid["marker_label"] = range(len(grid))
    #grid.crs = crs

    return grid



def load_geojson(path):
    db = gpd.read_file(path)
    db.name = db.name.apply(lambda x: my_custom_string_function(x))
    db = db[['name', 'geometry']]
    db = db.set_index('name', inplace = False)
    return db

def my_custom_string_function(input):
    return input.upper()

def load_json(path):
    db = pd.read_json(path)
    city = db.city.apply(lambda x: my_custom_string_function(x))
    return city

def city_list(path, path2):
    db = load_json(path)
    db_city = load_json(path2)
    city = [ele for ele in db if db_city.str.contains(ele).any()]
