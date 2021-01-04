from imports import *
import utils

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Insert dataset path", type=str)
parser.add_argument("-p", "--points", help="Insert dataset points", type=str)
parser.add_argument("-v", "--veichle", help="Insert type transport",type=str,default='motorscooter')
args = parser.parse_args()

type_veichle = args.veichle

folder_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
file_path = folder_path + '/' + args.file

dataframe = pd.read_csv(file_path)
dataframe = dataframe.loc[dataframe.type == type_veichle.upper()]
dataframe = dataframe.drop(columns=['type'])


new_points_df = pd.read_csv(folder_path + '/' + args.points)
dataframe['placelat'] = dataframe.apply(lambda x: \
    new_points_df.loc[new_points_df.marker_label == x['placename']].latitude.values[0], axis=1)
dataframe['placelon'] = dataframe.apply(lambda x: \
    new_points_df.loc[new_points_df.marker_label == x['placename']].longitude.values[0], axis=1)
dataframe['distance'] = dataframe.apply(lambda x: utils.distance(x), axis=1)
dataframe.to_csv(f'{folder_path}/data/new_fluctuo_{type_veichle}.csv', index=False)
