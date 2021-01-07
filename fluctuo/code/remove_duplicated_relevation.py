from imports import *
import utils

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Insert dataset path", type=str)
parser.add_argument("-c", "--clean", help="Insert dataset clean path", type=str)
args = parser.parse_args()

folder_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
file_path = folder_path + '/' + args.file
clean_file_path = folder_path + '/' + args.clean

dataframe = pd.read_csv(file_path)


start_shape_df = dataframe.shape[0]
end_shape_df = 0
i = 0

while not start_shape_df == end_shape_df:
    i += 1
    print(f'{i} - Start')
    start_shape_df = dataframe.shape[0]

    dataframe['lat&lng'] = list(zip(list(dataframe.lat.round(6)), list(dataframe.lng.round(6))))
    dataframe = dataframe.sort_values(by=['provider_id', 'created_at', 'lat&lng', 'placename'])
    dataframe['nextObservation'] = dataframe['lat&lng'].shift(-1)
    dataframe['prevObservation'] = dataframe['lat&lng'].shift(1)
    dataframe['nextCreated_at'] = dataframe['created_at'].shift(-1)
    dataframe['prevCreated_at'] = dataframe['created_at'].shift(1)
    dataframe['nextMarker_distance'] = dataframe['distance'].shift(-1)
    dataframe['prevMarker_distance'] = dataframe['distance'].shift(1)
    dataframe['nextHasSkipped_relevation'] = dataframe['skipped_next_relevation'].shift(-1)
    dataframe['prevHasSkipped_relevation'] = dataframe['skipped_next_relevation'].shift(1)

    dataframe['keep_relevation'] = dataframe.apply(lambda x: \
                    utils.compute_keep_relevation(x, 12), axis=1)
    dataframe['skipped_next_relevation'] = dataframe.apply(lambda x: \
                    utils.correct_skip_relevation(x), axis=1)
    dataframe = dataframe.loc[dataframe.keep_relevation == True]


    end_shape_df = dataframe.shape[0]
    print(f'{i} - End')
    print(f'{start_shape_df} - {end_shape_df}')

print()

start_shape_df = dataframe.shape[0]
end_shape_df = 0
i = 0

while not start_shape_df == end_shape_df:
    i += 1
    print(f'{i} - Start')
    start_shape_df = dataframe.shape[0]

    dataframe['lat&lng'] = list(zip(list(dataframe.lat.round(6)), list(dataframe.lng.round(6))))
    dataframe = dataframe.sort_values(by=['provider_id', 'created_at', 'lat&lng', 'placename'])
    dataframe['nextPlacename'] = dataframe['placename'].shift(-1)
    dataframe['prevPlacename'] = dataframe['placename'].shift(1)
    dataframe['nextObservation'] = dataframe['lat&lng'].shift(-1)
    dataframe['prevObservation'] = dataframe['lat&lng'].shift(1)
    dataframe['nextCreated_at'] = dataframe['created_at'].shift(-1)
    dataframe['prevCreated_at'] = dataframe['created_at'].shift(1)
    dataframe['nextMarker_distance'] = dataframe['distance'].shift(-1)
    dataframe['prevMarker_distance'] = dataframe['distance'].shift(1)
    dataframe['nextHasSkipped_relevation'] = dataframe['skipped_next_relevation'].shift(-1)
    dataframe['prevHasSkipped_relevation'] = dataframe['skipped_next_relevation'].shift(1)

    dataframe['keep_relevation'] = dataframe.apply(lambda x: \
                    utils.compute_final_keep_relevation(x), axis=1)
    dataframe['skipped_next_relevation'] = dataframe.apply(lambda x: \
                    utils.correct_final_skip_relevation(x), axis=1)
    dataframe = dataframe.loc[dataframe.keep_relevation == True]

    end_shape_df = dataframe.shape[0]
    print(f'{i} - End')
    print(f'{start_shape_df} - {end_shape_df}')


dataframe['skipped_next_relevation'] = dataframe['skipped_next_relevation'].astype(str)
dataframe = dataframe.drop(columns=['nextObservation', 'prevObservation', 'nextCreated_at', \
    'prevCreated_at', 'nextMarker_distance', 'prevMarker_distance', 'nextHasSkipped_relevation',
    'prevHasSkipped_relevation', 'keep_relevation', 'nextPlacename', 'prevPlacename'], axis=1)

dataframe.to_csv(clean_file_path, index=False)
