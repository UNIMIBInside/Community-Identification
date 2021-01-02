from imports import *

import utils

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Insert dataset path", type=str)
parser.add_argument("-t", "--time", help="Insert frequency sampling interval", type=int, default=15)
parser.add_argument('--labelling', help="Create labelled dataframe", \
                                                dest='labelling', action='store_true')
parser.set_defaults(labelling=False)
parser.add_argument('--skipped', help="Use skipped relevation information", \
                                                dest='skipped', action='store_true')
parser.set_defaults(labelling=False)
args = parser.parse_args()

threshold_time = args.time + 1
check_skipped = args.skipped
THRESHOLD_TIME_SKIPPED = 120
THRESHOLD_GPS = 12

folder_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
file_path = folder_path + '/' + args.file
if check_skipped:
    transport_dataframe = pd.read_csv(file_path, dtype={'skipped_next_relevation': str})
else:
    transport_dataframe = pd.read_csv(file_path)

transport_dataframe = utils.prepare_dataset(transport_dataframe.copy(), check_skipped=check_skipped)

transport_dataframe['samePlace'] = transport_dataframe.apply(lambda x: \
                        utils.check_same_place(x, threshold_time, THRESHOLD_GPS), axis=1)
transport_dataframe['type'] = transport_dataframe.apply(lambda x: utils.labelling_type(x, \
    THRESHOLD_GPS, check_skipped=check_skipped, threshold_time_skipped=THRESHOLD_TIME_SKIPPED), axis=1)
if check_skipped:
    transport_dataframe = transport_dataframe.drop(columns=['prevHas_skipped_relevation'], axis=1)
transport_dataframe = utils.split_dropoff_pickup(transport_dataframe.copy())


if args.labelling:
    transport_dataframe.to_csv(folder_path + '/data/transport_labelled.csv', index=False)


transport_dataframe = utils.remove_waiting(transport_dataframe.copy())
transport_dataframe = utils.labelling_outlier(transport_dataframe.copy())

transport_dataframe.to_csv(folder_path + '/data/transport_movimento.csv', index=False)
