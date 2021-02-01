from imports import *
import utils


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help="Insert vehicle dataset path", type=str, required=True)
parser.add_argument("-e", "--embedding", help="Insert map embedding pickle path", type=str, required=True)
parser.add_argument("-c", "--cluster_target", help="Insert target number for cluster solution", type=int, default=10)
parser.add_argument('--no_selfloop', help="Use transition matrix without self-loop", \
                                                dest='self_loop', action='store_false')
parser.set_defaults(self_loop=True)

parser.add_argument('--load_matrix', help="Use to load transition matrix", \
                                            dest='load_matrix', action='store_true')

parser.set_defaults(load_matrix=False)

parser.add_argument("-m", "--metrics", \
    help="Insert predominant metrics. Possible value are: modularity, distance, map_embedding", \
    type=str, default='modularity')
parser.add_argument('--modularity', help="Use gain by modularity", \
                                    dest='modularity', action='store_true')

parser.set_defaults(modularity=False)
parser.add_argument('--distance', help="Use gain by distance", \
                                    dest='distance', action='store_true')

parser.set_defaults(distance=False)
parser.add_argument('--map_embedding', help="Use gain by map_embedding", \
                                    dest='map_embedding', action='store_true')

parser.set_defaults(map_embedding=False)

parser.add_argument("-t", "--timeslot", \
    help="Insert timeslot division. Possible value are: global, weekday, weekend", \
    type=str, default='global')
parser.add_argument("-ts", "--time_range_start", help="Insert time interval start", type=int)
parser.add_argument("-te", "--time_range_end", help="Insert time interval end", type=int)
parser.add_argument("--day", help="Insert day of analysis", type=str)

args = parser.parse_args()


cluster_threshold = args.cluster_target
metric_predominant = args.metrics
timeslot = args.timeslot
time_range_start = args.time_range_start
time_range_end = args.time_range_end
day = args.day
load_matrix = args.load_matrix

if metric_predominant not in ['modularity', 'distance', 'map_embedding']:
    print('Error, possible value for --metrics are: modularity, distance, map_embedding')
    exit(1)
if timeslot not in ['global', 'weekday', 'weekend']:
    print('Error, possible value for --timeslot are: global, weekday, weekend')
    exit(1)
if cluster_threshold <= 0:
    print('Error, value for --cluster_target must be positive')
    exit(1)
if (time_range_start and time_range_start <= 0) or (time_range_end and time_range_end <=0):
    print('Error, value for time range must be positive')
    exit(1)
if time_range_start and time_range_end and time_range_start > time_range_end:
    print('Error, time range end must be greater than start')
    exit(1)

time_range_end = time_range_end if time_range_end else time_range_start
folder_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
vehicle_file_path = folder_path + '/' + args.dataset
map_embedding_file_path = folder_path + '/' + args.embedding
self_loop = args.self_loop
self_loop_save = 'self_loop' if self_loop else 'no_self_loop'
transition_matrix_path = folder_path + f'/clustering-algorithm/transition_matrix_{self_loop_save}.pickle'

alpha, beta, gamma = utils.choose_metrics_gain(metric_predominant, \
                    args.modularity, args.distance, args.map_embedding)
if not utils.metrics_valid(metric_predominant, alpha, beta, gamma):
    print('Error, metrics predominant is not selected')
    exit(1)
if day:
    day = utils.check_input_day(day)
    if not day and not isinstance(day, int):
        print('Error, value for day of analysis is not valid')
        exit(1)
    if day in [5, 6] and not timeslot == 'weekend':
        print('Error, saturday or sunday must be used with timeslot weekend')
        exit(1)
    if day in [0, 1, 2, 3, 4] and not timeslot == 'weekday':
        print('Error, day from monday to friday must be used with timeslot weekday')
        exit(1)


if os.path.isdir(folder_path + '/results') is False:
    os.mkdir(folder_path + '/results')

# Caricamento dataset
print('Load dataset...')
#start = time.time()
vehicle_dataframe, points_dataframe = utils.load_vehicle_data(vehicle_file_path)
#end = time.time()
#print(end-start)

# Remap tra marker label e loro posizione nel dataframe
marker_label_list = list(points_dataframe.marker_label.unique())
marker_position_list = list(range(len(marker_label_list)))
map_marker_to_position = dict(list(zip(marker_label_list, marker_position_list)))
map_position_to_marker = dict(list(zip(marker_position_list, marker_label_list)))


# Calcolo matrici distanze punti, similarità map embedding e marker vicini
print('Compute support matrix...')
#start = time.time()
points_dataframe['coords'] = list(zip(list(points_dataframe.latitude.round(6)), \
                                        list(points_dataframe.longitude.round(6))))
distance_matrix = np.array([[utils.distance(points_dataframe.iloc[i].coords, \
                                            points_dataframe.iloc[j].coords) \
                                    for j in range(points_dataframe.shape[0])] \
                                    for i in range(points_dataframe.shape[0])])

map_embdeddings = pickle.load(open(map_embedding_file_path, 'rb'))
similarity_ME_matrix = np.array([[utils.similarity(map_embdeddings[i], map_embdeddings[j]) \
                                                for j in range(points_dataframe.shape[0])] \
                                                for i in range(points_dataframe.shape[0])])

neighbors_matrix = [utils.neighbors(points_dataframe, points_dataframe.iloc[i].marker_label) \
                                                        for i in range(points_dataframe.shape[0])]
#end = time.time()
#print(end-start)

# Calcolo matrice delle transizioni veicoli
#start = time.time()
if not load_matrix:
    print('Compute transition matrix...')
    transition_matrix = np.array([[]]*points_dataframe.shape[0])
    for day_type in ['weekday', 'weekend']:

        saturday_dataframe = vehicle_dataframe.loc[vehicle_dataframe.created_at.map(lambda x: x.weekday()) == 5]
        sunday_dataframe = vehicle_dataframe.loc[vehicle_dataframe.created_at.map(lambda x: x.weekday()) == 6]
        weekend_dataframe = pd.concat([saturday_dataframe, sunday_dataframe])
        if day_type == 'weekday':
            timeslot_dataframe = vehicle_dataframe.drop(weekend_dataframe.index)
        else:
            timeslot_dataframe = weekend_dataframe

        timeslot_t_matrix = utils.transition_probabilities(timeslot_dataframe, self_loop, \
            points_dataframe.shape[0], day, time_range_start, time_range_end, map_marker_to_position)

        if not transition_matrix[0].size:
            transition_matrix = [np.r_[transition_matrix[k], t_matrix] \
                        for k, t_matrix in enumerate(timeslot_t_matrix)]
        else:
            transition_matrix = [np.r_['0,2', transition_matrix[k], t_matrix] \
                        for k, t_matrix in enumerate(timeslot_t_matrix)]

    transition_matrix = np.array(transition_matrix)
    with open(transition_matrix_path, 'wb') as f: pickle.dump(transition_matrix, f)

else:
    print('Load transition matrix...')
    transition_matrix_path = folder_path + f'/clustering-algorithm/transition_matrix_{self_loop_save}.pickle'
    transition_matrix = pickle.load(open(transition_matrix_path,'rb'))
#end = time.time()
#print(end-start)


# Setup iniziale soluzione clustering
vertices = list(points_dataframe.marker_label)
edges = np.array([[[1 if element else 0 for element in row] for row in rows] \
                                                for rows in transition_matrix])
weight_edges = transition_matrix.copy()
out_degree_vertices = np.array([[sum(row) for row in rows] for rows in edges])
in_degree_vertices = np.array([[sum(row) for row in rows] for rows in edges.transpose()])
degree_vertices = out_degree_vertices + in_degree_vertices
N = len(vertices)
E = len(vertices) * len(vertices)
K = len(vertices)
C = list(range(N))



# STEP 1: Aggregazione gerarchica dei marker
print('Hierarchical Aggregation...')
#start = time.time()
C = utils.hierarchical_aggregation(C, weight_edges, degree_vertices, K, E, \
            alpha, beta, gamma, distance_matrix, similarity_ME_matrix, timeslot)
#end = time.time()
#print(end-start)

## Rietichettatura cluster
remap_cluster = {}
for k, cluster_count in enumerate(Counter(C).most_common()):
    remap_cluster[cluster_count[0]] = k
C = [remap_cluster[element] for element in C]
points_dataframe['cluster_HA'] = C
K = len(set(C))

## Salvataggio immagine e calcolo valori metriche
save_image_file_path = folder_path + \
    f'/clustering-algorithm/results/clustering_{self_loop_save}_{metric_predominant}_{timeslot}_step1_HA.html'
utils.save_cluster_solution(save_image_file_path, points_dataframe, 'cluster_HA', K)

point_dataframe = utils.save_community_metrics('HA', points_dataframe, C, \
        weight_edges, degree_vertices, K, E, distance_matrix, similarity_ME_matrix)

# STEP 2: Rimozione cluster con numero di elementi sotto soglia
print('Cut cluster under threshold...')
#start = time.time()
C = utils.reallocation_cluster_under_threshold(C, weight_edges, degree_vertices, \
                K, E, alpha, beta, gamma, distance_matrix, similarity_ME_matrix, \
                                neighbors_matrix, timeslot, map_marker_to_position)
#end = time.time()
#print(end-start)

## Rietichettatura cluster
remap_cluster = {}
for k, cluster_count in enumerate(Counter(C).most_common()):
    remap_cluster[cluster_count[0]] = k
C = [remap_cluster[element] for element in C]
points_dataframe['cluster_CT'] = C
K = len(set(C))

## Salvataggio immagine e calcolo valori metriche
save_image_file_path = folder_path + \
    f'/clustering-algorithm/results/clustering_{self_loop_save}_{metric_predominant}_{timeslot}_step2_CT.html'
utils.save_cluster_solution(save_image_file_path, points_dataframe, 'cluster_CT', K)

point_dataframe = utils.save_community_metrics('CT', points_dataframe, C, \
        weight_edges, degree_vertices, K, E, distance_matrix, similarity_ME_matrix)



# STEP 3: Riduzione numero cluster al numero target
print('Reduction number of cluster...')
#start = time.time()
C = utils.reduction_cluster_number(cluster_threshold, C, weight_edges, degree_vertices, K, E, \
                    alpha, beta, gamma, distance_matrix, similarity_ME_matrix, timeslot)
#end = time.time()
#print(end-start)

## Rietichettatura cluster
remap_cluster = {}
for k, cluster_count in enumerate(Counter(C).most_common()):
    remap_cluster[cluster_count[0]] = k
C = [remap_cluster[element] for element in C]
points_dataframe['cluster_RC'] = C
K = len(set(C))

## Salvataggio immagine e calcolo valori metriche
save_image_file_path = folder_path + \
    f'/clustering-algorithm/results/clustering_{self_loop_save}_{metric_predominant}_{timeslot}_step3_RC.html'
utils.save_cluster_solution(save_image_file_path, points_dataframe, 'cluster_RC', K)

point_dataframe = utils.save_community_metrics('RC', points_dataframe, C, \
        weight_edges, degree_vertices, K, E, distance_matrix, similarity_ME_matrix)



# STEP 4: Riallocazione nodi isolati
print('Reallocation isolated marker...')
#start = time.time()
C = utils.reallocation_isolated_node(C, weight_edges, degree_vertices, K, E, \
                    alpha, beta, gamma, distance_matrix, similarity_ME_matrix, \
                    neighbors_matrix, timeslot, map_marker_to_position)
#end = time.time()
#print(end-start)

## Rietichettatura cluster
remap_cluster = {}
for k, cluster_count in enumerate(Counter(C).most_common()):
    remap_cluster[cluster_count[0]] = k
C = [remap_cluster[element] for element in C]
points_dataframe['cluster_RI'] = C
K = len(set(C))

## Salvataggio immagine e calcolo valori metriche
save_image_file_path = folder_path + \
    f'/clustering-algorithm/results/clustering_{self_loop_save}_{metric_predominant}_{timeslot}_step4_RI.html'
utils.save_cluster_solution(save_image_file_path, points_dataframe, 'cluster_RI', K)

point_dataframe = utils.save_community_metrics('RI', points_dataframe, C, \
        weight_edges, degree_vertices, K, E, distance_matrix, similarity_ME_matrix)



# Salvataggio dataset riassuntivo
save_dataframe_file_path = folder_path + \
    f'/clustering-algorithm/results/clustering_{self_loop_save}_{metric_predominant}_{timeslot}.csv'
points_dataframe.to_csv(save_dataframe_file_path, index=False, float_format='%.8f')
