from imports import *


def load_vehicle_data(vehicle_path_file):

    vehicle_dataframe = pd.read_csv(vehicle_path_file)
    vehicle_dataframe['created_at'] = pd.to_datetime(vehicle_dataframe['created_at'], \
                                                                format='%Y-%m-%d %H:%M:%S')
    vehicle_dataframe['prevCreated_at'] = pd.to_datetime(vehicle_dataframe['prevCreated_at'], \
                                                                format='%Y-%m-%d %H:%M:%S')
    vehicle_dataframe['nextCreated_at'] = pd.to_datetime(vehicle_dataframe['nextCreated_at'], \
                                                                format='%Y-%m-%d %H:%M:%S')

    points_dataframe = vehicle_dataframe[['placelat', 'placelon', 'placename']]
    points_dataframe = points_dataframe.rename(columns={'placelat': 'latitude', 'placelon': 'longitude', 'placename': 'marker_label'})
    points_dataframe = points_dataframe.drop_duplicates()
    points_dataframe = points_dataframe.sort_values('marker_label')

    return vehicle_dataframe, points_dataframe


def check_input_day(day):
    if day:
        day = day.lower()
        if day in ['monday', 'lunedì']:
            day = 0
        elif day in ['tuesday', 'martedì']:
            day = 1
        elif day in ['wednesday', 'mercoledì']:
            day = 2
        elif day in ['thursday', 'giovedì']:
            day = 3
        elif day in ['friday', 'venerdì']:
            day = 4
        elif day in ['saturday', 'sabato']:
            day = 5
        elif day in ['sunday', 'domenica']:
            day = 6
        else:
            day = None
    return day


def distance(first_coord, second_coords):
    return geopy.distance.geodesic(first_coord, second_coords).m

def similarity(v1, v2):
    return cosine_similarity(v1.reshape(1, -1),v2.reshape(1, -1))[0][0]

def neighbors(points_dataframe, marker, delta=0.00475):
    marker_row = points_dataframe.loc[points_dataframe.marker_label == marker]
    marker_latitude, marker_longitude = marker_row.latitude.values[0], marker_row.longitude.values[0]
    neighbors_rows = points_dataframe.loc[(points_dataframe.latitude >= marker_latitude - delta) & \
                                        (points_dataframe.latitude <= marker_latitude + delta) & \
                                        (points_dataframe.longitude >= marker_longitude - delta) & \
                                        (points_dataframe.longitude <= marker_longitude + delta) & \
                                        (points_dataframe.marker_label != marker)]
    return list(neighbors_rows.marker_label)


def keep_valid_row(actual_movement_type, next_movement_type):
    if actual_movement_type in ['pick-up', 'first seen / pick-up', 'pickup'] \
        and next_movement_type in ['drop-off', 'last seen / drop-off', 'dropoff']:
        return True
    return False

def label_valid_transition(row, self_loop, map_marker_to_position):

    if row.provider_id == row.succ_provider_id:
        if self_loop or (not self_loop and not row.placename == row.succ_placename):
            if keep_valid_row(row.type, row.succ_type) \
                and row.nextCreated_at == row.succ_created_at:
                actual_marker_pos = map_marker_to_position[row.placename]
                next_marker_pos = map_marker_to_position[row.succ_placename]
                return f'{actual_marker_pos}-{next_marker_pos}'

    return None

def transition_probabilities(timeslot_dataframe, self_loop, n_marker, day, \
                    time_range_start, time_range_end, map_marker_to_position):

    numerosity_threshold = 0.01 if self_loop else 0.02

    if day or (not day and isinstance(day, int)):
        timeslot_dataframe = timeslot_dataframe.loc[timeslot_dataframe.created_at.map(lambda x: x.weekday()) == day]
    if time_range_start:
        timeslot_dataframe = timeslot_dataframe.loc[timeslot_dataframe.created_at.map(lambda x: x.hour) >= time_range_start]
        timeslot_dataframe = timeslot_dataframe.loc[timeslot_dataframe.created_at.map(lambda x: x.hour) <= time_range_end]

    all_next_transition = np.zeros((n_marker, n_marker))

    timeslot_dataframe['succ_provider_id'] = timeslot_dataframe['provider_id'].shift(-1)
    timeslot_dataframe['succ_placename'] = timeslot_dataframe['placename'].shift(-1)
    timeslot_dataframe['succ_type'] = timeslot_dataframe['type'].shift(-1)
    timeslot_dataframe['succ_created_at'] = timeslot_dataframe['created_at'].shift(-1)
    timeslot_dataframe['transition'] = timeslot_dataframe.apply( \
        lambda x: label_valid_transition(x, self_loop, map_marker_to_position), axis=1)
    timeslot_dataframe = timeslot_dataframe.loc[timeslot_dataframe.transition.notnull()]

    for transition, count in Counter(list(timeslot_dataframe.transition)).most_common():
        start_marker, end_marker = transition.split('-')
        all_next_transition[int(start_marker)][int(end_marker)] = count

    temp_all_next_transition = [marker_next_transition/sum(marker_next_transition) \
                            for marker_next_transition in all_next_transition]
    temp_reset_next_transition = np.array(temp_all_next_transition) >= numerosity_threshold
    all_next_transition = [[transition if temp_reset_next_transition[k][l] else 0 \
                            for l, transition in enumerate(marker_next_transition)] \
                            for k, marker_next_transition in enumerate(all_next_transition)]
    all_next_transition = np.array(all_next_transition)
    all_next_transition = [marker_next_transition/sum(marker_next_transition) \
                            for marker_next_transition in all_next_transition]
    all_next_transition = np.nan_to_num(all_next_transition)

    return all_next_transition


def old_transition_probabilities_with_selfloop(timeslot_dataframe, n_marker, day, \
    time_range_start, time_range_end, map_marker_to_position, numerosity_threshold=0.01):

    if day or (not day and isinstance(day, int)):
        timeslot_dataframe = timeslot_dataframe.loc[timeslot_dataframe.created_at.map(lambda x: x.weekday()) == day]
    if time_range_start:
        timeslot_dataframe = timeslot_dataframe.loc[timeslot_dataframe.created_at.map(lambda x: x.hour) >= time_range_start]
        timeslot_dataframe = timeslot_dataframe.loc[timeslot_dataframe.created_at.map(lambda x: x.hour) <= time_range_end]

    all_next_transition = np.zeros((n_marker, n_marker))

    for i in range(timeslot_dataframe.shape[0]):
        try:
            if timeslot_dataframe.iloc[i].provider_id == timeslot_dataframe.iloc[i+1].provider_id:
                actual_marker_label = timeslot_dataframe.iloc[i].placename
                next_marker_label = timeslot_dataframe.iloc[i+1].placename
                actual_marker_type = timeslot_dataframe.iloc[i].type
                next_marker_type = timeslot_dataframe.iloc[i+1].type
                actual_nextCreatedAt = timeslot_dataframe.iloc[i].nextCreated_at
                next_CreatedAt = timeslot_dataframe.iloc[i+1].created_at

                if keep_valid_row(actual_marker_type, next_marker_type) \
                    and actual_nextCreatedAt == next_CreatedAt:
                    actual_marker_pos = map_marker_to_position[actual_marker_label]
                    next_marker_pos = map_marker_to_position[next_marker_label]
                    all_next_transition[actual_marker_pos][next_marker_pos] += 1

        except Exception as e:
            pass

    temp_all_next_transition = [marker_next_transition/sum(marker_next_transition) \
                            for marker_next_transition in all_next_transition]
    temp_reset_next_transition = np.array(temp_all_next_transition) >= numerosity_threshold
    all_next_transition = [[transition if temp_reset_next_transition[k][l] else 0 \
                            for l, transition in enumerate(marker_next_transition)] \
                            for k, marker_next_transition in enumerate(all_next_transition)]
    all_next_transition = np.array(all_next_transition)
    all_next_transition = [marker_next_transition/sum(marker_next_transition) \
                            for marker_next_transition in all_next_transition]
    all_next_transition = np.nan_to_num(all_next_transition)

    return all_next_transition

def old_transition_probabilities_without_selfloop(timeslot_dataframe, n_marker, day, \
    time_range_start, time_range_end, map_marker_to_position, numerosity_threshold=0.02):

    if day or (not day and isinstance(day, int)):
        timeslot_dataframe = timeslot_dataframe.loc[timeslot_dataframe.created_at.map(lambda x: x.weekday()) == day]
    if time_range_start:
        timeslot_dataframe = timeslot_dataframe.loc[timeslot_dataframe.created_at.map(lambda x: x.hour) >= time_range_start]
        timeslot_dataframe = timeslot_dataframe.loc[timeslot_dataframe.created_at.map(lambda x: x.hour) <= time_range_end]

    all_next_transition = np.zeros((n_marker, n_marker))

    for i in range(timeslot_dataframe.shape[0]):
        try:
            if timeslot_dataframe.iloc[i].provider_id == timeslot_dataframe.iloc[i+1].provider_id:
                actual_marker_label = timeslot_dataframe.iloc[i].placename
                next_marker_label = timeslot_dataframe.iloc[i+1].placename
                if not actual_marker_label == next_marker_label:
                    actual_marker_type = timeslot_dataframe.iloc[i].type
                    next_marker_type = timeslot_dataframe.iloc[i+1].type
                    actual_nextCreatedAt = timeslot_dataframe.iloc[i].nextCreated_at
                    next_CreatedAt = timeslot_dataframe.iloc[i+1].created_at

                    if keep_valid_row(actual_marker_type, next_marker_type) \
                        and actual_nextCreatedAt == next_CreatedAt:
                        actual_marker_pos = map_marker_to_position[actual_marker_label]
                        next_marker_pos = map_marker_to_position[next_marker_label]
                        all_next_transition[actual_marker_pos][next_marker_pos] += 1

        except Exception as e:
            pass

    temp_all_next_transition = [marker_next_transition/sum(marker_next_transition) \
                            for marker_next_transition in all_next_transition]
    temp_reset_next_transition = np.array(temp_all_next_transition) >= numerosity_threshold
    all_next_transition = [[transition if temp_reset_next_transition[k][l] else 0 \
                            for l, transition in enumerate(marker_next_transition)] \
                            for k, marker_next_transition in enumerate(all_next_transition)]
    all_next_transition = np.array(all_next_transition)
    all_next_transition = [marker_next_transition/sum(marker_next_transition) \
                            for marker_next_transition in all_next_transition]
    all_next_transition = np.nan_to_num(all_next_transition)

    return all_next_transition


def metrics_valid(metrics='modularity', alpha=1, beta=1, gamma=1):
    if metrics == 'modularity' and alpha == 0:
        return False
    if metrics == 'distance' and beta == 0:
        return False
    if metrics == 'map_embedding' and gamma == 0:
        return False

    return True

def choose_metrics_gain(metrics='modularity', modularity=False, distance=False, map_embedding=False):
    if metrics == 'modularity':
        alpha, beta, gamma = 5, 1, 1
    elif metrics == 'distance':
        alpha, beta, gamma = 1, 10, 1
    elif metrics == 'map_embedding':
        alpha, beta, gamma = 1, 1, 5
    else:
        alpha, beta, gamma = 1, 1, 1

    if modularity or distance or map_embedding:
        alpha = 0 if not modularity else alpha
        beta = 0 if not distance else beta
        gamma = 0 if not map_embedding else gamma

    return alpha, beta, gamma


def compute_gain_modularity(cX, cY, timeslot, weight_edges, degree_vertices, E):
    deltaQ = 1/(2*E)
    gain_couple_list = [weight_edges[i][timeslot][j] + weight_edges[j][timeslot][i] - \
                        ((degree_vertices[i][timeslot]*degree_vertices[j][timeslot])/(2*E)) \
                        for i in cX for j in cY]
    deltaQ *= sum(gain_couple_list)
    return deltaQ

def compute_gain_distance(cX, cY, distance_matrix, E):
    nX = len(cX) if len(cX) else 1
    nY = len(cY) if len(cY) else 1
    deltaD = 1/((E/30)*nX*nY)
    gain_couple_list = [1 / (1 + distance_matrix[i][j]) for i in cX for j in cY]
    deltaD *= sum(gain_couple_list)
    return deltaD

def compute_gain_map_embedding(cX, cY, similarity_ME_matrix, E):
    nX = len(cX) if len(cX) else 1
    nY = len(cY) if len(cY) else 1
    deltaM = 1/((5*E)*nX*nY)
    gain_couple_list = [similarity_ME_matrix[i][j] for i in cX for j in cY]
    deltaM *= sum(gain_couple_list)
    return deltaM


def compute_modularity(community, timeslot, weight_edges, degree_vertices, E):
    deltaQ = 1/(2*E)
    summation = 0
    for i in community:
        for j in community[community.index(i)+1:]:
            summation += weight_edges[i][timeslot][j] + weight_edges[j][timeslot][i] - ((degree_vertices[i][timeslot]*degree_vertices[j][timeslot])/(2*E))
    deltaQ *= summation
    return deltaQ

def compute_distance(community, distance_matrix):
    summation = 0
    for i in community:
        for j in community[community.index(i)+1:]:
            summation += 1 / (1 + distance_matrix[i][j])
    divider = (len(community)*len(community)-len(community))/2 if len(community) > 1 else 1
    summation /= divider
    return summation

def compute_map_embedding(community, similarity_ME_matrix):
    summation = 0
    for i in community:
        for j in community[community.index(i)+1:]:
            summation += similarity_ME_matrix[i][j]
    divider = (len(community)*len(community)-len(community))/2 if len(community) > 1 else 1
    summation /= divider
    return summation




def compute_gain(x, y, removed_cluster, C, weight_edges, degree_vertices, E, \
                alpha, beta, gamma, distance_matrix, similarity_ME_matrix, timeslot):

    if x == y or x in removed_cluster or y in removed_cluster:
        return -999

    cX_vertices = [vertex for vertex, cluster in enumerate(C) if cluster == x]
    cY_vertices = [vertex for vertex, cluster in enumerate(C) if cluster == y]
    gain_modularity_weekday = compute_gain_modularity(cX_vertices, cY_vertices, 0, \
                                                    weight_edges, degree_vertices, E)
    gain_modularity_weekend = compute_gain_modularity(cX_vertices, cY_vertices, 1, \
                                                    weight_edges, degree_vertices, E)
    if timeslot == 'weekday':
        gain_modularity = gain_modularity_weekday
    elif timeslot == 'weekend':
        gain_modularity = gain_modularity_weekend
    else:
        gain_modularity = gain_modularity_weekday + gain_modularity_weekend

    gain_distance = compute_gain_distance(cX_vertices, cY_vertices, distance_matrix, E)
    gain_map_embedding = compute_gain_map_embedding(cX_vertices, cY_vertices, \
                                                            similarity_ME_matrix, E)
    gain_total = alpha * gain_modularity + beta * gain_distance + gamma * gain_map_embedding
    return gain_total

def new_hierarchical_aggregation(C, weight_edges, degree_vertices, K, E, \
        alpha, beta, gamma, distance_matrix, similarity_ME_matrix, timeslot):

    iteration = 0
    removed_cluster = []

    while True:
        iteration += 1

        x_max, y_max = -1, -1
        max_gain_total = -999

        gain_list = [compute_gain(x, y, removed_cluster, C, weight_edges, degree_vertices, E, \
                            alpha, beta, gamma, distance_matrix, similarity_ME_matrix, timeslot) \
                            for x in range(K) for y in range(x+1, K)]

        max_gain_total = np.array(gain_list).max()
        index_max = np.array(gain_list).argmax()
        x_max = int(index_max/K)
        y_max = index_max%K

        if x_max == -1 or y_max == -1 or max_gain_total <= 0:
            #print(x_max, y_max, max_gain_total)
            break

        C = [x_max if e == y_max else e for e in C]
        removed_cluster.append(y_max)

    return C





def hierarchical_aggregation(C, weight_edges, degree_vertices, K, E, \
        alpha, beta, gamma, distance_matrix, similarity_ME_matrix, timeslot):

    iteration = 0
    removed_cluster = []

    while True:
        iteration += 1

        x_max, y_max = -1, -1
        max_gain_total = -999

        for x in range(K):
            for y in range(x+1, K):
                if x == y or x in removed_cluster or y in removed_cluster:
                    continue

                cX_vertices = [vertex for vertex, cluster in enumerate(C) if cluster == x]
                cY_vertices = [vertex for vertex, cluster in enumerate(C) if cluster == y]
                gain_modularity_weekday = compute_gain_modularity(cX_vertices, cY_vertices, 0, \
                                                                weight_edges, degree_vertices, E)
                gain_modularity_weekend = compute_gain_modularity(cX_vertices, cY_vertices, 1, \
                                                                weight_edges, degree_vertices, E)
                if timeslot == 'weekday':
                    gain_modularity = gain_modularity_weekday
                elif timeslot == 'weekend':
                    gain_modularity = gain_modularity_weekend
                else:
                    gain_modularity = gain_modularity_weekday + gain_modularity_weekend

                gain_distance = compute_gain_distance(cX_vertices, cY_vertices, distance_matrix, E)
                gain_map_embedding = compute_gain_map_embedding(cX_vertices, cY_vertices, \
                                                                        similarity_ME_matrix, E)
                gain_total = alpha * gain_modularity + beta * gain_distance + gamma * gain_map_embedding

                if gain_total > max_gain_total:
                    max_gain_total = gain_total
                    x_max = x
                    y_max = y

        if x_max == -1 or y_max == -1 or max_gain_total <= 0:
            #print(x_max, y_max, max_gain_total)
            break

        C = [x_max if e == y_max else e for e in C]
        removed_cluster.append(y_max)

    return C

def reallocation_cluster_under_threshold(C, weight_edges, degree_vertices, K, E, \
                alpha, beta, gamma, distance_matrix, similarity_ME_matrix, neighbors_matrix, \
                timeslot, map_marker_to_position, min_element_threshold=4):

    continue_loop = True
    iteration = 0
    community_iteration = [C]
    temp_C = C.copy()
    #print(community_iteration)

    while continue_loop:
        continue_loop = False
        iteration += 1
        temp_C = temp_C.copy()
        possible_shift = []
        #print(f'ITERATION {iteration}')

        for cluster_index, n_marker in Counter(temp_C).most_common()[::-1]:
            if n_marker >= min_element_threshold:
                break

            cluster_vertices = [vertex for vertex, cluster in enumerate(temp_C) if cluster == cluster_index]

            for vertex in cluster_vertices:
                vertex_neighbors_label = neighbors_matrix[vertex]
                vertex_next_cluster = [temp_C[map_marker_to_position[neighbor_label]] for neighbor_label in vertex_neighbors_label]
                vertex_next_cluster = list(set(vertex_next_cluster))

                if len(vertex_next_cluster) == 1:
                    possible_shift.append((vertex, vertex_next_cluster[0], 100))
                else:
                    cluster_max = -1
                    max_gain_total = -999

                    for cluster_neighbor in vertex_next_cluster:
                        if cluster_index == cluster_neighbor:
                            continue

                        cY_vertices = [v for v, cluster in enumerate(temp_C) if cluster == cluster_neighbor]
                        gain_modularity_weekday = compute_gain_modularity([vertex], cY_vertices, 0, weight_edges, degree_vertices, E)
                        gain_modularity_weekend = compute_gain_modularity([vertex], cY_vertices, 1, weight_edges, degree_vertices, E)
                        if timeslot == 'weekday':
                            gain_modularity = gain_modularity_weekday
                        elif timeslot == 'weekend':
                            gain_modularity = gain_modularity_weekend
                        else:
                            gain_modularity = gain_modularity_weekday + gain_modularity_weekend

                        gain_distance = compute_gain_distance([vertex], cY_vertices, distance_matrix, E)
                        gain_map_embedding = compute_gain_map_embedding([vertex], cY_vertices, similarity_ME_matrix, E)
                        gain_total = alpha * gain_modularity + beta * gain_distance + gamma * gain_map_embedding

                        if gain_total > max_gain_total:
                            max_gain_total = gain_total
                            cluster_max = cluster_neighbor

                    possible_shift.append((vertex, cluster_max, max_gain_total))

        possible_shift = sorted(possible_shift, key=lambda x: x[2], reverse=True)
        if possible_shift:
            marker_index_to_shift = possible_shift[0][0]
            marker_new_cluster = possible_shift[0][1]
            temp_C[marker_index_to_shift] = marker_new_cluster

        community_iteration.append(temp_C)
        #print(temp_C)
        #print()
        if community_iteration.count(temp_C) <= 1:
            continue_loop = True

    #print(community_iteration[-1])
    return community_iteration[-1]

def reduction_cluster_number(cluster_threshold, C, weight_edges, degree_vertices, K, E, \
                alpha, beta, gamma, distance_matrix, similarity_ME_matrix, timeslot):

    iteration = 0
    removed_cluster = []

    while iteration < K - cluster_threshold:
        iteration += 1

        x_max, y_max = -1, -1
        max_gain_total = -999

        for x in range(K):
            for y in range(K):
                if x == y or x in removed_cluster or y in removed_cluster:
                    continue

                cX_vertices = [vertex for vertex, cluster in enumerate(C) if cluster == x]
                cY_vertices = [vertex for vertex, cluster in enumerate(C) if cluster == y]
                gain_modularity_weekday = compute_gain_modularity(cX_vertices, cY_vertices, 0, \
                                                                weight_edges, degree_vertices, E)
                gain_modularity_weekend = compute_gain_modularity(cX_vertices, cY_vertices, 1, \
                                                                weight_edges, degree_vertices, E)
                if timeslot == 'weekday':
                    gain_modularity = gain_modularity_weekday
                elif timeslot == 'weekend':
                    gain_modularity = gain_modularity_weekend
                else:
                    gain_modularity = gain_modularity_weekday + gain_modularity_weekend

                gain_distance = compute_gain_distance(cX_vertices, cY_vertices, distance_matrix, E)
                gain_map_embedding = compute_gain_map_embedding(cX_vertices, cY_vertices, \
                                                                        similarity_ME_matrix, E)
                gain_total = alpha * gain_modularity + beta * gain_distance + gamma * gain_map_embedding

                if gain_total > max_gain_total:
                    max_gain_total = gain_total
                    x_max = x
                    y_max = y

        C = [x_max if e == y_max else e for e in C]
        removed_cluster.append(y_max)

    #print(C)
    return C

def reallocation_isolated_node(C, weight_edges, degree_vertices, K, E, alpha, beta, gamma, \
    distance_matrix, similarity_ME_matrix, neighbors_matrix, timeslot, map_marker_to_position):

    continue_loop = True
    iteration = 0
    community_iteration = [C]
    temp_C = C.copy()
    #print(community_iteration)

    while continue_loop:
        continue_loop = False
        iteration += 1
        temp_C = temp_C.copy()
        possible_shift = []
        #print(f'ITERATION {iteration}')

        for cluster_index, _ in Counter(temp_C).most_common():

            cluster_vertices = [vertex for vertex, cluster in enumerate(temp_C) if cluster == cluster_index]

            for vertex in cluster_vertices:
                vertex_neighbors_label = neighbors_matrix[vertex]
                vertex_next_cluster = [temp_C[map_marker_to_position[neighbor_label]] for neighbor_label in vertex_neighbors_label]
                vertex_next_cluster = list(set(vertex_next_cluster))

                if cluster_index in vertex_next_cluster:
                    continue

                elif len(vertex_next_cluster) == 1:
                    possible_shift.append((vertex, vertex_next_cluster[0], 100))
                else:
                    cluster_max = -1
                    max_gain_total = -999

                    for cluster_neighbor in vertex_next_cluster:
                        if cluster_index == cluster_neighbor:
                            continue

                        cY_vertices = [v for v, cluster in enumerate(temp_C) if cluster == cluster_neighbor]
                        gain_modularity_weekday = compute_gain_modularity([vertex], cY_vertices, 0, weight_edges, degree_vertices, E)
                        gain_modularity_weekend = compute_gain_modularity([vertex], cY_vertices, 1, weight_edges, degree_vertices, E)
                        if timeslot == 'weekday':
                            gain_modularity = gain_modularity_weekday
                        elif timeslot == 'weekend':
                            gain_modularity = gain_modularity_weekend
                        else:
                            gain_modularity = gain_modularity_weekday + gain_modularity_weekend

                        gain_distance = compute_gain_distance([vertex], cY_vertices, distance_matrix, E)
                        gain_map_embedding = compute_gain_map_embedding([vertex], cY_vertices, similarity_ME_matrix, E)
                        gain_total = alpha * gain_modularity + beta * gain_distance + gamma * gain_map_embedding

                        if gain_total > max_gain_total:
                            max_gain_total = gain_total
                            cluster_max = cluster_neighbor

                    possible_shift.append((vertex, cluster_max, max_gain_total))

        possible_shift = sorted(possible_shift, key=lambda x: x[2], reverse=True)
        if possible_shift:
            marker_index_to_shift = possible_shift[0][0]
            marker_new_cluster = possible_shift[0][1]
            temp_C[marker_index_to_shift] = marker_new_cluster

        community_iteration.append(temp_C)
        #print(temp_C)
        #print()
        if community_iteration.count(temp_C) <= 1:
            continue_loop = True

    #print(community_iteration[-1])
    return community_iteration[-1]


def save_cluster_solution(save_file_path, points_dataframe, cluster_field, K, \
                                        delta_lat = 0.00230, delta_lng = 0.00220):

    if K <= 27:
        color_spot = ['00', '99', 'ff']
    elif K <= 64:
        color_spot = ['00', '33', '99', 'ff']
    elif K <= 125:
        color_spot = ['00', '33', '99', 'cc', 'ff']
    elif K <= 216:
        color_spot = ['00', '33', '66', '99', 'cc', 'ff']
    else:
        color_spot = ['00', '33', '66', '80', '99', 'cc', 'ff']
    cluster_colors = [f'#{r}{g}{b}' for r in color_spot for g in color_spot for b in color_spot]

    #if K == 10:
    #    cluster_colors = ['#000000', '#990000', '#009900', '#000099', '#999900', \
    #                        '#990099', '#009999', '#ff9900', '#00ff99', '#9900ff']

    latitute_mean = points_dataframe.latitude.mean()
    longitude_mean = points_dataframe.longitude.mean()

    m = folium.Map(location=[latitute_mean, longitude_mean], tiles='OpenStreetMap', \
                                            zoom_start=13, min_zoom=12, max_zoom=16)
    for i in range(0, len(points_dataframe)):
        latitude_coord = points_dataframe.iloc[i].latitude
        longitude_coord = points_dataframe.iloc[i].longitude
        rectangle_points = [(latitude_coord + delta_lat, longitude_coord + delta_lng), \
                            (latitude_coord - delta_lat, longitude_coord + delta_lng), \
                            (latitude_coord - delta_lat, longitude_coord - delta_lng), \
                            (latitude_coord + delta_lat, longitude_coord - delta_lng)]
        folium.Polygon(rectangle_points, fill=True, fill_opacity=0.6, \
                            color=cluster_colors[int(points_dataframe.iloc[i][cluster_field])],
                            radius=3).add_to(m)

    #img_data = m._to_png(5)
    #img = Image.open(io.BytesIO(img_data))
    #img.save(save_file_path)
    m.save(save_file_path)

def save_community_metrics(step_name, points_dataframe, C, weight_edges, degree_vertices, \
                                                K, E, distance_matrix, similarity_ME_matrix):

    communities_modularity = []
    communities_distance = []
    communities_map_embedding = []

    for k in range(K):
        cK_vertices = [vertex for vertex, cluster in enumerate(C) if cluster == k]
        cK_modularity = compute_modularity(cK_vertices, 0, weight_edges, degree_vertices, E) + \
                        compute_modularity(cK_vertices, 1, weight_edges, degree_vertices, E)
        communities_modularity.append(cK_modularity)
        cK_distance = compute_distance(cK_vertices, distance_matrix)
        communities_distance.append(cK_distance)
        cK_map_embedding = compute_map_embedding(cK_vertices, similarity_ME_matrix)
        communities_map_embedding.append(cK_map_embedding)

    points_dataframe[f'cluster_{step_name}_modularity'] = 0
    points_dataframe[f'cluster_{step_name}_distance'] = 0
    points_dataframe[f'cluster_{step_name}_map_embedding'] = 0
    for k in range(K):
        points_dataframe.loc[points_dataframe.cluster_HA == k, \
                    f'cluster_{step_name}_modularity'] = communities_modularity[k]
        points_dataframe.loc[points_dataframe.cluster_HA == k, \
                    f'cluster_{step_name}_distance'] = communities_distance[k]
        points_dataframe.loc[points_dataframe.cluster_HA == k, \
                    f'cluster_{step_name}_map_embedding'] = communities_map_embedding[k]

    return points_dataframe
