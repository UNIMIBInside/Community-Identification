from imports import *

folder_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
points_dataframe = pd.read_csv(f'{folder_path}/data/points.csv')
first_relevation_marker = [list(points_dataframe.marker_label[i*9:(i+1)*9]) for i in range(0,40,2)]
first_relevation_marker = [e for row in first_relevation_marker for e in row ]
second_relevation_marker = [list(points_dataframe.marker_label[i*9:(i+1)*9]) for i in range(1,40,2)]
second_relevation_marker = [e for row in second_relevation_marker for e in row ]

def distance(row):
    return geopy.distance.geodesic((row.lat, row.lng), (row.placelat, row.placelon)).m

def relevation_close(position, referencePosition, threshold_distance):
    if not referencePosition:
        return False
    if geopy.distance.geodesic(position, referencePosition).m > threshold_distance:
        return False
    return True

def compute_keep_relevation(row, threshold_distance):
    if row.created_at == row.nextCreated_at \
        and relevation_close(row['lat&lng'], row['nextObservation'], threshold_distance) \
        and row.distance > row.nextMarker_distance:
        return False
    if row.created_at == row.prevCreated_at \
        and relevation_close(row['lat&lng'], row['prevObservation'], threshold_distance) \
        and row.distance > row.prevMarker_distance:
        return False
    return True

def correct_skip_relevation(row):
    if row.skipped_next_relevation and row.created_at == row.nextCreated_at \
        and relevation_close(row['lat&lng'], row['nextObservation'], 12) \
        and (not row.nextHasSkipped_relevation or row.nextHasSkipped_relevation == 'True - To Update'):
        return 'True - To Update'
    if row.skipped_next_relevation and row.created_at == row.prevCreated_at \
        and relevation_close(row['lat&lng'], row['prevObservation'], 12) \
        and (not row.prevHasSkipped_relevation or row.prevHasSkipped_relevation == 'True - To Update'):
        return 'True - To Update'

    return row.skipped_next_relevation

def compute_final_keep_relevation(row):
    
    row_type = '1^ Rel' if row.placename in first_relevation_marker else '2^ Rel'
    next_row_type = '1^ Rel' if row.nextPlacename in first_relevation_marker else '2^ Rel'
    prev_row_type = '1^ Rel' if row.prevPlacename in first_relevation_marker else '2^ Rel'

    if row.created_at == row.nextCreated_at and row_type == '1^ Rel' and next_row_type == '2^ Rel':
        return False
    if row.created_at == row.prevCreated_at and row_type == '1^ Rel' and prev_row_type == '2^ Rel':
        return False
    if row.created_at == row.nextCreated_at and row_type == next_row_type \
        and row.distance >= row.nextMarker_distance:
        return False
    if row.created_at == row.prevCreated_at and row_type == prev_row_type \
        and row.distance > row.prevMarker_distance:
        return False
    return True

def correct_final_skip_relevation(row):
    if row.skipped_next_relevation and row.created_at == row.nextCreated_at \
        and (not row.nextHasSkipped_relevation or row.nextHasSkipped_relevation == 'True - To Update'):
        return 'True - To Update'
    if row.skipped_next_relevation and row.created_at == row.prevCreated_at \
        and (not row.prevHasSkipped_relevation or row.prevHasSkipped_relevation == 'True - To Update'):
        return 'True - To Update'

    return row.skipped_next_relevation