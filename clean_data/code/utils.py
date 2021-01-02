from imports import *


def shift_position_valide(row, row_compare):
    """Funzione per dividere sequenza posizioni rilevazioni al cambio di provider_id"""
    if row_compare == 'prev' and \
        (not row.prev_provider_id or not row.provider_id == row.prev_provider_id):
        return None
    if row_compare == 'next' and \
        (not row.next_provider_id or not row.provider_id == row.next_provider_id):
        return None
    if row_compare == 'prev':
        return row.prevObservation
    if row_compare == 'next':
        return row.nextObservation
    
def shift_datetime_valide(observation, shift_datetime):
    """Funzione per dividere sequenza datetime rilevazioni al cambio di provider_id"""
    if not observation:
        return None
    else:
        return shift_datetime


def distance(position, reference_position):
    """Funzione per il calcolo della distanza tra due punti"""
    if not reference_position:
        return 0
    return geopy.distance.geodesic(position, reference_position).m

def difference_time(referenceObservation, actual_datetime, following_datetime):
    """Funzione per il calcolo del delta tra due datetime"""
    if not referenceObservation or actual_datetime is pd.NaT or following_datetime is pd.NaT:
        return 0
    return ((following_datetime - actual_datetime).seconds)/60 + \
                ((following_datetime - actual_datetime).days)*24*60


def prepare_dataset(transport_dataframe, check_skipped=False):
    """Funzione di preparazione dataset e creazione campi di supporto"""

    transport_dataframe['prev_provider_id'] = transport_dataframe['provider_id'].shift(1)
    transport_dataframe['next_provider_id'] = transport_dataframe['provider_id'].shift(-1)
    if check_skipped:
        transport_dataframe['prevHas_skipped_relevation'] = transport_dataframe['skipped_next_relevation'].shift(1)
    transport_dataframe['lat&lng'] = transport_dataframe['lat&lng'].apply(lambda x: eval(x))
    transport_dataframe = transport_dataframe.replace({pd.np.nan: None})

    transport_dataframe['prevObservation'] = transport_dataframe['lat&lng'].shift(1)
    transport_dataframe['prevObservation'] = transport_dataframe.apply(lambda x: \
                                                shift_position_valide(x, 'prev'), axis=1)
    transport_dataframe['nextObservation'] = transport_dataframe['lat&lng'].shift(-1)
    transport_dataframe['nextObservation'] = transport_dataframe.apply(lambda x: \
                                                shift_position_valide(x, 'next'), axis=1)
    transport_dataframe = transport_dataframe.drop(columns=\
                                        ['prev_provider_id', 'next_provider_id'], axis=1) 

    transport_dataframe['prevCreated_at'] = transport_dataframe['created_at'].shift(1)
    transport_dataframe['prevCreated_at'] = transport_dataframe.apply(lambda x: \
                shift_datetime_valide(x['prevObservation'], x['prevCreated_at']), axis=1)
    transport_dataframe['nextCreated_at'] = transport_dataframe['created_at'].shift(-1)
    transport_dataframe['nextCreated_at'] = transport_dataframe.apply(lambda x: \
                shift_datetime_valide(x['nextObservation'], x['nextCreated_at']), axis=1)
    transport_dataframe['created_at'] = pd.to_datetime(transport_dataframe['created_at'], \
                                                                    format='%Y-%m-%d %H:%M:%S')
    transport_dataframe['prevCreated_at'] = pd.to_datetime(transport_dataframe['prevCreated_at'], \
                                                                    format='%Y-%m-%d %H:%M:%S')
    transport_dataframe['nextCreated_at'] = pd.to_datetime(transport_dataframe['nextCreated_at'], \
                                                                    format='%Y-%m-%d %H:%M:%S')

    transport_dataframe['prevDistance'] = transport_dataframe.apply(lambda x: \
                                        distance(x['lat&lng'], x['prevObservation']), axis=1)
    transport_dataframe['nextDistance'] = transport_dataframe.apply(lambda x: \
                                        distance(x['lat&lng'], x['nextObservation']), axis=1)
    transport_dataframe['prevTimeDiff'] = transport_dataframe.apply(lambda x: \
        difference_time(x['prevObservation'], x['prevCreated_at'], x['created_at']), axis=1)
    transport_dataframe['nextTimeDiff'] = transport_dataframe.apply(lambda x: \
        difference_time(x['nextObservation'], x['created_at'], x['nextCreated_at']), axis=1)

    return transport_dataframe


def check_same_place(row, threshold_time, threshold_gps):
    """Funzione per etichettare le rilevazioni sui mezzi a distanza del tempo, per periodo 
    non ho informazioni, e il mezzo è nello stesso punto. Probabili rilevazioni mancate."""
    nextObservation, nextDistance = row['nextObservation'], row['nextDistance']
    current_datetime, next_datetime = row['created_at'], row['nextCreated_at']

    if not nextObservation:
        return False
    elif ((next_datetime - current_datetime).seconds)/60 > threshold_time \
        and nextDistance < threshold_gps and nextDistance > 0:
        return True
    else:
        return False

def labelling_type(row, threshold_gps, check_skipped=False, threshold_time_skipped=120):
    """Funzione di etichettatura per le rilevazioni"""

    if not row.prevObservation and not row.nextObservation:
        return 'first seen / last seen'
  
    elif not row.prevObservation:
        if geopy.distance.geodesic(row['lat&lng'], row['nextObservation']).m < threshold_gps:
            return 'first seen / waiting'
        else:
            return 'first seen / pick-up'
  
    elif not row.nextObservation:
        if geopy.distance.geodesic(row['lat&lng'], row['prevObservation']).m < threshold_gps:
            return 'last seen / waiting'
        else:
            return 'last seen / drop-off'
  
    else:

        if geopy.distance.geodesic(row['lat&lng'], row['prevObservation']).m < threshold_gps \
            and geopy.distance.geodesic(row['lat&lng'], row['nextObservation']).m < threshold_gps:
            if row.samePlace:
                return 'waiting long'
            else:
                return 'waiting'

        elif geopy.distance.geodesic(row['lat&lng'], row['prevObservation']).m < threshold_gps \
            and not geopy.distance.geodesic(row['lat&lng'], row['nextObservation']).m < threshold_gps:
            if check_skipped and row.skipped_next_relevation == 'True' and row.nextTimeDiff > threshold_time_skipped:
                return 'waiting (skipped)'
            else:
                return 'pick-up'

        elif not geopy.distance.geodesic(row['lat&lng'], row['prevObservation']).m < threshold_gps \
            and geopy.distance.geodesic(row['lat&lng'], row['nextObservation']).m < threshold_gps:
            if check_skipped and row.prevHas_skipped_relevation == 'True' and row.prevTimeDiff > threshold_time_skipped:
                return 'waiting (skipped)'
            else:
                return 'drop-off'

        elif not geopy.distance.geodesic(row['lat&lng'], row['prevObservation']).m < threshold_gps \
            and not geopy.distance.geodesic(row['lat&lng'], row['nextObservation']).m < threshold_gps:
            if check_skipped and row.skipped_next_relevation == 'True' and row.nextTimeDiff > threshold_time_skipped \
                and row.prevHas_skipped_relevation == 'True' and row.prevTimeDiff > threshold_time_skipped:
                return 'waiting (skipped)'
            elif check_skipped and row.skipped_next_relevation == 'True' and row.nextTimeDiff > threshold_time_skipped:
                return 'drop-off'
            elif check_skipped and row.prevHas_skipped_relevation == 'True' and row.prevTimeDiff > threshold_time_skipped:
                return 'pick-up'
            else:
                return 'drop-off / pick-up'
        
        else:
            return 'Default'


def split_dropoff_pickup(transport_dataframe):
    """Funzione per dividere le rilevazioni drop-off / pick-up in due righe una per ciascun tipo"""

    dropoff_pickup_df = transport_dataframe.loc[transport_dataframe.type == 'drop-off / pick-up']
    temporary_split_df = pd.DataFrame()

    for i in range(dropoff_pickup_df.shape[0]):
        row = dropoff_pickup_df.iloc[i].copy()
        row.type = 'drop-off'
        temporary_split_df = temporary_split_df.append(row)
        row.type = 'pick-up'
        temporary_split_df = temporary_split_df.append(row)

    transport_dataframe = transport_dataframe.loc[transport_dataframe.type != 'drop-off / pick-up']
    transport_dataframe = transport_dataframe.append(temporary_split_df)
    transport_dataframe = transport_dataframe.sort_values(by=['provider_id', 'created_at', 'type'])

    return transport_dataframe

def remove_waiting(transport_dataframe):
    """Funzione per la rimozione delle rilevazioni di waiting"""
    transport_dataframe = transport_dataframe.loc[transport_dataframe['type'] != 'waiting']
    transport_dataframe = transport_dataframe.loc[transport_dataframe['type'] != 'first seen / waiting']
    transport_dataframe = transport_dataframe.loc[transport_dataframe['type'] != 'last seen / waiting']
    transport_dataframe = transport_dataframe.loc[transport_dataframe['type'] != 'first seen / last seen']
    return transport_dataframe


def label_outlier_type(row, fence_lowP, fence_highP, fence_lowD, fence_highD):
    """Funzione di calcolo etichetta outlier temporali"""
    if row.type in ['pick-up', 'first seen / pick-up'] \
        and (row.nextTimeDiff < fence_lowP or row.nextTimeDiff > fence_highP):
        return 'pick-up outlier'
    elif row.type in ['drop-off', 'last seen / drop-off'] \
        and (row.prevTimeDiff < fence_lowD or row.prevTimeDiff > fence_highD):
        return 'drop-off outlier'
    else:
        return row.type

def labelling_outlier(transport_dataframe):
    """Funzione di rietichettatura degli outlier temporali"""

    pickup_dataframe = transport_dataframe.loc[(transport_dataframe.type != 'drop-off') & \
                                            (transport_dataframe.type != 'waiting long') & \
                                            (transport_dataframe.type != 'waiting (skipped)') & \
                                            (transport_dataframe.type != 'last seen / drop-off')]
    q1P = float(pd.DataFrame(pickup_dataframe.nextTimeDiff).quantile(0.25))
    q3P = float(pd.DataFrame(pickup_dataframe.nextTimeDiff).quantile(0.75))
    iqrP = q3P - q1P
    fence_lowP = q1P - (1.5*iqrP)
    fence_highP = q3P + (1.5*iqrP)

    dropoff_dataframe = transport_dataframe.loc[(transport_dataframe.type != 'pick-up') & \
                                            (transport_dataframe.type != 'waiting long') & \
                                            (transport_dataframe.type != 'waiting (skipped)') & \
                                            (transport_dataframe.type != 'first seen / pick-up')]
    q1D = float(pd.DataFrame(dropoff_dataframe.prevTimeDiff).quantile(0.25))
    q3D = float(pd.DataFrame(dropoff_dataframe.prevTimeDiff).quantile(0.75))
    iqrD = q3D - q1D
    fence_lowD = q1D - (1.5*iqrD)
    fence_highD = q3D + (1.5*iqrD)

    transport_dataframe['type'] = transport_dataframe.apply(lambda x: \
        label_outlier_type(x, fence_lowP, fence_highP, fence_lowD, fence_highD), axis=1)

    return transport_dataframe
