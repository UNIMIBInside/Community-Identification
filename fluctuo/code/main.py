"""
File per lo scaricamento delle rilevazioni di bici, moto ed automobili dallle API Fluctuo.

N_MARKER: numero di marker su cui fare le per ogni esecuzione della funzione (max 10/min).
N_ITERATION: numero di iterazioni su cui lavora ciascun token.

TOKEN_N: token per l'esecuzione delle richieste alle API.
FOLDER_N = nome della cartella in cui i dati, per il token di riferimento, vengono salvati.
OFFSET_N: offset del marker da cui partire a fare le richieste.
INDEX_N: offset per alternare la porzione di marker su chi fare le richieste per un dato token.
N_OPERATION_N: conteggio esecuzione funzione per dato token; divide i file ogni 8 iterazioni.
PATH_DATAFRAME_N: nome dei file incrementali dei dati ottenuti.
"""

from imports import *


FOLDER_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
N_MARKER = 10
N_ITERATION = 2


TOKEN_1 = 'INSERIRE TOKEN QUI'
FOLDER_1 = 'token1'
OFFSET_1 = 0
N_OPERATION_1 = -1
INDEX_1 = -1
PATH_DATAFRAME_1 = 0

TOKEN_2 = 'INSERIRE TOKEN QUI'
FOLDER_2 = 'token2'
OFFSET_2 = 18
N_OPERATION_2 = -1
INDEX_2 = -1
PATH_DATAFRAME_2 = 0


try:
    os.mkdir(FOLDER_PATH+'/data/'+FOLDER_1)
except FileExistsError:
    pass
try:
    os.mkdir(FOLDER_PATH+'/data/'+FOLDER_2)
except FileExistsError:
    pass

points_dataframe = pd.read_csv(FOLDER_PATH+'/data/points.csv')


def get_fluctuo_data_token_1():
    """Funzione per lo scaricamento dei dati di N_MARKER"""

    global N_OPERATION_1
    global INDEX_1
    global PATH_DATAFRAME_1
    N_OPERATION_1 += 1

    headers = {'content-type': 'application/json',}
    params = (('access_token', TOKEN_1),)

    INDEX_1 += 1
    INDEX_1 = INDEX_1 % N_ITERATION
    relevation_datetime = datetime.datetime.now()
    if N_OPERATION_1 % 8 == 0:
        PATH_DATAFRAME_1 = relevation_datetime.strftime("%Y-%m-%d_%H")
    relevation_datetime = relevation_datetime.strftime("%Y-%m-%d %H:%M:%S")

    try:
        file_path = FOLDER_1+'/iteration_'+PATH_DATAFRAME_1+'.csv'
        new_movement_dataframe = pd.read_csv(file_path)
    except FileNotFoundError:
        new_movement_dataframe = pd.DataFrame()


    group_dataframe = points_dataframe[OFFSET_1+INDEX_1*N_MARKER:OFFSET_1+(INDEX_1+1)*N_MARKER]

    print(relevation_datetime, 'Token1')
    #print(group_dataframe)
    for i in range(group_dataframe.shape[0]):
        latitude = group_dataframe.iloc[i].latitude
        longitude = group_dataframe.iloc[i].longitude
        placename = group_dataframe.iloc[i].marker_label
        data = '{"query":"query ($lat: Float!, $lng: Float!) {\\n vehicles(lat: $lat, lng: $lng) {\\n\\t\\tid\\ntype\\nattributes\\nlat\\nlng\\nprovider {name} }\\n}","variables":{"lat":'+str(latitude)+',"lng":'+str(longitude)+'}}'

        response = requests.post('https://flow-api.fluctuo.com/v1', \
                                    headers=headers, params=params, data=data)
        print(response)
        if not response.status_code == 200:
            print(response.status_code)
            return True
        response_json = response.json()
        vehicles = response_json['data']['vehicles']
        new_row = pd.DataFrame.from_records(vehicles)
        relevation_dataframe = pd.DataFrame()
        relevation_dataframe = relevation_dataframe.append(new_row)
        relevation_dataframe = relevation_dataframe.rename(columns={'id':'provider_id'})
        relevation_dataframe['placename'] = placename
        relevation_dataframe['created_at'] = relevation_datetime
        new_movement_dataframe = pd.concat([new_movement_dataframe, relevation_dataframe], axis=0)

    new_movement_dataframe.to_csv(FOLDER_1+'/iteration_'+PATH_DATAFRAME_1+'.csv', index=False)
    return True


def get_fluctuo_data_token_2():
    """Funzione per lo scaricamento dei dati di N_MARKER"""

    global N_OPERATION_2
    global INDEX_2
    global PATH_DATAFRAME_2
    N_OPERATION_2 += 1

    headers = {'content-type': 'application/json',}
    params = (('access_token', TOKEN_2),)

    INDEX_2 += 1
    INDEX_2 = INDEX_2 % N_ITERATION
    relevation_datetime = datetime.datetime.now()
    if N_OPERATION_2 % 8 == 0:
        PATH_DATAFRAME_2 = relevation_datetime.strftime("%Y-%m-%d_%H")
    relevation_datetime = relevation_datetime.strftime("%Y-%m-%d %H:%M:%S")

    try:
        file_path = FOLDER_2+'/iteration_'+PATH_DATAFRAME_2+'.csv'
        new_movement_dataframe = pd.read_csv(file_path)
    except FileNotFoundError:
        new_movement_dataframe = pd.DataFrame()

    group_dataframe = points_dataframe[OFFSET_2+INDEX_2*N_MARKER:OFFSET_2+(INDEX_2+1)*N_MARKER]

    print(relevation_datetime, 'Token2')
    #print(group_dataframe)
    for i in range(group_dataframe.shape[0]):
        latitude = group_dataframe.iloc[i].latitude
        longitude = group_dataframe.iloc[i].longitude
        placename = group_dataframe.iloc[i].marker_label
        data = '{"query":"query ($lat: Float!, $lng: Float!) {\\n vehicles(lat: $lat, lng: $lng) {\\n\\t\\tid\\ntype\\nattributes\\nlat\\nlng\\nprovider {name} }\\n}","variables":{"lat":'+str(latitude)+',"lng":'+str(longitude)+'}}'

        response = requests.post('https://flow-api.fluctuo.com/v1', \
                                    headers=headers, params=params, data=data)
        #print(response)
        if not response.status_code == 200:
            print(response.status_code)
            return True
        response_json = response.json()
        vehicles = response_json['data']['vehicles']
        new_row = pd.DataFrame.from_records(vehicles)
        relevation_dataframe = pd.DataFrame()
        relevation_dataframe = relevation_dataframe.append(new_row)
        relevation_dataframe = relevation_dataframe.rename(columns={'id':'provider_id'})
        relevation_dataframe['placename'] = placename
        relevation_dataframe['created_at'] = relevation_datetime
        new_movement_dataframe = pd.concat([new_movement_dataframe, relevation_dataframe], axis=0)

    new_movement_dataframe.to_csv(FOLDER_2+'/iteration_'+PATH_DATAFRAME_2+'.csv', index=False)
    return True


schedule.every().hours.at(':00').do(get_fluctuo_data_token_1)
schedule.every().hours.at(':00').do(get_fluctuo_data_token_2)
schedule.every().hours.at(':02').do(get_fluctuo_data_token_1)
schedule.every().hours.at(':02').do(get_fluctuo_data_token_2)

schedule.every().hours.at(':30').do(get_fluctuo_data_token_1)
schedule.every().hours.at(':30').do(get_fluctuo_data_token_2)
schedule.every().hours.at(':32').do(get_fluctuo_data_token_1)
schedule.every().hours.at(':32').do(get_fluctuo_data_token_2)

while True:
    schedule.run_pending()
    time.sleep(1)
