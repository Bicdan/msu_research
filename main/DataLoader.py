import os
import struct
import numpy as np
import pandas as pd
import plotly.graph_objects as go

class DataHolder:
    emotion_to_index = {
        'gnev': 0,
        'gore': 1,
        'liubov': 2,
        'neitralino': 3,
        'onvrashenie': 4,
        'prezrenie': 5,
        'radost': 6,
        'trevoga': 7,
        'udivlenie': 8,
        'uzas': 9,
        'vina': 10,
        'zastenchivost': 11
    }

    emotion_to_russian = {
        'gnev': 'гнев',
        'gore': 'горе',
        'liubov': 'любовь',
        'neitralino': 'нейтрально',
        'onvrashenie': 'отвращение',
        'prezrenie': 'презрение',
        'radost': 'радость',
        'trevoga': 'тревога',
        'udivlenie': 'удивление',
        'uzas': 'ужас',
        'vina': 'вина',
        'zastenchivost': 'застенчивость'
    }

    index_to_russian = {
        0: 'гнев',
        1: 'горе',
        2: 'любовь',
        3: 'нейтрально',
        4: 'отвращение',
        5: 'презрение',
        6: 'радость',
        7: 'тревога',
        8: 'удивление',
        9: 'ужас',
        10: 'вина',
        11: 'застенчивость'
    }

    # Для того, чтобы убрать отрезки с простоем
    index_to_range = {
        0: [(10, 160), (225, 350)],
        1: [(60, 365)],
        2: [(7, 120), (160, 245)],
        3: [(48, 157), (176, 271)],
        4: [(65, 180), (200, 290)],
        5: [(8, 93), (113, 197), (220, 306)],
        6: [(44, 172), (201, 303)],
        7: [(7, 287)],
        8: [(5, 170), (210, 360)],
        9: [(57, 267)],
        10: [(45, 144), (160, 245)],
        11: [(176, 440)],
        12: [(36, 330)],
        13: [(7, 240)],
        14: [(6, 119), (135, 227)],
        15: [(47, 180), (206, 330)],
        16: [(68, 186), (212, 320)],
        17: [(42, 144), (165, 264)],
        18: [(5, 260)],
        19: [(6, 109), (132, 213)],
        20: [(4, 146), (172, 284)],
        21: [(55, 340)],
        22: [(4, 210)],
        23: [(46, 364)],
        24: [(9, 96), (131, 230)],
        25: [(61, 404)],
        26: [(6, 172), (217, 355)],
        27: [(4, 235)],
        28: [(11, 300)],
        29: [(79, 214), (250, 365)],
        30: [(3, 147), (189, 323)],
        31: [(3, 120), (150, 238)],
        32: [(3, 117), (130, 225)],
        33: [(10, 270)],
        34: [(88, 190), (214, 305)],
        35: [(50, 245)],
        36: [(112, 250), (284, 395)],
        37: [(7, 175), (229, 373)],
        38: [(6, 171), (206, 332)],
        39: [(64, 213), (234, 339)],
        40: [(70, 202), (238, 346)],
        41: [(4, 181)],
        42: [(42, 242)],
        43: [(3, 197)],
        44: [(85, 287)],
        45: [(8, 250)],
        46: [(3, 212)],
        47: [(108, 394)]
    }

    # This is for eeg parsing
    names = ['vina', 'onvrashenie', 'gnev', 'vina', 'udivlenie', 'zastenchivost',
             'radost', 'liubov', 'uzas', 'onvrashenie', 'trevoga', 'liubov',
            'trevoga', 'uzas', 'neitralino', 'udivlenie', 'udivlenie', 'gore',
            'prezrenie', 'uzas', 'radost', 'neitralino', 'zastenchivost', 'zastenchivost',
            'liubov', 'gore', 'gnev', 'gore', 'onvrashenie', 'udivlenie',
            'gnev', 'neitralino', 'radost', 'prezrenie', 'prezrenie', 'trevoga',
            'zastenchivost', 'vina', 'liubov', 'radost', 'gore', 'onvrashenie',
            'trevoga', 'vina', 'uzas', 'prezrenie', 'gnev', 'neitralino']
    
    # This is for eeg parsing
    labels = [12, 10, 26, 16, 2, 13, 12, 16, 23, 30, 21,
              28, 9, 11, 5, 14, 10, 27, 7, 3, 20, 9, 5, 25,
              8, 7, 2, 3, 18, 22, 6, 29, 32, 11, 31, 1, 17,
              4, 4, 8, 15, 6, 13, 24, 15, 19, 14, 17]
    

class DataLoader:
    """
    Main loader for raw data
    """
    def __init__(self):
        pass
    
    def read_projections(self, folder_path):
        """
        Read list of .csv from folder_path
        Return dataframes, raw_data, stat_data, stat_shared_data
        """
        dataframes = []
        raw_data = []
        stat_data = []
        stat_shared_data = []
        index_to_filename = {}

        for i, filename in enumerate(os.listdir(folder_path)):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                index_to_filename[i] = filename
                emotion = filename.split('_')[-1].split('.')[0]
                emotion_idx = DataHolder.emotion_to_index[emotion]
                
                # Чтение CSV
                df = pd.read_csv(file_path)
                
                # Подготовка признакового пространства
                features = []
                stat_features = []

                for row in df:
                    feature_row = np.array(df[row])
                    stat_features.append(np.mean(feature_row))
                    stat_features.append(np.std(feature_row))
                    features.append(np.array(df[row]))
                for i in range(0, len(feature_row) - 50, 25):
                    stat_shared_features = []
                    for row in df:
                        feature_row = np.array(df[row])
                        shared_feature_row = feature_row[i:i+50]
                        stat_shared_features.append(np.mean(shared_feature_row))
                        stat_shared_features.append(np.std(shared_feature_row))
                    stat_shared_features = np.array(stat_shared_features)
                    stat_shared_data.append((stat_shared_features, emotion_idx))
                
                features = np.array(features)
                stat_features = np.array(stat_features)

                dataframes.append(df)
                raw_data.append((features, emotion_idx))
                stat_data.append((stat_features, emotion_idx))

        return dataframes, raw_data, stat_data, stat_shared_data
    
    def get_preprocessed_data(self, raw_data):
        index_to_features = {}
        for i, data in enumerate(raw_data):
            matrix = data[0]
            emotion_index = data[1]
            data_ranges = DataHolder.index_to_range[i]

            if emotion_index not in index_to_features:
                index_to_features[emotion_index] = None
            
            for data_range in data_ranges:
                start, end = data_range
                submatrix = matrix[:, start:(end+1)]
                if index_to_features[emotion_index] is None:
                    index_to_features[emotion_index] = submatrix
                else:
                    index_to_features[emotion_index] = np.hstack((index_to_features[emotion_index], submatrix))
        return index_to_features
    
    def draw_merged_data(self, index_to_features):
        for index in index_to_features:
            matrix = index_to_features[index]
            emotion_russian = DataHolder.index_to_russian[index]
            fig = go.Figure()
            for component in range(matrix.shape[0]):
                func = matrix[component]
                trace = go.Scatter(x=np.arange(len(func)), y=func, mode='lines', name=f'Компонента {component+1}')
                fig.add_trace(trace)
            fig.update_layout(title=f'График проекции, эмоция: {emotion_russian}', xaxis_title='Время', yaxis_title='Значение компоненты')
            fig.show()
            
    def read_eeg(self, file_path):
        """
        Read single .byt file from file_path
        Return raw_data, an array of (idx, time, label, values)
        """
        format_string = 'd'*9
        data_size = struct.calcsize(format_string)
        raw_data = []

        with open(file_path, 'rb') as file:
            while True:
                raw_time = file.read(8)
                if not raw_time:
                    break
                time = struct.unpack('d', raw_time)[0]

                raw_label = file.read(4)
                if not raw_label:
                    break
                label = struct.unpack('i', raw_label)[0]

                data = file.read(data_size)
                if not data:
                    break
                try:
                    values = struct.unpack(format_string, data)
                except:
                    continue
                raw_data.append((len(raw_data), time, label, values))
        
        return raw_data
    
    def get_non_zero_labels_with_time(self, eeg_raw_data):
        non_zero_labels_with_time = []
        for idx, time, label, _ in eeg_raw_data:
            if label != 0:
                non_zero_labels_with_time.append((idx, label, time))
        return non_zero_labels_with_time
    
    def parse_eeg(self, eeg_raw_data, max_delta_time = 10):
        """
        Return a list of (idx_start, idx_end, emotion) for an eeg_raw_data
        """
        res = []
        is_start, time_start = False, 0
        idx_start, idx_end = 0, 0
        cur_emotion, cur_index = None, 0

        non_zero_labels = self.get_non_zero_labels_with_time(eeg_raw_data)
        for idx, label, time in non_zero_labels:
            if label == 1:
                is_start = True
                idx_start, time_start = idx, time
            else:
                if is_start and time - time_start <= max_delta_time:
                    is_start = False
                    idx_end = idx
                    while DataHolder.labels[cur_index] != label:
                        cur_index += 1
                    cur_emotion = DataHolder.emotion_to_russian[DataHolder.names[cur_index]]
                    res.append((idx_start, idx_end, cur_emotion))
        return res
    
    def draw_eeg(self, file_to_meta, file_to_data, record_name):
        record_meta = file_to_meta[record_name]
        record_data = file_to_data[record_name]

        for idx in range(len(record_meta)):
            start, end, emotion = record_meta[idx]
            times = np.zeros(end-start)
            time_start = record_data[start][1]
            matrix = np.zeros((9, end-start))

            for i in range(start, end):
                _, time, _, values = record_data[i]
                for j, value in enumerate(values):
                    times[i-start] = time - time_start
                    matrix[j][i-start] = value

            fig = go.Figure()
            for component in range(matrix.shape[0]):
                func = matrix[component]
                trace = go.Scatter(x=times, y=func, mode='lines', name=f'Сигнал {component+1}')
                fig.add_trace(trace)
            fig.update_layout(title=f'График ЭЭГ области Вернике, эмоция: {emotion}',
                            xaxis_title='Время, сек.',
                            yaxis_title='Значение ЭЭГ сигнала')
            fig.show()