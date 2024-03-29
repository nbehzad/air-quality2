import numpy as np
from numpy import genfromtxt
from multiprocessing import Pool
import pandas as pd
import math
import matplotlib.pyplot as plt


class Dataset(object):
    def __init__(self, filename, feature_stride_size=None):
        self.filename = filename
        self.feature_stride_size = feature_stride_size

    def read_csv_data(self):
        self.data_matrix = genfromtxt(self.filename, delimiter=',', skip_header=1)
        self.data_matrix = self.data_matrix[:, 4:]

    def read_csv_with_cyclic_features(self):
        df = pd.read_csv(self.filename)
        df['Hour_cycle'] = 0
        df['Month_cycle'] = 0
        df['u_component'] = 0
        df['v_component'] = 0

        for i in range(len(df)):
            df['u_component'].iloc[i] = df['Wind_speed'].iloc[i] * math.sin((270 - (df['Wind_direction']
                                                                                    .iloc[i] * (360 / 16))) * (
                                                                                        math.pi / 180))
            df['v_component'].iloc[i] = df['Wind_speed'].iloc[i] * math.cos((270 - (df['Wind_direction']
                                                                                    .iloc[i] * (360 / 16))) * (
                                                                                        math.pi / 180))
            df['Hour_cycle'].iloc[i] = math.sin(2 * math.pi * 60 * df['Hour'].iloc[i] / 1440)
            df['Month_cycle'].iloc[i] = math.sin(2 * math.pi * df['Month'].iloc[i] / 12)

        df.drop(['Wind_speed', 'Wind_direction', 'Year', 'Month', 'Day', 'Hour'], axis=1, inplace=True)
        cols = df.columns.tolist()
        cols = cols[-4:] + cols[0:-4]
        df = df[cols]

        self.data_matrix = df.to_numpy()

        x = df['u_component'].values  # returns a numpy array
        x_scaled = (x - x.min(0)) / (x.ptp(0))

        fig, axis = plt.subplots(2)
        axis[0].plot(np.arange(0, 30), x_scaled[0:30])
        axis[1].plot(np.arange(0, 30), x[0:30])

        plt.show()

    def normalize(self):
        self.data_matrix = (self.data_matrix - self.data_matrix.min(0)) / self.data_matrix.ptp(0)

    def set_categorized_features(self, col_index, cat_number):
        label_matrix = np.eye(cat_number)
        cat_matrix = []
        for i in range(self.data_matrix.shape[0]):
            cat_matrix.append(label_matrix[int(self.data_matrix[i, col_index] - 1)])

        cat_matrix = np.array(cat_matrix)
        self.data_matrix = np.hstack((self.data_matrix[:, 0:col_index], cat_matrix, self.data_matrix[:, col_index + 1:]))
        print(self.data_matrix.shape)

    def get_train_test(self, cut_point_index=35040, window_size=168, predict_period=24):
        x_train = []
        x_test = []
        y_train = []
        y_test = []

        end_boundary = self.data_matrix.shape[0] - window_size - predict_period + 1
        for i in range(end_boundary):
            next_index = i + window_size
            if i < cut_point_index:
                x_train.append(self.data_matrix[i:next_index])
                y_train.append(self.data_matrix[next_index:next_index + predict_period, -1])
            else:
                x_test.append(self.data_matrix[i:next_index])
                y_test.append(self.data_matrix[next_index:next_index + predict_period, -1])

        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        '''
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]

        
        indices = np.arange(x_test.shape[0])
        np.random.shuffle(indices)
        x_test = x_test[indices]
        y_test = y_test[indices]
        '''
        return x_train, y_train, x_test, y_test

    def get_up_scale_data(self, data, stride=None):
        up_scale_data = []
        for i in range(data.shape[0]):
            stride_data = [np.hstack((np.min(data[i, j:j+stride, :], axis=0),
                                      np.max(data[i, j:j+stride, :], axis=0),
                                      np.mean(data[i, j:j+stride, :], axis=0),
                                      np.std(data[i, j:j+stride, :], axis=0)))
                           for j in range(0, data.shape[1], stride)]
            up_scale_data.append(stride_data)

        return np.array(up_scale_data)

    def calc_up_scale_data(self, data_frame):
        strided_data = [np.hstack((np.min(data_frame[j:j + self.feature_stride_size, :], axis=0),
                                   np.max(data_frame[j:j + self.feature_stride_size, :], axis=0),
                                   np.mean(data_frame[j:j + self.feature_stride_size, :], axis=0),
                                   np.std(data_frame[j:j + self.feature_stride_size, :], axis=0)))
                        for j in range(0, data_frame.shape[0], self.feature_stride_size)]
        return strided_data

    def get_up_scale_data_parallel(self, data, stride=None):
        result = []
        self.feature_stride_size = stride
        with Pool(processes=8) as pool:
            result = pool.map(self.calc_up_scale_data, data)

        return np.array(result)

    def save_data_matrix(self, file_name):
        if self.data_matrix is not None and self.data_matrix.any():
            np.savetxt(file_name, self.data_matrix, delimiter=',')



'''
ds = Dataset('./data/Basaksehir_full_data.csv')
ds.read_csv_data()
# ds.normalize()
ds.set_categorized_features(4, 16)
ds.get_train_test()
'''