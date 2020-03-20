import h5py
import numpy as np
from numba import jit
from tensorflow.keras.utils import Sequence
from preprocess import PITCH_RANGE


class DataGenerator(Sequence):
    def __init__(self, file_name, batch_size, is_test=False):
        self.file_name = file_name
        self.batch_size = batch_size
        self.is_test = is_test
        self.indices = np.arange(self.__len__())

    def __len__(self):
        return int(np.ceil(self.get_length() / self.batch_size))

    def __getitem__(self, index):
        data = []
        label = []

        with h5py.File(self.file_name + '.h5', 'r') as h5f:
            start_index = index * self.batch_size
            end_index = (index + 1) * self.batch_size

            if not self.is_test:
                data = h5f['train_data'][start_index:end_index]
                label = h5f['train_label'][start_index:end_index]
            else:
                data = h5f['test_data'][start_index:end_index]
                label = h5f['test_label'][start_index:end_index]

        return data, label

    def get_length(self):
        length = 0
        with h5py.File(self.file_name + '.h5', 'r') as h5f:
            length = h5py.Dataset.len(h5f['train_data'])

        return length


class DataGenerator88(Sequence):
    def __init__(self, file_name, batch_size, is_test=False):
        self.file_name = file_name
        self.batch_size = batch_size
        self.is_test = is_test
        self.indices = np.arange(self.__len__())

    def __len__(self):
        return int(np.ceil(self.get_length() / self.batch_size))

    def __getitem__(self, index):
        data = []
        label = []
        final_label = {}

        with h5py.File(self.file_name + '.h5', 'r') as h5f:
            start_index = index * self.batch_size
            end_index = (index + 1) * self.batch_size

            if not self.is_test:
                data = h5f['train_data'][start_index:end_index]
                label = h5f['train_label'][start_index:end_index]
                #print("=", data.shape, label.shape)
            else:
                data = h5f['test_data'][start_index:end_index]
                label = h5f['test_label'][start_index:end_index]

            # =====
            '''if self.is_test:
                print('+++++++++++++++++++++', index, self.get_length(), start_index, end_index)
            else:
                print('=====================', index)
            '''
            for i in range(PITCH_RANGE):
                one_hot = self.get_one_hot_arr(i, label)
                # print(label.shape[0], one_hot.shape)
                final_label['out{}'.format(i)] = one_hot

        return data, final_label

    def get_length(self):
        length = 0
        with h5py.File(self.file_name + '.h5', 'r') as h5f:
            if not self.is_test:
                length = h5py.Dataset.len(h5f['train_data'])
            else:
                length = h5py.Dataset.len(h5f['test_data'])

        return length

    def get_one_hot_arr(self, pitch, label):
        one_hot_arr = []
        for col in range(label.shape[0]):
            one_hot_arr.append(self.label_to_one_hot(pitch, label[col]))
        #print(len(one_hot_arr))

        return np.asarray(one_hot_arr)

    def label_to_one_hot(self, pitch, label):
        if label[pitch] == 1:  # Onset
            return np.asarray([1, 0, 0])
        elif label[pitch + PITCH_RANGE] == 1:  # Still
            return np.asarray([0, 1, 0])
        else:  # None
            return np.asarray([0, 0, 1])
