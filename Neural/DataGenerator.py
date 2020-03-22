import h5py
import numpy as np
from numba import jit
from tensorflow.keras.utils import Sequence
from preprocess import PITCH_RANGE


class DataGenerator(Sequence):
    def __init__(self, file_name, batch_size, training_ratio, is_test=False):
        self.file_name = file_name
        self.batch_size = batch_size
        self.training_ratio = training_ratio
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

            if self.is_test:
                start_index += self.get_length_spec(False)
                end_index += self.get_length_spec(False)
            else:
                end_index = np.clip(end_index, 0, self.get_length())

            data = h5f['data'][start_index:end_index]
            label = h5f['label'][start_index:end_index]

        return data, label

    def get_length(self):
        length = 0
        with h5py.File(self.file_name + '.h5', 'r') as h5f:
            ratio = self.training_ratio
            if self.is_test:
                ratio = 1.0 - ratio

            length = np.floor(h5py.Dataset.len(h5f['data']) * ratio)

        return length

    def get_length_spec(self, is_test):
        length = 0
        with h5py.File(self.file_name + '.h5', 'r') as h5f:
            ratio = self.training_ratio
            if is_test:
                ratio = 1.0 - ratio

            length = np.floor(h5py.Dataset.len(h5f['data']) * ratio)

        return length


class DataGenerator88(Sequence):
    def __init__(self, file_name, batch_size, training_ratio, is_test=False):
        self.file_name = file_name
        self.batch_size = batch_size
        self.training_ratio = training_ratio
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

            if self.is_test:
                start_index += self.get_length_spec(False)
                end_index += self.get_length_spec(False)
            else:
                if end_index >= self.get_length():
                    end_index = self.get_length() - 1

            data = h5f['data'][start_index:end_index]
            label = h5f['label'][start_index:end_index]

            # =====
            for i in range(PITCH_RANGE):
                final_label['out{}'.format(i)] = label[:, i + PITCH_RANGE]

        return data, final_label

    def get_length(self):
        length = 0
        with h5py.File(self.file_name + '.h5', 'r') as h5f:
            ratio = self.training_ratio
            if self.is_test:
                ratio = 1.0 - ratio

            length = np.floor(h5py.Dataset.len(h5f['data']) * ratio)

        return int(length)

    def get_length_spec(self, is_test):
        length = 0
        with h5py.File(self.file_name + '.h5', 'r') as h5f:
            ratio = self.training_ratio
            if is_test:
                ratio = 1.0 - ratio

            length = np.floor(h5py.Dataset.len(h5f['data']) * ratio)

        return int(length)
