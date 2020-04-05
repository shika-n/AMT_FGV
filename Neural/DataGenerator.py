import h5py
import numpy as np
from numba import jit
from tensorflow.keras.utils import Sequence
from preprocess import PITCH_RANGE


class DataGenerator(Sequence):
    def __init__(self, file_name, batch_size, used_ratio, is_test=False):
        self.file_name = file_name
        self.batch_size = batch_size
        self.used_ratio = used_ratio
        self.is_test = is_test
        self.indices = np.arange(self.get_length())  # training/test length

    def __len__(self):
        return int(np.ceil(self.get_length() / self.batch_size))

    def __getitem__(self, index):
        data = []
        label = []

        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size

        offset = 0
        if self.is_test:
            offset = self.get_training_length()

        indices_to_use = np.sort(self.indices[start_index:end_index]) + offset

        with h5py.File(self.file_name + '.h5', 'r') as h5f:
            data = h5f['data'][:, indices_to_use].T
            label = h5f['label'][:, indices_to_use].T

        return data, label

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def get_length(self):
        length = 0
        with h5py.File(self.file_name + '.h5', 'r') as h5f:
            ratio = self.used_ratio

            if self.is_test:
                ratio = 1.0 - ratio  # test data ratio

            length = np.floor(h5py.Dataset.len(h5f['data'][0]) * ratio)

        return int(length)

    def get_training_length(self):
        length = 0
        with h5py.File(self.file_name + '.h5', 'r') as h5f:
            length = np.floor(h5py.Dataset.len(h5f['data'][0]) * self.used_ratio)

        return int(length)


class DataGenerator88(DataGenerator):
    def __init__(self, file_name, batch_size, used_ratio, is_test=False):
        super().__init__(file_name, batch_size, used_ratio, is_test)

    def __getitem__(self, index):
        data = []
        label = []
        final_label = {}

        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size

        offset = 0
        if self.is_test:
            offset = self.get_training_length()

        indices_to_use = np.sort(self.indices[start_index:end_index]) + offset

        with h5py.File(self.file_name + '.h5', 'r') as h5f:
            data = h5f['data'][:, indices_to_use].T
            label = h5f['label'][:, indices_to_use]

            # =====
            for i in range(PITCH_RANGE):
                final_label['out{}'.format(i)] = label[i + PITCH_RANGE, :]

        return data, final_label
