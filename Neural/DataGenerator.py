import h5py
import numpy as np
from numba import jit
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import Callback
from preprocess import PITCH_RANGE


class FoldGen:
    def __init__(self, file_name):
        self.file_name = file_name
        self.indices = np.arange(self._get_sample_count())

        self._shuffle_indices()

    def _shuffle_indices(self):
        np.random.shuffle(self.indices)

    def _get_sample_count(self):
        length = 0
        with h5py.File(self.file_name + '.h5', 'r') as h5f:
            length = h5py.Dataset.len(h5f['data'][0])

        return int(length)

    def fold_gen_pair(self, batch_size, fold_n, n=0):
        test_ratio = 1.0 / fold_n
        train_ratio = 1.0 - test_ratio

        train_sample_count = int(np.floor(self.indices.shape[0] * train_ratio))
        test_sample_count = int(np.floor(self.indices.shape[0] * test_ratio))

        test_offset = n * test_sample_count

        test_indices = self.indices[test_offset:(test_offset + test_sample_count)]
        train_indices = np.concatenate([self.indices[:test_offset], self.indices[(test_offset + test_sample_count):]])

        train_gen = DataGenerator88(self.file_name, batch_size, 1.0 - test_ratio)
        test_gen = DataGenerator88(self.file_name, batch_size, test_ratio)

        train_gen.overwrite_indices(train_indices)
        test_gen.overwrite_indices(test_indices)

        return train_gen, test_gen


class DataGenerator(Sequence):
    def __init__(self, file_name, batch_size, used_ratio, is_test=False):
        self.file_name = file_name
        self.batch_size = batch_size
        self.used_ratio = used_ratio
        self.is_test = is_test
        self.indices = np.arange(self.get_length(True))  # training/test length

        if self.is_test:
            offset = 0
            offset = self._get_training_length()
            self.indices += offset

    def __len__(self):
        return int(np.ceil(self.get_length() / self.batch_size))

    def __getitem__(self, index):
        data = []
        label = []

        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size

        indices_to_use = np.sort(self.indices[start_index:end_index])

        with h5py.File(self.file_name + '.h5', 'r') as h5f:
            data = h5f['data'][:, indices_to_use].T
            label = h5f['label'][:, indices_to_use].T

        return data, label

    def overwrite_indices(self, indices):
        # assert self.indices.shape[0] == indices.shape[0], 'Overwriting indices need both to have the same length {}, {}'.format(self.indices.shape[0], indices.shape[0])
        self.indices = indices

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def get_length(self, force_read_file=False):
        if force_read_file:
            length = 0
            with h5py.File(self.file_name + '.h5', 'r') as h5f:
                ratio = self.used_ratio

                if self.is_test:
                    ratio = 1.0 - ratio  # test data ratio

                length = np.floor(h5py.Dataset.len(h5f['data'][0]) * ratio)

            return int(length)
        else:
            return self.indices.shape[0]

    def _get_training_length(self): # Should not be accessed outside of this class
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

        indices_to_use = np.sort(self.indices[start_index:end_index])

        with h5py.File(self.file_name + '.h5', 'r') as h5f:
            data = h5f['data'][:, indices_to_use].T
            label = h5f['label'][:, indices_to_use]

            # =====
            for i in range(PITCH_RANGE):
                final_label['out{}'.format(i)] = label[i + PITCH_RANGE, :]

        return data, final_label


class EpochGraphCallback(Callback):
    def __init__(self, file_name):
        self.file_name = file_name
        with h5py.File(file_name + '.h5', 'w') as h5f:
            # epoch, loss, val_loss
            h5f.create_dataset('data', shape=(0, 3), compression='gzip', chunks=True, maxshape=(None, 3))

    def on_epoch_end(self, epoch, logs=None):
        print('Epoch #{}, loss={}, val_loss={}'.format(epoch, logs['loss'], logs['val_loss']))

        with h5py.File(self.file_name + '.h5', 'a') as h5f:
            h5f['data'].resize(h5f['data'].shape[0] + 1, axis=0)

            h5f['data'][-1:] = [epoch, logs['loss'], logs['val_loss']]
