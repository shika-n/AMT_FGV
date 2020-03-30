import h5py
import numpy as np
from os.path import isfile
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model
from Neural.DataGenerator import DataGenerator88
from preprocess import PITCH_RANGE

def main():
    checkpoint_file_path = 'model_train_88.best.h5'

    model = create_model()

    plot_model(model, to_file='88_model.png', show_shapes=True)
    print('saved')

    if isfile(checkpoint_file_path):
        model = load_model(checkpoint_file_path)

    train_gen = DataGenerator88('../data/merged', 512, 0.8)
    test_gen = DataGenerator88('../data/merged', 512, 0.8, is_test=True)

    checkpoint = ModelCheckpoint(checkpoint_file_path, monitor='loss', verbose=1, save_best_only=True, mode='min')

    # model.fit(train_gen, verbose=1)
    model.fit(x=train_gen,
              epochs=10,
              verbose=1,
              validation_data=test_gen,
              callbacks=[checkpoint])


def open_h5(file_name):
    return h5py.File(file_name + '.h5', 'r')


def create_model():
    input_layer = Input(shape=(252,), name='input')

    sub_model_outputs = []

    for i in range(PITCH_RANGE):
        sub_model_outputs.append(create_sub_model(i, input_layer))

    model = Model(inputs=[input_layer], outputs=sub_model_outputs)

    losses = {}
    for i in range(PITCH_RANGE):
        losses['out{}'.format(i)] = 'binary_crossentropy'

    model.compile(optimizer='rmsprop',
                  loss=losses,
                  metrics=['accuracy'])

    return model


def create_sub_model(i, input_layer):
    sub_model_input = Dense(100, activation='relu', name='sub_in{}'.format(i))(input_layer)
    sub_model_output = Dense(1, activation='sigmoid', name='out{}'.format(i))(sub_model_input)

    return sub_model_output


if __name__ == '__main__':
    main()