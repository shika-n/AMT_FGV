import sys
sys.path.append('')
print(sys.path)

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
from Neural.DataGenerator import DataGenerator88, EpochGraphCallback, FoldGen
from Neural.run_prediction import calculate_accuracy
from preprocess import PITCH_RANGE

EPOCH_NUM = 20
BATCH_SIZE = 32
K_FOLD = 5

def main():
    fold_gen = FoldGen('../data/merged')


    # checkpoint_file_path = 'model_train_88.best.h5'

    # if isfile(checkpoint_file_path):
    #    model = load_model(checkpoint_file_path)

    accs = []
    f1s = []
    recalls = []
    precisions = []

    for i in range(K_FOLD):
        model = create_model()

        # plot_model(model, to_file='88_model.png', show_shapes=True)
        # print('saved')

        # train_gen = DataGenerator88('../data/merged', BATCH_SIZE, 0.8)
        # test_gen = DataGenerator88('../data/merged', BATCH_SIZE, 0.8, is_test=True)
        train_gen, test_gen = fold_gen.fold_gen_pair(BATCH_SIZE, K_FOLD, i)

        # checkpoint = ModelCheckpoint(checkpoint_file_path, monitor='loss', verbose=1, save_best_only=True, mode='min')

        # model.fit(train_gen, verbose=1)
        model.fit(x=train_gen,
                  epochs=EPOCH_NUM,
                  verbose=0,
                  validation_data=test_gen,
                  callbacks=[EpochGraphCallback('losses_log')])

        # evaluate
        prediction = model.predict(x=test_gen,
                                   verbose=1)
        label = []
        for i in range(test_gen.__len__()):
            label += test_gen.__getitem__(i)

        acc, f1, recall, precision = calculate_accuracy(label, prediction)

        print('Acc: {}\nF1: {}\nRecall: {}\nPrecision: {}'.format(acc, f1, recall, precision))
        accs.append(acc)
        f1s.append(f1)
        recalls.append(recall)
        precisions.append(precision)

        # save model
        model.save('model_88_mlp_{}.h5'.format(i))

    with open('acc.txt', 'w') as f:
        for i in range(K_FOLD):
            f.writelines([
                'Model #{}'.format(i),
                'Acc: {}'.format(accs[i]),
                'F1: {}'.format(f1s[i]),
                'Recall: {}'.format(recalls[i]),
                'Precision: {}\n'.format(precisions[i])
            ])

        f.writelines([
            '================',
            'Acc: {} (+/- {})'.format(np.mean(accs), np.std(accs)),
            'F1: {} (+/- {})'.format(np.mean(f1s), np.std(f1s)),
            'Recall: {} (+/- {})'.format(np.mean(recalls), np.std(recalls)),
            'Precision: {} (+/- {})\n'.format(np.mean(precisions), np.std(precisions))
        ])


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