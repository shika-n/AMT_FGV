import sys
sys.path.append('')
print(sys.path)

import h5py
import numpy as np
from os.path import isfile
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model
from Neural.DataGenerator import DataGenerator88, EpochGraphCallback, FoldGen
from Neural.eval_collection import save_eval, evaluate_model_sk, evaluate_model_cnn, evaluate_model_lstm
from preprocess import PITCH_RANGE

EPOCH_NUM = 50
BATCH_SIZE = 128
K_FOLD = 5

def main():
    fold_gen = FoldGen('merged')


    # checkpoint_file_path = 'model_train_88.best.h5'

    # if isfile(checkpoint_file_path):
    #    model = load_model(checkpoint_file_path)

    cnn_eval = []
    lstm_eval = []
    sk_macro_eval = []
    sk_micro_eval = []
    for i in range(K_FOLD):
        model = create_model()
        print('======== FOLD #{} ========'.format(i))

        # plot_model(model, to_file='88_model.png', show_shapes=True)
        # print('saved')

        train_gen, test_gen = fold_gen.fold_gen_pair(BATCH_SIZE, K_FOLD, i)

        # checkpoint = ModelCheckpoint(checkpoint_file_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
        
        early = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

        model.fit(x=train_gen,
                  epochs=EPOCH_NUM,
                  verbose=0,
                  validation_data=test_gen,
                  callbacks=[EpochGraphCallback('losses_log_{}'.format(i)), early])

        # save model
        model.save('model_88_mlp_{}.h5'.format(i))
        
        # evaluate
        #prediction = np.asarray(model.predict(x=test_gen))
        
        #prediction = prediction.reshape((PITCH_RANGE, -1))
        #prediction = prediction.round()
        
        label = np.zeros([PITCH_RANGE, 0])
        for j in range(test_gen.__len__()):
            x, y = test_gen.__getitem__(j)
            label = np.concatenate([label, np.asarray(y)], axis=1)
        
        cnn_eval.append(evaluate_model_cnn(model, test_gen, label, reshape=(88, -1)))
        lstm_eval.append(evaluate_model_lstm(model, test_gen, label, reshape=(88, -1)))
        sk_macro_eval.append(evaluate_model_sk(model, test_gen, label, average='macro', reshape=(88, -1), transpose=True))
        sk_micro_eval.append(evaluate_model_sk(model, test_gen, label, average='micro', reshape=(88, -1), transpose=True))

    save_eval('cnn_acc', cnn_eval)
    save_eval('lstm_acc', lstm_eval)
    save_eval('sk_macro_acc', sk_macro_eval)
    save_eval('sk_micro_acc', sk_micro_eval)


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