import h5py
import numpy as np
from os.path import isfile
from tensorflow.keras import Sequential, backend
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import BinaryAccuracy
from Neural.DataGenerator import DataGenerator


def main():
    checkpoint_file_path = 'model_train.best.h5'

    model = create_model()
    if isfile(checkpoint_file_path):
        model.load_weights(checkpoint_file_path)

    train_gen = DataGenerator('../data/merged', 128)
    test_gen = DataGenerator('../data/merged', 128, is_test=True)

    checkpoint = ModelCheckpoint(checkpoint_file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model.fit(x=train_gen,
              epochs=1000,
              verbose=2,
              validation_data=test_gen,
              callbacks=[checkpoint])

    model.save('model.h5')


def open_h5(file_name):
    return h5py.File(file_name + '.h5', 'r')


def create_model():
    model = Sequential([
        Dense(252, input_dim=252, activation='relu'),
        Dense(240, activation='relu'),
        Dense(200, activation='relu'),
        Dense(180, activation='relu'),
        Dense(176, activation='sigmoid')
    ])

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy')
                  #metrics=['categorical_accuracy', single_class_accuracy])

    return model

def single_class_accuracy(interesting_class_id):
    def fn(y_true, y_pred):
        class_id_preds = backend.argmax(y_pred, axis=-1)
        # Replace class_id_preds with class_id_true for recall here
        positive_mask = backend.cast(backend.equal(class_id_preds, interesting_class_id), 'int32')
        true_mask = backend.cast(backend.equal(y_true, interesting_class_id), 'int32')
        acc_mask = backend.cast(backend.equal(positive_mask, true_mask), 'float32')
        class_acc = backend.mean(acc_mask)
        return class_acc

    return fn
'''
def custom_acc(y_true, y_pred):
    sess = backend.
    y_true_np = backend.eval(y_true)
    y_pred_np = backend.eval(y_pred)

    diff = np.sum(np.abs(np.subtract(y_true_np, y_pred_np)))
    if diff == 0:
        return 1.0
    else:
        return 0.0'''


if __name__ == '__main__':
    main()