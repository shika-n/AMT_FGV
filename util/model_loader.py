import streamlit as st
import numpy as np
import h5py
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Bidirectional, Input
from tensorflow.keras import backend as K

def load_model_sl(file_name, st_status):
    st_status.text('Loading model... ({})'.format(file_name))
    
    model = load_model(file_name)
    
    st_status.info('Model loaded!')

    return model

def run_model_sl(file_name, input, st_status, architecture_selected):
    model = load_model_sl(file_name, st_status)

    print('--------')
    print(model.summary())

    st_status.text('Predicting... {}'.format(input.shape))
    if architecture_selected == 'mlp':
        input_data = input.T
    elif architecture_selected == 'lstm' or architecture_selected == 'bilstm' or architecture_selected == 'dnn':
        if architecture_selected == 'bilstm':
            # For simulation purposes
            newInput = Input(shape=(100, 252))
            newOutputs = model(newInput)
            model = Model(newInput, newOutputs)

        data_max_shape = (input.shape[0] // 100) * 100 # 100 might be size_samples
        
        x_data = np.reshape(input[:data_max_shape, :], (input.shape[0] // 100, 100, 252))
        
        input_data = x_data
    else:
        input_data = input

    '''
    # BIDIRECTIONAL
    layer_outputs = []
    for t_layer in model.layers:
        if isinstance(t_layer, Sequential):
            for b_layer in t_layer.layers:
                if isinstance(b_layer, Bidirectional):
                    print(b_layer)
                    print(b_layer.layer)
                    layer_outputs.append(b_layer.output)
                    print(b_layer.backward_layer)
                    layer_outputs.append(b_layer.backward_layer.output)
                else:
                    print(b_layer)
                    layer_outputs.append(b_layer.output)
        else:
            print(t_layer)
            layer_outputs.append(t_layer.output)
    #####

    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    #activations = activation_model.predict(np.expand_dims(input[:, int(input.shape[1] / 2)], axis=0))
    print(input_data.shape)
    activations = activation_model.predict(np.expand_dims(input_data[input_data.shape[0] // 2], axis=0))

    print(len(activations))
    with h5py.File('simulation.h5', 'w') as h5f:
        h5f.create_dataset('length', data=[len(activations)], compression='gzip')
        for i in range(len(activations)):
            h5f.create_dataset('data_{}'.format(i), data=activations[i], compression='gzip')
        print('simulation.h5 is saved')
    ###
    '''

    prediction = np.asarray(model.predict(input_data))

    if architecture_selected == 'lstm' or architecture_selected == 'bilstm' or architecture_selected == 'dnn':
        prediction = np.reshape(prediction, (prediction.shape[0] * prediction.shape[1], 88))

    del model
    model = None
    clear_session()

    st_status.info('Prediction done! {}'.format(prediction.shape))
    prediction = np.reshape(prediction, (prediction.shape[0], prediction.shape[1]))

    return np.clip(prediction, 0, 1)

