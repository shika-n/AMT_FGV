import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from util.groundtruth import NoteEvents
import midi
import glob
from random import shuffle
import h5py
import os
from util.file_handler import auto_load
import streamlit as st

DURATION = None
DOWNSAMPLED_SR = 16000
HOP_LENGTH = 512
NUM_OCTAVES = 7
BINS_PER_OCTAVE = 36
NUM_BINS = NUM_OCTAVES * BINS_PER_OCTAVE
WINDOW_SIZE = 7

def preprocess_wav_file(file_path_or_bytes, Y_numSlice):
    # returns 1 example (downsampled, cqt, normalized)
    np_array_list = []

    y, sr = auto_load(file_path_or_bytes, sr =None)
    y_downsample = librosa.resample(y, orig_sr=sr, target_sr=DOWNSAMPLED_SR)
    CQT_result = librosa.cqt(y_downsample, sr=DOWNSAMPLED_SR, hop_length=HOP_LENGTH, n_bins=NUM_BINS, bins_per_octave=BINS_PER_OCTAVE)
    CQT_result = np.absolute(CQT_result)
    np_array_list.append(CQT_result)

# normalize data
    combined = np.concatenate(np_array_list, axis = 1)
    
    ####
    '''
    max_val = combined.max()
    min_val = combined.min()
    
    combined_norm = (combined - min_val) / (max_val - min_val)
    mean_per_label = np.mean(combined_norm, axis = 1)
    mean_per_label = np.reshape(mean_per_label, (-1, 1))
    
    for i in range(len(np_array_list)):
        np_array_list[i] = (np_array_list[i] - min_val) / (max_val - min_val)
        np_array_list[i] = np_array_list[i] - mean_per_label
        
    with h5py.File('minmax_meanlabel.h5', 'w') as h5f:
        h5f.create_dataset('min_max', data=[min_val, max_val], compression='gzip')
        h5f.create_dataset('mean_per_label', data=mean_per_label, compression='gzip')
    '''
    ########

    with h5py.File('sl_data/std/means_stds-nm.h5', 'r') as h5f:
        #cqt_result = np.divide(np.subtract(cqt_result, h5f['means']), h5f['stds'])
    
        mean = h5f['means'][:]#np.mean(combined, axis = 1, keepdims =True)
        std = h5f['stds'][:]#np.std(combined, axis = 1, keepdims=True)
    
    for i in range(len(np_array_list)):
        np_array_list[i] = np.divide(np.subtract(np_array_list[i], mean), std)
    '''    
    with h5py.File('means_stds.h5', 'w') as h5f:
        h5f.create_dataset('means', data=mean, compression='gzip')
        h5f.create_dataset('stds', data=std, compression='gzip')
    '''
    ####

    
    frame_windows_list = []
    numSlices_list = []
    for i in range(len(np_array_list)):
        CQT_result = np_array_list[i]
        # print (CQT_result.shape[0])
        # print ("====")
        # print (CQT_result.shape[1])
        paddedX = np.zeros((CQT_result.shape[0], CQT_result.shape[1] + WINDOW_SIZE - 1), dtype=float)
        pad_amount = WINDOW_SIZE / 2
        pad_amount = int(pad_amount)
        paddedX[:, pad_amount:-pad_amount] = CQT_result
        # print (paddedX[:, pad_amount:-pad_amount])
        frame_windows = np.array([paddedX[:, j:j+WINDOW_SIZE] for j in range(CQT_result.shape[1])])
        frame_windows = np.expand_dims(frame_windows, axis=3)
        
        if Y_numSlice is not None:
            numSlices = min(frame_windows.shape[0], Y_numSlice) #Y_numSlices[i])
        else:
            numSlices = frame_windows.shape[0]

        numSlices_list.append(numSlices)
        frame_windows_list.append(frame_windows[:numSlices])
    
    # return np.concatenate(frame_windows_list, axis=0), numSlices_list
    return frame_windows_list, numSlices_list

def preprocess_midi_truth(filename):
    # returns 1 ground truth binary vector (size 88)
    pattern = midi.read_midifile(filename)
    events = NoteEvents(pattern)
    truth = events.get_ground_truth(31.25, DURATION) # (88, numSlices)
    return truth

def get_wav_midi_data(file_path_or_bytes, st_status):
    Y_numSlices = None
    Y_list = []

    if isinstance(file_path_or_bytes, str) and os.path.isfile(file_path_or_bytes[:-4] + '.mid'):
        st_status.text('Generating ground truth from midi')
        Y_i = preprocess_midi_truth(file_path_or_bytes[:-4] + '.mid')
        Y_numSlices = Y_i.shape[1]
    else:
        Y_i = None

    Y_list.append(Y_i)


    st_status.text('Preprocessing WAV file')
    X, numSlices = preprocess_wav_file(file_path_or_bytes, Y_numSlices)
    
    # Custom - modified preprocess_wav_file return value
    ####
    #print(np.asarray(X).shape)
    X = np.concatenate(X, axis=0)
    #print(X.shape)
    
    ########
    '''
    with h5py.File('temp.h5', 'w') as h5f:
        for i, x_data in enumerate(X):
            print('Appending #{}'.format(i))
            if i == 0:
                h5f.create_dataset('data', shape=(0, 252, 7, 1), compression='gzip', chunks=True, maxshape=(None, 252, 7, 1))
            
            h5f['data'].resize(h5f['data'].shape[0] + x_data.shape[0], axis=0)
            
            h5f['data'][-x_data.shape[0]:] = x_data
            
        print(h5f['data'].shape)
        
    X = []
    with h5py.File('temp.h5', 'r') as h5f:
        X = h5f['data'][:]
    ####'''
    
    if Y_i is not None:
        Y_list = [Y_list[i][:,:numSlices[i]] for i in range(len(Y_list))]
        Y = np.concatenate(Y_list, axis=1)
        Y = [Y[i] for i in range(Y.shape[0])]
    else:
        Y = None

    return X, Y

def process(file_path_or_bytes, st_status):
    st_status.text('Processing...')
    X, Y = get_wav_midi_data(file_path_or_bytes, st_status)
    #print ("Number of Training Examples: {}".format(X.shape[0]))
    return X, np.asarray(Y)