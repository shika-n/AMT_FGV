import util.preprocess_mlp as preprocess_mlp
import util.preprocess_cnn as preprocess_cnn
import util.WAV2mat_batch as preprocess_lstm

import librosa
import numpy as np
import soundfile as sf
import io
import streamlit as st
from numba import jit

# Returns (cqt, labels)
@st.cache(hash_funcs={st.DeltaGenerator.DeltaGenerator: lambda x : None})
def preprocess(file_path_or_bytes, st_status, architecture_selected):
    if architecture_selected == 'mlp':
        return preprocess_mlp.process(file_path_or_bytes, st_status)
    elif architecture_selected == 'cnn':
        return preprocess_cnn.process(file_path_or_bytes, st_status)
    elif architecture_selected == 'lstm' or architecture_selected == 'bilstm':
        return preprocess_lstm.process(file_path_or_bytes, st_status)


def post_process(predictions_arg, rounding_threshold, minimum_fill, gap_threshold, iter_count):
    predictions = np.array(predictions_arg)
    predictions = np.clip(np.ceil(predictions - rounding_threshold), 0, 1).astype(int)

    if predictions.shape[0] > predictions.shape[1]:
        predictions = predictions.T

    #processed_prediction = np.array(predictions).

    print(predictions.shape)
    for x in range(iter_count):
        for pitch in range(predictions.shape[0]):
            last_val = 0
            start_pos = 0
            for current_pos in range(predictions.shape[1]):
                current_val = predictions[pitch, current_pos]

                if current_val != last_val:
                    if last_val == 1 and current_pos - start_pos < minimum_fill: # now is 0, remove fill
                        for j in range(start_pos, current_pos):
                            predictions[pitch, j] = 0
                    elif last_val == 0 and current_pos - start_pos < gap_threshold: # now is 1, fill gap
                        for j in range(start_pos, current_pos):
                            predictions[pitch, j] = 1
                    else:
                        start_pos = current_pos
                        last_val = current_val

    print('post process done')
    return predictions.astype(float)
        