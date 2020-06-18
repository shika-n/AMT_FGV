import csv
import h5py
import matplotlib.pyplot as plt
import numpy as np
import random
import threading
import streamlit as st

from numba import jit
from os import walk
from os.path import isfile, join
from librosa.core import load, to_mono, resample, cqt, amplitude_to_db, frames_to_time
from librosa.display import specshow
from util.file_handler import auto_load

TARGET_SAMPLE_RATE = 16000
HOP_LENGTH = 512
TOTAL_BINS = 252
BINS_PER_OCTAVE = 36

PITCH_RANGE = 108 - 21 + 1

def process(file_path_or_bytes, st_status):
    cqt_result = generate_cqt(file_path_or_bytes, st_status)  # (PITCH_RANGE, time)

    if isinstance(file_path_or_bytes, str) and isfile(file_path_or_bytes[:-4] + '.txt'):
        print(isfile(file_path_or_bytes[:-4] + '.txt'))
        label_result = process_csv_data(file_path_or_bytes[:-4] + '.txt', cqt_result.shape[1], st_status)
        label_result = label_result[PITCH_RANGE:, :]
    else:
        label_result = None

    with h5py.File('sl_data/std/means_stds.h5', 'r') as h5f:
        cqt_result = np.divide(np.subtract(cqt_result, h5f['means']), h5f['stds'])

    st_status.success('Data loaded successfully')

    return cqt_result, label_result

def process_csv_data(file_path, cqt_length, st_status):
    st_status.text('Processing CSV data...')
    result = np.array([])
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        # Skip header
        next(reader)
        row_count = sum(1 for row in reader)

        sets = np.zeros([row_count, 3])

        csvfile.seek(0)
        next(reader)
        for i, row in enumerate(reader):
            if len(row) != 3:
                continue

            onset = row[0]
            offset = row[1]
            midi_pitch = row[2]

            sets[i] = [onset, offset, midi_pitch]

        result = np.zeros([PITCH_RANGE * 2, cqt_length])
        time_stamps = frames_to_time(range(0, cqt_length + 1), sr=TARGET_SAMPLE_RATE, hop_length=HOP_LENGTH)
        process_csv_data_jit(sets, time_stamps, cqt_length, result)

    return result


@jit
def process_csv_data_jit(sets, time_stamps, cqt_length, out_result):
    for col in range(cqt_length):
        start_time = time_stamps[col]
        until_time = time_stamps[col + 1]

        for i in range(len(sets)):
            onset = sets[i, 0]
            offset = sets[i, 1]
            pitch = sets[i, 2]

            if start_time <= onset < until_time:
                out_result[int(pitch - 21), col] = 1
                out_result[int(pitch - 21 + PITCH_RANGE), col] = 1
            elif onset <= start_time and (offset > until_time or start_time <= offset < until_time):
                out_result[int(pitch - 21 + PITCH_RANGE), col] = 1


def generate_cqt(file_path, st_status):
    st_status.text('Opening {}'.format(file_path))
    data, sample_rate = auto_load(file_path, sr=None)
    print('Sample Rate:', sample_rate, 'shape:', data.shape)

    if len(data.shape) == 2:
        print('Converting to mono channel...')
        data = to_mono(data)

    st_status.text('Resampling to {} Hz...'.format(TARGET_SAMPLE_RATE))
    downsampled_data = resample(data, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
    # downsampled_data = data
    st_status.text('Downsampled to {} Hz, shape is now {}'.format(TARGET_SAMPLE_RATE, downsampled_data.shape))

    st_status.text('Generating CQT...')
    cqt_result = np.abs(cqt(downsampled_data, sr=TARGET_SAMPLE_RATE, hop_length=HOP_LENGTH, n_bins=TOTAL_BINS, bins_per_octave=BINS_PER_OCTAVE))

    return cqt_result