import csv
import h5py
import matplotlib.pyplot as plt
import numpy as np
import random
import threading
from numba import jit
from os import walk
from os.path import isfile, join
from librosa.core import load, to_mono, resample, cqt, amplitude_to_db, frames_to_time
from librosa.display import specshow
from Util.Timer import Timer

TARGET_SAMPLE_RATE = 16000
HOP_LENGTH = 512
TOTAL_BINS = 252
BINS_PER_OCTAVE = 36

PITCH_RANGE = 108 - 21 + 1

dataset_mutex = threading.Lock()


def main():
    # create_empty_h5('data/merged_one')
    # process(0, 'data/predict/MAPS_MUS-bk_xmas1_ENSTDkAm', 'data/merged_one')
    # with h5py.File('data/merged_one.h5', 'r+') as h5f:
    #    mean = np.mean(h5f['data'][:], axis=1, keepdims=True)
    #    std = np.std(h5f['data'][:], axis=1, keepdims=True)
    #    h5f['data'][:] = np.divide(np.subtract(h5f['data'][:], mean), std)
    #    save_spectrogram(h5f['data'][:, :313], 'merged_one.png')

    # quit()

    file_names = get_file_list('data/train')
    print(len(file_names), 'files detected')

    # file_names = ['data/predict/MAPS_MUS-bk_xmas1_ENSTDkAm']
    # file_names = file_names[:50]

    create_empty_h5('data/merged')
    with Timer('Batch process'):
        batch_process(file_names, 5, 'data/merged')

    with Timer('Standardization'):
        print('Standardization...')
        # Standardization, time-wise (axis=1, when doing all of them at once)
        with h5py.File('data/merged.h5', 'r+') as h5f:
            for i in range(TOTAL_BINS):
                data = h5f['data'][i, :]
                mean = np.mean(data, keepdims=True)
                std = np.std(data, keepdims=True)
                h5f['data'][i, :] = np.divide(np.subtract(data, mean), std)

                print('{}/{}'.format(i + 1, TOTAL_BINS))

            #save_spectrogram(h5f['data'][:, :313], 'merged.png')


def batch_process(file_names, batch_size, h5file_name=''):
    batch_timer = Timer()
    thread_batch = []
    for i, file_name in enumerate(file_names):
        if i % batch_size == 0:
            thread_batch.clear()

        thread = threading.Thread(target=process, args=(i, file_name, h5file_name))
        thread_batch.append(thread)
        thread.start()

        if (i + 1) % batch_size == 0 or i == len(file_names) - 1:
            for thread in thread_batch:
                thread.join()

            print('======== {}/{} files processed.'.format(i + 1, len(file_names)), batch_timer.toc(), 'second(s) has passed')


def process(i, file_name, h5file_name=''):
    cqt_result = generate_cqt(i, file_name + '.wav')  # (PITCH_RANGE, time)

    #cqt_result = cqt_result.T  # (time, PITCH_RANGE)
    # print('CQT shape:', cqt_result.shape)

    with Timer('[{}] CSV processing'.format(i)):
        label_result = process_csv_data(i, file_name + '.txt', cqt_result.shape[1])
        # print('Label shape:', label_result.shape)
        np.savetxt('label.txt', label_result, fmt='%i')

    if h5file_name == '':
        with h5py.File(file_name + '.h5', 'w') as h5f:
            h5f.create_dataset('data', data=cqt_result, compression='gzip')
            h5f.create_dataset('label', data=label_result, compression='gzip')
    else:
        with dataset_mutex:
            with h5py.File(h5file_name + '.h5', 'a') as h5f:
                data_length = cqt_result.shape[1]

                h5f['data'].resize(h5f['data'].shape[1] + data_length, axis=1)
                h5f['label'].resize(h5f['label'].shape[1] + data_length, axis=1)

                h5f['data'][:, -data_length:] = cqt_result
                h5f['label'][:, -data_length:] = label_result


def create_empty_h5(file_name):
    with h5py.File(file_name + '.h5', 'w') as h5f:
        h5f.create_dataset('data', shape=(TOTAL_BINS, 0), compression='gzip', chunks=True, maxshape=(TOTAL_BINS, None))
        h5f.create_dataset('label', shape=(PITCH_RANGE * 2, 0), compression='gzip', chunks=True, maxshape=(PITCH_RANGE * 2, None))


def process_csv_data(i, file_path, cqt_length):
    print('[{}] Processing CSV data...'.format(i))
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


def generate_cqt(i, file_path, offset=0, duration=None):
    print('[{}] Opening'.format(i), file_path)
    data, sample_rate = load(file_path, sr=None, offset=offset, duration=duration)
    print('[{}] Sample Rate:'.format(i), sample_rate, 'shape:', data.shape)

    if len(data.shape) == 2:
        with Timer('[{}] Converted to mono'.format(i)):
            print('[{}] Converting to mono channel...'.format(i))
            data = to_mono(data)

    with Timer('[{}] Resampling'.format(i)):
        print('[{}] Resampling to'.format(i), TARGET_SAMPLE_RATE, 'Hz...')
        downsampled_data = resample(data, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
        # downsampled_data = data
        print('[{}] Downsampled to'.format(i), TARGET_SAMPLE_RATE, 'Hz shape is now', downsampled_data.shape)

    with Timer('[{}] CQT'.format(i)):
        print('[{}] Generating CQT...'.format(i))
        cqt_result = np.abs(cqt(downsampled_data, sr=TARGET_SAMPLE_RATE, hop_length=HOP_LENGTH, n_bins=TOTAL_BINS, bins_per_octave=BINS_PER_OCTAVE))

    return cqt_result


def save_spectrogram(data, output_file):
    specshow(amplitude_to_db(data, ref=np.max), sr=TARGET_SAMPLE_RATE, x_axis='time', y_axis='cqt_note', hop_length=HOP_LENGTH,bins_per_octave=BINS_PER_OCTAVE)
    plt.colorbar(format='%+2.0f dB')
    plt.title("Constant-Q Transform")
    plt.tight_layout()
    plt.savefig(output_file)
    # plt.show()


def get_file_list(root_path):
    print('Gathering files...')

    files = []

    for dir_path, dir_names, file_names in walk(root_path):
        for file_name in file_names:
            if file_name.endswith('.wav'):
                file_path = join(dir_path, file_name[:-4])
                if isfile(file_path + '.txt'):
                    files.append(file_path)

    return files


if __name__ == '__main__':
    main()