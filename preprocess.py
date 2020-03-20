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
LABEL_COUNT = PITCH_RANGE * 2

dataset_mutex = threading.Lock()

def main():
    #process(0, 'data/predict/MAPS_MUS-bk_xmas1_ENSTDkAm', 1.0)

    file_names = get_file_list('data/train')
    create_empty_h5('data/merged')
    batch_process(file_names, 5, 0.7, 'data/merged')

    # SLOW
    # shuffle_dataset('data/merged')


def batch_process(file_names, batch_size, training_percentage, h5file_name=''):
    batch_timer = Timer()
    thread_batch = []
    for i, file_name in enumerate(file_names):
        if i % batch_size == 0:
            thread_batch.clear()

        thread = threading.Thread(target=process, args=(i, file_name, training_percentage, h5file_name))
        thread_batch.append(thread)
        thread.start()

        if (i + 1) % batch_size == 0 or i == len(file_names):
            for thread in thread_batch:
                thread.join()

            print('======== {}/{} files processed.'.format(i + 1, len(file_names)), batch_timer.toc(), 'second(s) has passed')


# SLOW!
def shuffle_dataset(h5file_name):
    with Timer('Shuffling') as _:
        with h5py.File(h5file_name + '.h5', 'r+') as h5f:
            print('Shuffling ...')
            shuffle_in_unison(h5f['train_data'], h5f['train_label'])
            shuffle_in_unison(h5f['test_data'], h5f['test_label'])


def process(i, file_name, training_percentage, h5file_name=''):
    cqt_result = generate_cqt(i, file_name + '.wav')

    # save_spectrogram(cqt_result, 'cqt result.png')
    cqt_result = np.transpose(cqt_result)
    # print('CQT shape:', cqt_result.shape)

    with Timer('[{}] CSV processing'.format(i)) as _:
        label_result = process_csv_data(i, file_name + '.txt', len(cqt_result))
        # print('Label shape:', label_result.shape)
        np.savetxt('label.txt', label_result, fmt='%i')
        quit()

    with Timer('[{}] Shuffling'.format(i)) as _:
        print('[{}] Shuffling...'.format(i))
        shuffle_in_unison(cqt_result, label_result)

    if h5file_name == '':
        with h5py.File(file_name + '.h5', 'w') as h5f:
            split_index = int(len(cqt_result) * training_percentage)
            h5f.create_dataset('train_data', data=cqt_result[:split_index], compression='gzip')
            h5f.create_dataset('train_label', data=label_result[:split_index], compression='gzip')
            h5f.create_dataset('test_data', data=cqt_result[split_index:], compression='gzip')
            h5f.create_dataset('test_label', data=label_result[split_index:], compression='gzip')
    else:
        with dataset_mutex:
            with h5py.File(h5file_name + '.h5', 'a') as h5f:
                split_index = int(len(cqt_result) * training_percentage)

                train_data = cqt_result[:split_index]
                test_data = cqt_result[split_index:]
                train_data_length = train_data.shape[0]
                test_data_length = test_data.shape[0]

                h5f['train_data'].resize(h5f['train_data'].shape[0] + train_data_length, axis=0)
                h5f['train_label'].resize(h5f['train_label'].shape[0] + train_data_length, axis=0)
                h5f['test_data'].resize(h5f['test_data'].shape[0] + test_data_length, axis=0)
                h5f['test_label'].resize(h5f['test_label'].shape[0] + test_data_length, axis=0)

                h5f['train_data'][-train_data_length:] = train_data
                h5f['train_label'][-train_data_length:] = label_result[:split_index]
                h5f['test_data'][-test_data_length:] = test_data
                h5f['test_label'][-test_data_length:] = label_result[split_index:]


def create_empty_h5(file_name):
    with h5py.File(file_name + '.h5', 'w') as h5f:
        h5f.create_dataset('train_data', shape=(0, TOTAL_BINS), compression='gzip', chunks=True, maxshape=(None, TOTAL_BINS))
        h5f.create_dataset('train_label', shape=(0, LABEL_COUNT), compression='gzip', chunks=True, maxshape=(None, LABEL_COUNT))
        h5f.create_dataset('test_data', shape=(0, TOTAL_BINS), compression='gzip', chunks=True, maxshape=(None, TOTAL_BINS))
        h5f.create_dataset('test_label', shape=(0, LABEL_COUNT), compression='gzip', chunks=True, maxshape=(None, LABEL_COUNT))


def shuffle_in_unison(a, b):
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


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

        result = np.zeros([cqt_length, LABEL_COUNT])
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
                out_result[col, int(pitch - 21)] = 1
                out_result[col, int(pitch - 21 + PITCH_RANGE)] = 1
            elif onset <= start_time and (offset > until_time or start_time <= offset < until_time):
                out_result[col, int(pitch - 21 + PITCH_RANGE)] = 1


def generate_cqt(i, file_path, offset=0, duration=None):
    print('[{}] Opening'.format(i), file_path)
    data, sample_rate = load(file_path, sr=None, offset=offset, duration=duration)
    print('[{}] Sample Rate:'.format(i), sample_rate, 'shape:', data.shape)

    if len(data.shape) == 2:
        with Timer('[{}] Converted to mono'.format(i)) as _:
            print('[{}] Converting to mono channel...'.format(i))
            data = to_mono(data)

    with Timer('[{}] Resampling'.format(i)) as _:
        print('[{}] Resampling to'.format(i), TARGET_SAMPLE_RATE, 'Hz...')
        downsampled_data = resample(data, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
        # downsampled_data = data
        print('[{}] Downsampled to'.format(i), TARGET_SAMPLE_RATE, 'Hz shape is now', downsampled_data.shape)

    with Timer('[{}] CQT'.format(i)) as _:
        print('[{}] Generating CQT...'.format(i))
        cqt_result = np.abs(cqt(downsampled_data, sr=TARGET_SAMPLE_RATE, hop_length=HOP_LENGTH, n_bins=TOTAL_BINS, bins_per_octave=BINS_PER_OCTAVE))

    # mean = np.mean(cqt_result, axis=1, keepdims=True)
    # std = np.std(cqt_result, axis=1, keepdims=True)
    # cqt_result = np.divide(np.subtract(cqt_result, mean), std)

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