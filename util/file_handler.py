import soundfile as sf
import librosa

from os import walk, listdir
from os.path import isfile, join, isdir

def get_file_name_list(root_path):
    files = []

    for dir_path, dir_names, file_names in walk(root_path):
        for file_name in file_names:
            files.append(file_name)

    return files

def get_folder_name_list(root_path, prefix=None):
    folders = []

    for dir_name in listdir(root_path):
        if prefix is not None:
            if dir_name[:len(prefix)].lower() != prefix.lower():
                continue

        dir_path = join(root_path, dir_name)
        if isdir(dir_path):
            folders.append(dir_name)

    return folders

def auto_load(file_path_or_bytes, sr=22050):
    if isinstance(file_path_or_bytes, str):
        return load_from_file(file_path_or_bytes, sr=sr)
    else:
        return load_from_bytes(file_path_or_bytes, sr=sr)

def load_from_file(file_path, sr):
    data, sample_rate = librosa.load(file_path, sr=sr)

    return data, sample_rate

def load_from_bytes(file_bytes, sr):
    data, sample_rate = sf.read(file_bytes, dtype='float32')
    data = data.T

    if sr is not None:
        data = librosa.resample(data, sample_rate, sr)

    return data, sample_rate

def get_file_bytes(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
        return data