from os import walk, listdir
from os.path import isfile, join, isdir

def get_file_name_list(root_path):
    files = []

    for dir_path, dir_names, file_names in walk(root_path):
        for file_name in file_names:
            files.append(file_name)

    return files

def get_folder_name_list(root_path):
    folders = []

    for dir_name in listdir(root_path):
        dir_path = join(root_path, dir_name)
        if isdir(dir_path):
            folders.append(dir_name)

    return folders