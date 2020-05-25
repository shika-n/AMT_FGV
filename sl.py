from os.path import join

import streamlit as st
import numpy as np
import pandas as pd
import h5py
from numba import jit

import util.model_loader as loader
import util.preprocessing_handler as preprocessing_handler
import util.file_handler as file_handler

MODEL_FOLDER = 'sl_data/models'
PRESET_FOLDER = 'sl_data/presets'

presets = {
    'Music 1': 'MAPS_MUS-chpn-p1_AkPnBcht.wav',
    'Music 2': 'B',
}

loaded_model = None

def main():
    # ======== SIDE BAR ========
    st.sidebar.header('Models')

    architecture_selected = st.sidebar.selectbox('Architecture', ['MLP', 'CNN', 'LSTM', 'BiLSTM', 'DNN'])
    architecture_selected = architecture_selected.lower()

    folders = file_handler.get_folder_name_list(MODEL_FOLDER, prefix=architecture_selected)
    
    for i in range(len(folders)):
        parts = folders[i].split('_')
        folders[i] = '{} Epoch, {} Early'.format(parts[1][:-5], parts[2][:-5])

    selected_conf = st.sidebar.selectbox('Configurations', folders)

    if selected_conf is not None:
        selected_conf_parts = selected_conf.split()
        selected_conf_name = '{}_{}epoch_{}early'.format(architecture_selected, selected_conf_parts[0], selected_conf_parts[2])
        file_names = file_handler.get_file_name_list(join(MODEL_FOLDER, selected_conf_name))
    else:
        file_names = []

    keys, losses = parse_file_list(file_names)
    selected_variant = st.sidebar.selectbox('Fold Variants', keys)

    include_negatives = st.sidebar.checkbox('Include negatives (Preview only)')
    brightness_multiplier = st.sidebar.slider('Brightness multiplier (Preview only)', 1.0, 10.0, value=1.0)
    brightness_offset = st.sidebar.slider('Brightness offset (Preview only)', -2.0, 2.0, value=0.0)
    enable_color_ramp = st.sidebar.checkbox('Enable color ramp')

    # ======== MAIN PAGE ========

    st.title('AMT_FGV')

    if selected_variant is None:
        st.error('Please select a model')
        return

    selected_model_path = join(MODEL_FOLDER, selected_conf_name, selected_variant)
    run_data = load_losses(join(MODEL_FOLDER, selected_conf_name, losses[selected_variant]))

    st.text('This model is ran for %i epochs.' % (run_data.iloc[-1, 0] + 1))
    st.subheader('Loss Graph')

    # Quick plot
    # Change to altair/matplotlib if axis labels are needed
    # https://github.com/streamlit/streamlit/issues/1129
    st.line_chart(run_data.iloc[:, 1:3])
    if st.checkbox('Show raw data'):
        st.dataframe(run_data.iloc[:])

    ##################################################

    st.title('TEST AREA')

    preset_options = ['Upload a file']
    for key in presets:
        preset_options.append(key)

    selected_preset = st.selectbox('Select a preset or upload', preset_options)
    
    status = st.empty()

    file_bytes = None
    cqt = None
    labels = None
    if selected_preset == preset_options[0]:
        file_bytes = st.file_uploader('Upload file', type=['wav', 'txt'])
        if file_bytes is not None:
            cqt, labels = preprocessing_handler.preprocess(file_bytes, status, architecture_selected)

            if architecture_selected == 'cnn':
                cqt_view, labels_view = preprocessing_handler.preprocess(file_bytes, status, 'mlp')
    else:
        st.text('Selected preset: {}'.format(presets[selected_preset]))
        file_path = join(PRESET_FOLDER, presets[selected_preset])

        file_bytes = file_handler.get_file_bytes(file_path)
        cqt, labels = preprocessing_handler.preprocess(file_path, status, architecture_selected)

        if architecture_selected == 'cnn':
            cqt_view, labels_view = preprocessing_handler.preprocess(file_path, status, 'mlp')

    if file_bytes is not None:
        st.audio(file_bytes)

    if cqt is not None:
        st.text('CQT\'s shape: {}'.format(cqt.shape))
        if labels is not None:
            st.text('Label\'s shape: {}'.format(labels.shape))

        cqt_view_widget = st.empty()
        label_view_widget = st.empty()
        
        # CQT

        #slider = st.slider('Offset', max_value=get_cqt_length('sl_data/tmp/train_X.npy') - 800, step=1)
        #pre_processed_CQT = load_preprocessed_CQT('sl_data/tmp/train_X.npy', 800, offset=slider)

        if architecture_selected != 'cnn':
            slider = st.slider('Offset', max_value=cqt.shape[1] - 800, step=1)
            pre_processed_CQT = prepare_view_CQT(cqt, 800, offset=slider)
        else:
            slider = st.slider('Offset', max_value=cqt_view.shape[1] - 800, step=1)
            pre_processed_CQT = prepare_view_CQT(cqt_view, 800, offset=slider)

        #### !!! Get min_val from file
        min_val = np.min(pre_processed_CQT) if include_negatives else 0
        
        #st.write(min_val)
        # Color ramp attempt

        image = np.clip((pre_processed_CQT - np.clip(min_val, np.NINF, 0.0)) * brightness_multiplier + brightness_offset, 0.0, 1.0)
        if enable_color_ramp:
            image = color_ramp(image)
        cqt_view_widget.image(image, use_column_width=True)

        # Labels

        if labels is not None:
            #pre_processed_CQT2 = load_preprocessed_CQT('sl_data/model1/train_Y.npy', 800, offset=slider)
            if architecture_selected != 'cnn':
                pre_processed_labels = prepare_view_CQT(labels, 800, offset=slider)
            else:
                pre_processed_labels = prepare_view_CQT(labels_view, 800, offset=slider)
            label_image = np.clip(pre_processed_labels, 0.0, 1.0)
            label_view_widget.image(label_image, use_column_width=True)
        else:
            label_view_widget.warning('Labels are not supported for this file')

        # Running
        if (st.button('Run Model')):
            model_status = st.text('[Model Status]')
            loaded_model = loader.load_model_sl(selected_model_path + '.h5', model_status)

def color_ramp(data):
    colors = [
        [0, 0.258, 0.615],
        [0.180, 0.349, 0.658],
        [0.278, 0.443, 0.698],
        [0.364, 0.541, 0.741],
        [0.450, 0.635, 0.776],
        [0.541, 0.737, 0.811],
        [0.647, 0.835, 0.847],
        [0.772, 0.929, 0.874],
        [1, 1, 0.878],  
    ]
    
    mapped_data = np.clip(data / (1.0 / len(colors)), 0, len(colors) - 1)
    mapped_data = mapped_data.astype(np.int)
    image = np.take(colors, mapped_data, axis=0)

    return image


def parse_file_list(files):
    keys = []
    losses = {}

    for file_name in files:
        if file_name[:5] == 'model':
            losses_name = 'losses_log_{}.h5'.format(file_name[-4])
            if losses_name in files:
                keys.append(file_name[:-3])
                losses[file_name[:-3]] = losses_name[:-3]
                
    return keys, losses
        

@st.cache
def load_losses(file_path):
    with h5py.File(file_path + '.h5', 'r') as h5f:
        return pd.DataFrame(
                            h5f['data'][:],
                            columns=['Epoch #', 'train_loss', 'val_loss', 'Time (s)']
                           )

@st.cache
def load_preprocessed_CQT(file_path, length, offset=0):
    if file_path[-4:] == '.npy':
        data = np.load(file_path, mmap_mode='r')
        if data.shape[0] > data.shape[1]:
            return data[offset:(offset + length), :].T
        else:
            return data[:, offset:(offset + length)]
    else:
        with h5py.File(file_path, 'r') as h5f:
            if h5f['data'].shape[0] > h5f['data'].shape[1]:
                return h5f['data'][offset:(offset + length), :].T
            else:
                return h5f['data'][:, offset:(offset + length)]

def prepare_view_CQT(data, length, offset=0):
    if data.shape[0] > data.shape[1]:
        return data[offset:(offset + length), :].T
    else:
        return data[:, offset:(offset + length)]

def get_cqt_length(file_path):
    if file_path[-4:] == '.npy':
        data = np.load(file_path, mmap_mode='r')
        if data.shape[0] > data.shape[1]:
            return data.shape[0]
        else:
            return data.shape[1]
    else:
        with h5py.File(file_path, 'r') as h5f:
            if h5f['data'].shape[0] > h5f['data'].shape[1]:
                return h5py.Dataset.len(h5f['data'])
            else:
                return h5py.Dataset.len(h5f['data'][0])


if __name__ == '__main__':
    main()