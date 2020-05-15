from os.path import join

import streamlit as st
import numpy as np
import pandas as pd
import h5py

import util.gather as gather

DATA_FOLDER = 'sl_data'

def main():
    st.title('AMT_FGV')

    folders = gather.get_folder_name_list('sl_data')
    st.sidebar.header('Models')
    selected_model = st.sidebar.selectbox('Models available', folders)

    file_names = gather.get_file_name_list(join(DATA_FOLDER, selected_model))
    keys, models = parse_file_list(file_names)
    # st.write(models)

    selected_variant = st.sidebar.selectbox('Variants', keys)

    run_data = load_losses(join(DATA_FOLDER, selected_model, models[selected_variant]))

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
    image_slot = st.empty()
    slider = st.slider('Offset', max_value=get_cqt_length('sl_data/model1/train_X.npy') - 800, step=1)

    pre_processed_CQT = load_preprocessed_CQT('sl_data/model1/train_X.npy', 800, offset=slider)

    st.text('CQT\'s shape: {}'.format(pre_processed_CQT.shape))
    image = (pre_processed_CQT)
    min_val = image.min()
    max_val = image.max()
    image_slot.image((image - min_val) / (max_val - min_val), use_column_width=True)

    image_slot2 = st.empty()
    pre_processed_CQT2 = load_preprocessed_CQT('sl_data/model1/train_Y.npy', 800, offset=slider)

    st.text('Label\'s shape: {}'.format(pre_processed_CQT2.shape))
    image2 = (pre_processed_CQT2)
    min_val2 = image2.min()
    max_val2 = image2.max()
    image_slot2.image((image2 - min_val2) / (max_val2 - min_val2), use_column_width=True)

def parse_file_list(files):
    keys = []
    models = {}

    for file_name in files:
        if file_name[:5] == 'model':
            losses_name = 'losses_log_{}.h5'.format(file_name[-4])
            if losses_name in files:
                keys.append(file_name[:-3])
                models[file_name[:-3]] = losses_name[:-3]
                
    return keys, models
        

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