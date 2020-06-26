from os.path import join

import streamlit as st
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import plotly.express as px
import librosa
import io
import html
from numba import jit

import base64

from MidiUtil import MIDIFile

import SessionState
import util.model_loader as loader
import util.preprocessing_handler as preprocessing_handler
import util.file_handler as file_handler
from util.pred2sheet import pred2sheet 

MODEL_FOLDER = 'sl_data/models'
PRESET_FOLDER = 'sl_data/presets'

presets = {
    'Chopin 15': 'MAPS_MUS-chpn-p15_ENSTDkAm.wav',
    'Mozart Turkish March': 'Mozart - Rondo Alla Turca (Turkish March) (online-audio-converter.com).wav',
    'Chopin 14': 'MAPS_MUS-chpn-p14_ENSTDkAm.wav',
}

VIEW_WIDTH = 800

def main():
    session_state = SessionState.get(
        last_predictions = None,
        last_post_predictions = None,
        cqt = None,
        labels = None,
        cqt_view = None,
        labels_view = None,
        last_config = ""
    )

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

    #include_negatives = st.sidebar.checkbox('Include negatives (Preview only)')
    #brightness_multiplier = st.sidebar.slider('Brightness multiplier (Preview only)', 1.0, 10.0, value=1.0)
    #brightness_offset = st.sidebar.slider('Brightness offset (Preview only)', -2.0, 2.0, value=0.0)
    #enable_color_ramp = st.sidebar.checkbox('Enable color ramp')

    st.sidebar.header('Post-Processing')
    rounding_threshold = st.sidebar.slider('Rounding threshold', 0.0, 1.0, value=0.5, step=0.05)
    minimum_fill = st.sidebar.slider('Minimum fill', 0, 50, value=8)
    gap_threshold = st.sidebar.slider('Gap threshold', 0, 50, value=5)
    post_iter_count = st.sidebar.slider('Post processing iteration count', 0, 5, value = 1)

    # ======== MAIN PAGE ========

    st.title('AMT_FGV')

    if selected_variant is None:
        st.error('Please select a model')
        return

    selected_model_path = join(MODEL_FOLDER, selected_conf_name, selected_variant)
    run_data = load_losses(join(MODEL_FOLDER, selected_conf_name, losses[selected_variant]))

    st.text(selected_model_path)
    st.text('This model is ran for %i epochs.' % (run_data.iloc[-1, 0] + 1))

    # ============== LOSS RAW DATA ====================
    # Change to altair/matplotlib if axis labels are needed
    # https://github.com/streamlit/streamlit/issues/1129
    if st.checkbox('Show loss'):
        st.subheader('Loss Graph')
        st.line_chart(run_data.iloc[:, 1:3])
        #if st.checkbox('Show raw data'):
        #    st.dataframe(run_data.iloc[:])

    ##################################################

    preset_options = ['Upload a file']
    for key in presets:
        preset_options.append(key)

    selected_preset = st.selectbox('Select a preset or upload', preset_options)
    audio_player = st.empty()

    status = st.empty()

    file_bytes = None
    if selected_preset == preset_options[0]:
        file_bytes = st.file_uploader('Upload file', type=['wav', 'txt'])

        if file_bytes is not None:
            if st.button('Load'):
                cqt, labels = preprocessing_handler.preprocess(file_bytes, status, architecture_selected)

                cqt_view, labels_view = cqt, labels

                if architecture_selected == 'cnn' or architecture_selected == 'mlp':
                    cqt_view, labels_view = preprocessing_handler.preprocess(file_bytes, status, 'mlp')
                elif architecture_selected == 'lstm' or architecture_selected == 'bilstm' or architecture_selected == 'dnn':
                    cqt_view = cqt_view.T
                    if labels_view is not None:
                        labels_view = labels_view.T

                session_state.cqt = cqt
                session_state.labels = labels
                session_state.cqt_view = cqt_view
                session_state.labels_view = labels_view
    else:
        file_path = join(PRESET_FOLDER, presets[selected_preset])
        file_bytes = file_handler.get_file_bytes(file_path)

        if st.button('Load'):
            cqt, labels = preprocessing_handler.preprocess(file_path, status, architecture_selected)

            cqt_view, labels_view = cqt, labels

            if architecture_selected == 'cnn' or architecture_selected == 'mlp':
                cqt_view, labels_view = preprocessing_handler.preprocess(file_path, status, 'mlp')
            elif architecture_selected == 'lstm' or architecture_selected == 'bilstm' or architecture_selected == 'dnn':
                    cqt_view = cqt_view.T
                    if labels_view is not None:
                        labels_view = labels_view.T

            session_state.cqt = cqt
            session_state.labels = labels
            session_state.cqt_view = cqt_view
            session_state.labels_view = labels_view

    if file_bytes is not None:
        audio_player.audio(file_bytes)

    cqt = session_state.cqt
    labels = session_state.labels
    cqt_view = session_state.cqt_view
    labels_view = session_state.labels_view

    config = selected_model_path
    if file_bytes is not None:
        if isinstance(file_bytes, io.BytesIO):
            config += base64.b64encode(file_bytes.read()).decode()[:32]
        else:
            config += base64.b64encode(file_bytes).decode()[:32]
    
    if config != session_state.last_config:
        session_state.last_predictions = None
        session_state.last_post_predictions = None
        session_state.cqt = None
        session_state.labels = None
        session_state.cqt_view = None
        session_state.labels_view = None

        session_state.last_config = config

        return

    # ================= Data freq =====================
    #data = np.load('sl_data/tmp/Y_input_shuffled-nm.npy', mmap_mode='r')
    #if data.shape[0] > data.shape[1]:
    #    data = data[:].T
    #else:
    #    data = data[:]

    #st.write(data.shape)
    #freq = np.sum(np.clip(data, 0, 1), axis=1)
    #st.write(freq.shape)

    ##zeroes = np.sum(np.abs(np.clip(data, 0, 1) -1), axis=1)
    #zeroes = np.zeros(data.shape[0]) + data.shape[1]

    #st.write(freq.shape)

    #ones_zeroes_data = np.asarray([zeroes, freq]).T
    #st.write(ones_zeroes_data.shape)

    #ones_zeroes = pd.DataFrame(
    #    ones_zeroes_data,
    #    columns=['zeroes', 'ones']
    #)

    #st.bar_chart(ones_zeroes)

    #st.write((freq / data.shape[1]) * 100)
    


    #########

    if cqt is not None:
        #st.text('CQT\'s shape: {}'.format(cqt.shape))
        #if labels is not None:
        #    st.text('Label\'s shape: {}'.format(labels.shape))
        st.text('Label')

        cqt_view_widget = st.empty()
        label_view_widget = st.empty()
        
        # CQT

        #slider = st.slider('Offset', max_value=get_cqt_length('sl_data/tmp/train_X.npy') - 800, step=1)
        #pre_processed_CQT = load_preprocessed_CQT('sl_data/tmp/train_X.npy', 800, offset=slider)
        
        long_width = int(np.max(cqt_view.shape))
        if long_width - VIEW_WIDTH <= 0:
            slider = 0
        else:
            slider = st.slider('Offset', max_value=long_width - VIEW_WIDTH, step=1)
        #view_cqt = prepare_view_data(cqt_view, VIEW_WIDTH, offset=slider)

        #### !!! Get min_val from file
        #min_val = np.min(view_cqt) if include_negatives else 0
        
        # Color ramp attempt

        #image = np.clip((view_cqt - np.clip(min_val, np.NINF, 0.0)) * brightness_multiplier + brightness_offset, 0.0, 1.0)
        #if enable_color_ramp:
        #    image = color_ramp(image)
        #cqt_view_widget.image(image, use_column_width=True)

        #D = librosa.amplitude_to_db(view_cqt[:, slider:(slider+800)], ref=np.max)
        #librosa.display.specshow(D, x_axis='time', y_axis='cqt_note', bins_per_octave=36)
        #st.pyplot()

        #cqt_fig = px.imshow(view_cqt[:, slider:(slider + 800)], origin='lower', zmax=2)
        #cqt_fig.update_layout(autosize=False, width=500, height=400, margin=dict(l=10, r=10, b=10, t=1000))
        #st.plotly_chart(cqt_fig, use_container_width=True)

        # Labels
        if labels_view is not None:
            #pre_processed_CQT2 = load_preprocessed_CQT('sl_data/model1/train_Y.npy', 800, offset=slider)

            pre_processed_labels = prepare_view_data(labels_view, VIEW_WIDTH, offset=slider)
            label_image = np.clip(pre_processed_labels, 0.0, 1.0)
            label_view_widget.image(label_image, use_column_width=True)

            label_fig = px.imshow(label_image[:, slider:(slider + VIEW_WIDTH)], height=220, origin='lower')
            #st.plotly_chart(label_fig, use_container_width=True)
        else:
            label_view_widget.warning('Labels are not supported for this file')

        # Running
        if (st.button('Run Model')):
            model_status = st.text('[Model Status]')
            predictions = np.asarray(loader.run_model_sl(selected_model_path + '.h5', cqt, model_status, architecture_selected))
            session_state.last_predictions = predictions

            # Post
            post_predictions = preprocessing_handler.post_process(predictions, rounding_threshold, minimum_fill, gap_threshold, post_iter_count)
            session_state.last_post_predictions = post_predictions

        # ============ RESULT ===============
        post_last_run_widget = st.empty()
        if session_state.last_predictions is not None:
            last_predictions = session_state.last_predictions
            prepared_predictions = prepare_view_data(last_predictions, VIEW_WIDTH, offset=slider)
            
            st.text('Last run result')
            st.image(prepared_predictions, use_column_width=True)

        if session_state.last_post_predictions is not None:
            if st.button('Re-run Post Processing'):
                session_state.last_post_predictions = preprocessing_handler.post_process(last_predictions, rounding_threshold, minimum_fill, gap_threshold, post_iter_count)

            post_predictions = session_state.last_post_predictions
            post_prepared_predictions = prepare_view_data(post_predictions, VIEW_WIDTH, offset=slider)
            
            st.text('Last run after post process')
            st.image(post_prepared_predictions, use_column_width=True)
            
            if st.button('Process Sheet'):
                byte_data = create_midi(post_predictions, architecture_selected)
                lily_pond = pred2sheet(post_predictions)

                #np.save('post_predictions.npy', post_predictions)
                st.info('Click on the URL to download')
                st.markdown(get_download_link(byte_data, 'midi_result.mid','Download MIDI'), unsafe_allow_html=True)
                st.text_area('LilyPond', lily_pond)

def get_download_link(data, target_name, text):
    b64 = base64.b64encode(data)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{target_name}">{text}</a>'

def create_midi(predictions, architecture_selected):
    track    = 0
    channel  = 0
    time     = np.zeros(88)   # In beats|| initialize 88 list
    tempo    = 240   # In BPM           w masi gtw gmn cara nentuin tempo
    volume   = 100  # 0-127, as per the MIDI standard

    ms_per_frame = 22500 / 512

    if architecture_selected == 'cnn' or architecture_selected == 'mlp':
        ms_per_frame = 16000 / 512
        tempo = 60 # 31.25 / 43.95

    miditrack = MIDIFile(1)  # One track
    miditrack.addTempo(track, 0, tempo)

    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    print ("\nSaving track to MIDI . . .")
    for timeframe in range(predictions.shape[0]):
        for pitch in range(predictions.shape[1]):
            current_time = timeframe * (ms_per_frame / 1000)
            if predictions[timeframe,pitch] == 1 and predictions[timeframe-1,pitch] == 0:
                time[pitch] = current_time
            if predictions[timeframe, pitch] == 0 and predictions[timeframe-1,pitch] == 1:
                duration = (current_time - time[pitch])
                miditrack.addNote(track, channel, pitch + 21, time[pitch], duration, volume)

    #with open("result.mid", "wb") as output_file:
    #    miditrack.writeFile(output_file)
    #print('saved')
    return miditrack.getBytes()

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

def prepare_view_data(data, length, offset=0):
    if data.shape[0] > data.shape[1]:
        return data[offset:(offset + length), :].T
    else:
        return data[:, offset:(offset + length)]

if __name__ == '__main__':
    main()
