import util.preprocess_mlp as preprocess_mlp
import util.preprocess_cnn as preprocess_cnn

import librosa
import numpy as np
import soundfile as sf
import io
import streamlit as st

@st.cache(hash_funcs={st.DeltaGenerator.DeltaGenerator: lambda x : None})
def preprocess(file_path_or_bytes, st_status, architecture_selected):
    if architecture_selected == 'mlp':
        return preprocess_mlp.process(file_path_or_bytes, st_status)
    elif architecture_selected == 'cnn':
        return preprocess_cnn.process(file_path_or_bytes, st_status)