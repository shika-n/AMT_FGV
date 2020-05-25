import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session

def load_model_sl(file_name, st_status):
    st_status.text('Loading model... ({})'.format(file_name))
    
    model = load_model(file_name)
    
    st_status.info('Model loaded!')

    # del model
    clear_session()

    return model