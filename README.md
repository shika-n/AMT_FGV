# AMT_FGV
Automatic Music Transcription

CNN training codes retrieved from [here](https://github.com/mbereket/music-transcription).
DNN/LSTM/BiLSTM codes retrieved from [here](https://github.com/diegomorin8/Deep-Neural-Networks-for-Piano-Music-Transcription).

## Web App Setup
Install dependencies: `pip install numpy tensorflow streamlit librosa h5py pandas matplotlib plotly numba scipy SoundFile`

Put model files in `/sl_data/models/{modelName}_{nEpoch}_{nEarly}`  folder. This includes models and loss logs (`model_{modelName}_{fold}.h5` and `losses_log_{fold}.h5`).

Put music files in `/sl_data/presets` folder. This comes in the form of `.wav` files and optionally `.txt` and `.mid` (for CNN) ground truth files with the same name.

Put standardization or normalization boundaries in `/sl_data/std` folder. The files should be named as `means_stds.h5`, `mean_stds_nm`, and `minmax_meanlabel.h5`

*`losses_log` and `means` files are custom, tweaking might be needed to suit you.

## Running Web App
Run sl.py via streamlit:
`streamlit run sl.py`