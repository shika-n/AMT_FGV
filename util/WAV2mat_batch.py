import sys
sys.path.append('')
print(sys.path)

import numpy as np
import sys
import os
import librosa
import matplotlib.pyplot as plt
import h5py
from scipy.io import wavfile,savemat
#from python_speech_features import mfcc

from util.file_handler import auto_load

# Parameters
hop_length_in = 512
n_bins_in = 252
bins_octaves_in = 36
win_step = 0.01
number_notes = 88
num_cep_def = 40
num_filt_def = 40
length_per_file = 4000000

# name of the .lst file
#source_List = 'Preprocessing/train_tr/train.lst' # sys.argv[1];
# source_WAV = sys.argv[2];
# source_Txt = sys.argv[3];
# out_mat = sys.argv[2];



# Get the name of the list
#source_list_split = source_List.split('.')
#source_list_split = source_list_split[0].split('/')
#list_name = source_list_split[-1]

# Open the .lst file
#file_List = open( source_List , "r")

def process(file_path_or_path, st_status):
    # Output .npz
    train2mat = []
    labels2mat = []
    contador = 0

    file_List = [file_path_or_path]

    # Iterate on every file
    for i, filename in enumerate(file_List):
        #if (i > 10):
        #    break
        if isinstance(file_path_or_path, str) and os.path.isfile(file_path_or_path[:-4] + '.txt'):
            print('[{}] {}'.format(i, filename[:-4]))
            #os.chdir('D:\Tugas dan Bahan ulangan\Kuliah\SKRIPSIKIRBY\lstm\LSTM\Preprocessing\MUS') #ganti directory  di tmpt .wav MUS

            filename_split = filename[:-4] #.split('.')

        #### MFCC extraction ####
        # Transform to raw data from wav. Get the sampling rate 2
        # Renamed stereo_vector to mono_vector
        #sampling_freq, stereo_vector = wavfile.read(filename_split + '.wav')
        stereo_vector, sampling_freq = auto_load(file_path_or_path, sr=None)

        #stereo_vector = stereo_vector[:(sampling_freq * 60), :]
            
        win_len = 512/float(sampling_freq)
        #plt.imshow( np.array(np.absolute(cqt_feat)))
        #plt.show()
        
        # Transform to mono
        # auto_load (librosa) should default to mono
        #mono_vector = np.mean(stereo_vector, axis = 1)
        mono_vector = stereo_vector
        
        # Extract mfcc_features
        cqt_feat = np.absolute(librosa.cqt(mono_vector, sampling_freq, hop_length=hop_length_in,n_bins=n_bins_in,bins_per_octave=bins_octaves_in)).transpose()
        #### LABELING ####
        # Number of frames in the file
        number_Frames = np.max( cqt_feat.shape[0])
        # Aux_Vector of times
        vector_aux = np.arange(1, number_Frames + 1)*win_len
        # Binary labels - we need multiple labels at the same time to represent the chords
        labels = np.zeros((number_Frames, number_notes))

        if isinstance(file_path_or_path, str) and os.path.isfile(file_path_or_path[:-4] + '.txt'):
            # Open the align txt labels
            #os.chdir('D:\Tugas dan Bahan ulangan\Kuliah\SKRIPSIKIRBY\lstm\LSTM\Preprocessing\MUStext') #ganti directory  di tmpt txt MUS
            file = open(filename_split + '.txt' , "r")
            #f = open(out_mat + filename_split[0] + 'label.lst','w')
            # Loop over all the lines 
            for line in file: 
                line_split = line.split()
                if len(line_split) != 3:
                    continue
                if line_split[0] == "OnsetTime":
                    print ("Preprocessing operations . . .")
                else:
                # Get the values from the text
                    init_range, fin_range, pitch = float(line_split[0]), float(line_split[1]), int(line_split[2])
                    # Pitch move to 0-87 range
                    pitch = pitch - 21
                    # Get the range indexes
                    index_min = np.where(vector_aux >= init_range)
                    index_max = np.where(vector_aux - 0.01 > int((fin_range)*100)/float(100))
                    
                    if vector_aux[-1] - 0.01 <= int((fin_range) * 100) / float(100):
                        index_max = np.zeros([1, 1], dtype=np.int)
                        index_max[0][0] = int(len(vector_aux) - 1)
                        
                    if vector_aux[-1] < init_range:
                        break
                    
                    labels[index_min[0][0]:index_max[0][0],pitch] = 1
            
            #If you want to save the labels to a txt file 
            """for i in range( number_Frames):
                for j in range( 88 ):
                    if labels[i][j] == 1:
                        f.write('%f' %vector_aux[i] + ' - ' + '%d\n' %j)
            f.close()
            """
            file.close()
        else:
            labels = None

        train2mat.extend(cqt_feat)
        if labels is not None:
            labels2mat.extend(labels)
        
        """
        plt.figure()
        plt.imshow( np.array(labels.transpose()),aspect='auto')
        plt.figure()
        plt.imshow( np.array(np.absolute(cqt_feat)), aspect='auto')
        plt.show()
        """
        '''
        # ni gw gtw buat ap
        while (len(train2mat) + len(cqt_feat)) >= length_per_file:
            size_to_add = length_per_file - len(train2mat)
            # Append to add to npz
            train2mat.extend(cqt_feat[0:size_to_add,:])
            # Append the labels 
            labels2mat.extend(labels[0:size_to_add,:])
            train2mat = np.array(train2mat)
            labels2mat = np.array(labels2mat)
            # Plotting stuff
            print (" Shape of MFCC is " + str(train2mat.shape) + " - Saved in " + out_mat + list_name + '/' + str(contador) + list_name)
            print (" Shape of Labels is " + str(labels2mat.shape)  + " - Saved in " + out_mat + list_name + '/' + str(contador) + list_name)
            np.save('{}_X'.format(out_mat + list_name + '/' + str(contador) + list_name ), train2mat)
            np.save('{}_y'.format(out_mat + list_name + '/' + str(contador) + list_name), labels2mat)
            print("test")
            contador = contador + 1;
            train2mat = []
            labels2mat = []
            cqt_feat = cqt_feat[size_to_add:,:]
            labels = labels[size_to_add:,:]
        
        # ni gw gtw buat ap
        if len(cqt_feat) == length_per_file:
            # Append to add to npz
            train2mat.extend(cqt_feat)
            # Append the labels 
            labels2mat.extend(labels)
            train2mat = np.array(train2mat)
            labels2mat = np.array(labels2mat)
            # Plotting stuff
            print (" Shape of MFCC is " + str(train2mat.shape)  + " - Saved in " + out_mat + list_name + '/' + str(contador) + list_name)
            print (" Shape of Labels is " + str(labels2mat.shape)  + " - Saved in " + out_mat + list_name + '/' + str(contador) + list_name)
            np.save('{}_X'.format(out_mat + list_name + '/' + str(contador) + list_name ), train2mat)
            np.save('{}_y'.format(out_mat + list_name + '/' + str(contador) + list_name), labels2mat)

            contador = contador + 1;
            train2mat = []
            labels2mat = []
        elif len(cqt_feat) > 0:
            # Append to add to npz
            train2mat.extend(cqt_feat)
            # Append the labels 
            labels2mat.extend(labels)
        '''

    train2mat = np.array(train2mat)

    if labels is None:
        labels2mat = None
    else:
        labels2mat = np.array(labels2mat)
    """
    plt.figure()
    plt.imshow( np.array(labels2mat.transpose()),aspect='auto')
    plt.colorbar()
    plt.figure()
    plt.imshow( np.array(train2mat.transpose()), aspect='auto')
    plt.colorbar()
    plt.show()
    """
    # Plotting stuff
    # print (" Shape of MFCC is " + str(train2mat.shape)  + " - Saved in " + list_name + '/' + str(contador) + list_name)
    # print (" Shape of Labels is " + str(labels2mat.shape)  + " - Saved in " + list_name + '/' + str(contador) + list_name)

    #os.chdir('D:\Tugas dan Bahan ulangan\Kuliah\SKRIPSIKIRBY\lstm\LSTM\Preprocessing\\train_tr') #save directory 
    #np.save('Preprocessing/train_tr/train_X', train2mat) #ganti jd train_x eval_x test_x 
    #np.save('Preprocessing/train_tr/train_y', labels2mat) #ganti jd train_y eval_y

    print("Done")
    #print ("saved in D:\Tugas dan Bahan ulangan\Kuliah\SKRIPSIKIRBY\lstm\LSTM\Preprocessing\\train_tr")

    with h5py.File('sl_data/std/minmax_meanlabel.h5', 'r') as h5f:
        # Normalization - From mat2nrom_batch.py
        #max_train = train2mat.max()
        #min_train = train2mat.min()

        #train_mean = np.mean(train2mat, axis=0)

        min_train = h5f['min_max'][0]
        max_train = h5f['min_max'][1]
        train_mean = (h5f['mean_per_label'][:]).T

        X_train_norm = (train2mat - min_train)/(max_train - min_train) 
        X_train_norm = X_train_norm - train_mean

    #return train2mat, labels2mat
    return X_train_norm, labels2mat