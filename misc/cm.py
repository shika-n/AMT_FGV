import matplotlib.pyplot as plt
import numpy as np
import random
import os
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import time
from io import StringIO
import codecs
import h5py

'''
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import Convolution2D, AveragePooling2D, BatchNormalization, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
'''

X = np.load('X_input_shuffled.npy', mmap_mode='r')
Y = np.load('Y_input_shuffled.npy', mmap_mode='r')

indices = np.arange(X.shape[0])

################
K_FOLD = 5
i = 0

###############

test_ratio = 1.0 / K_FOLD
train_ratio = 1.0 - test_ratio
        
numSlicesForTrain = int(X.shape[0] * train_ratio)
numSlicesForTest = X.shape[0] - numSlicesForTrain
test_offset = i * numSlicesForTest
        
train_indices = np.sort(np.concatenate([indices[:test_offset], indices[(test_offset + numSlicesForTest):]]))
test_indices = np.sort(indices[test_offset:(test_offset + numSlicesForTest)])
    
# X_train = X[train_indices] # np.concatenate([X[:test_offset], X[(test_offset + numSlicesForTest):]]) # X_train.shape = (numSlices, 252, 7, 1)
# Y_train = Y[:, train_indices] # np.concatenate([Y[:, :test_offset], Y[:, (test_offset + numSlicesForTest):]], axis=1)
# Y_train = Y_train.T #Y_train = [Y_train[i] for i in range(Y_train.shape[0])] # list of 88 things of shape (numSliceForTrain,)

X_test = X[test_indices] # X_test.shape = (numSlices, 252, 7, 1)
Y_test = Y[:, test_indices]
Y_test = Y_test.T #Y_test = [Y_test[i] for i in range(Y_test.shape[0])] # list of 88 things of shape (numSliceForTrain,)

'''
model = load_model('model.h5')

predictions = np.clip(np.round(model.predict(X_test)), 0, 1)

np.save('predictions.npy', np.asarray(predictions))
'''

predictions = np.load('predictions.npy')

cm = []
print(len(Y_test[0]))
print(np.asarray(Y_test).shape)
print(np.asarray(predictions).shape)
for i in range(len(Y_test[0])):
    cm.append(confusion_matrix(Y_test[:, i], predictions[:, i], labels=[0, 1]))


for i in range(len(cm)):
    plt.imshow(cm[i], cmap=plt.cm.Blues)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title('cm {}'.format(i))
    plt.colorbar()
    plt.savefig('CM{}.png'.format(i))
    plt.clf()

    np.savetxt('CM{}.csv'.format(i), cm[i], fmt='%i', delimiter=',')

