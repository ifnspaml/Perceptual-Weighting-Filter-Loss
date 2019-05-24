#####################################################################################
# GitHub_mask_dnn_baseline_train -
# Training the BASELINE mask-based DNN model for speech enhancement, i.e., MSE as loss
# Given data:
#       Training input
#       Training unnorm input
#       Training target
#       Validation input
#       Validation unnorm input
#       Validation target
# Output data:
#       Trained DNN model (with MSE loss)
#
# Technische Universität Braunschweig
# Institute for Communications Technology (IfN)
# Schleinitzstrasse 22
# 38106 Braunschweig
# Germany
# 2019 - 05 - 23
# (c) Ziyue Zhao
#
# Use is permitted for any scientific purpose when citing the paper:
# Z. Zhao, S. Elshamy, and T. Fingscheidt, "A Perceptual Weighting Filter
# Loss for DNN Training in Speech Enhancement", arXiv preprint arXiv:
# 1905.09754.
#####################################################################################


import numpy as np
np.random.seed(1337)  # for reproducibility
from numpy import random
import os
import tensorflow as tf
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import *
from keras import backend as K
import keras.optimizers as optimizers
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard, LearningRateScheduler
import keras.callbacks as cbs
from numpy import random
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.io.wavfile as swave
from sklearn import preprocessing
import math
import time
from tensorflow.python.framework import ops
from keras.backend.tensorflow_backend import set_session
import h5py
from keras.optimizers import Adam
from sklearn import preprocessing
from keras.constraints import maxnorm


#####################################################################################
# 0. Settings
#####################################################################################
SNR_situ = "6snrs"
fram_length = 129
n1 = 1024 # DNN model parameters
n2 = 512
n3 = 512
drop_out=0.2 # dropou rate

LOOK_BACKWARD = 2
LOOK_FORWARD = 2
INPUT_SHAPE = (fram_length*(LOOK_BACKWARD + 1 + LOOK_FORWARD),)
INPUT_SHAPE2 = (fram_length,)

# Look to the past and to the future
def reshapeDataMatrix(data_matrix, look_backward=1, look_forward=2):
    new_dim_len = look_backward + look_forward + 1
    data_matrix_out = np.zeros((data_matrix.shape[0], data_matrix.shape[1], new_dim_len))
    for i in range(0, data_matrix.shape[0]):
        for j in range(-look_backward, look_forward + 1):
            if i < look_backward:
                idx = max(i + j, 0)
            elif i >= data_matrix.shape[0] - look_forward:
                idx = min(i + j, data_matrix.shape[0] - 1)
            else:
                idx = i + j
            data_matrix_out[i, :, j + look_backward] = data_matrix[idx, :]

    return data_matrix_out


#####################################################################################
# 1. load data
#####################################################################################
print('> Loading data... ')

# Load Input Data
print('  >> Loading input data... ')
mat_input = "./training data/training_input_" + SNR_situ + ".mat"
file_h5py_train_input = h5py.File(mat_input,'r')
x_train_noisy = file_h5py_train_input.get('training_input')
x_train_noisy = np.array(x_train_noisy)  # For converting to numpy array
x_train_noisy = np.transpose(x_train_noisy)
print('  >> Reshaping input data... ')
x_train_noisy = reshapeDataMatrix(x_train_noisy, look_backward=LOOK_BACKWARD, look_forward=LOOK_FORWARD)
print('     Input data shape: %s' % str(x_train_noisy.shape))

# Load Input Data for Validation
print('  >> Loading input validation data... ')
mat_input_vali = "./training data/validation_input_" + SNR_situ + ".mat"
file_h5py_vali_input = h5py.File(mat_input_vali,'r')
x_train_noisy_vali = file_h5py_vali_input.get('validation_input')
x_train_noisy_vali = np.array(x_train_noisy_vali)
x_train_noisy_vali = np.transpose(x_train_noisy_vali)
print('  >> Reshaping input validation data... ')
x_train_noisy_vali = reshapeDataMatrix(x_train_noisy_vali, look_backward=LOOK_BACKWARD, look_forward=LOOK_FORWARD)
print('     Input validation data shape: %s' % str(x_train_noisy_vali.shape))

# load auxiliary input
print('  >> Loading auxiliary input data... ')
mat_input_aux = "./training data/training_input_unnorm_" + SNR_situ + ".mat"
file_h5py_train_input_aux = h5py.File(mat_input_aux,'r')
x_train_noisy_aux = file_h5py_train_input_aux.get('training_input')
x_train_noisy_aux = np.array(x_train_noisy_aux)  # For converting to numpy array
x_train_noisy_aux = np.transpose(x_train_noisy_aux)
print('     Input data shape: %s' % str(x_train_noisy_aux.shape))

# Load auxiliary input Data for Validation
print('  >> Loading auxiliary input validation data... ')
mat_input_vali_aux = "./training data/validation_input_unnorm_" + SNR_situ + ".mat"
file_h5py_vali_input_aux = h5py.File(mat_input_vali_aux,'r')
x_train_noisy_vali_aux = file_h5py_vali_input_aux.get('validation_input')
x_train_noisy_vali_aux = np.array(x_train_noisy_vali_aux)
x_train_noisy_vali_aux = np.transpose(x_train_noisy_vali_aux)
print('     Input validation data shape: %s' % str(x_train_noisy_vali_aux.shape))

# Load Target Data
print('  >> Loading target data... ')
mat_target = "./training data/training_target_" + SNR_situ + ".mat"
training_target = h5py.File(mat_target,'r')
x_train = training_target.get('training_target')
x_train = np.array(x_train)
x_train = np.transpose(x_train)
print('     Target data shape: %s' % str(x_train.shape))

# Load Target Data for Validation
print('  >> Loading target validation data... ')
mat_target_vali = "./training data/validation_target_" + SNR_situ + ".mat"
mat_target_vali = h5py.File(mat_target_vali,'r')
x_train_vali = mat_target_vali.get('validation_target')
x_train_vali = np.array(x_train_vali)
x_train_vali = np.transpose(x_train_vali)
print('     Target validation data shape: %s' % str(x_train_vali.shape))

# Reshape input data
x_train_noisy = np.reshape(x_train_noisy, (x_train_noisy.shape[0], fram_length*(LOOK_BACKWARD + 1 + LOOK_FORWARD)),order='F')
print('     Input data shape: %s' % str(x_train_noisy.shape))
x_train_noisy_vali = np.reshape(x_train_noisy_vali, (x_train_noisy_vali.shape[0], fram_length*(LOOK_BACKWARD + 1 + LOOK_FORWARD)),order='F')
print('     Input validation data shape: %s' % str(x_train_noisy_vali.shape))

# Prepare weighting factors for frequency bins in loss: As only half-plus-one frequency bins are input to NN, loss is
# justified by this weighting factors as energy normalization (see (5) in the paper).
f_train_wfac = np.ones_like(x_train_noisy_aux)
f_train_wfac[:, 1:fram_length-1] = int(2)
print('     weighting factors for frequency bins: %s' % str(f_train_wfac.shape))
f_vali_wfac = np.ones_like(x_train_noisy_vali_aux)
f_vali_wfac[:, 1:fram_length-1] = int(2)
print('     weighting factors for frequency bins: %s' % str(f_vali_wfac.shape))

# Energy normalization for training/validation target
x_train_wfac = np.multiply(x_train, f_train_wfac)
x_train_vali_wfac = np.multiply(x_train_vali, f_vali_wfac)

print('> Data Loaded. Compiling...')



#####################################################################################
# 2. define model
#####################################################################################
input_img = Input(shape=(INPUT_SHAPE))
auxiliary_input = Input(shape=(INPUT_SHAPE2)) # unnorm input data
wfac_input = Input(shape=(INPUT_SHAPE2)) # weighting factors for frequency bins energy normalization in loss

d1 = Dense(n1)(input_img)
d1 =BatchNormalization()(d1)
d1 = LeakyReLU(0.2)(d1)
d1 =Dropout(0.2)(d1)

d2 = Dense(n2)(d1)
d2 =BatchNormalization()(d2)
d2 = LeakyReLU(0.2)(d2)
d2=Dropout(0.2)(d2)

d3 = Dense(n3)(d2)
d3 =BatchNormalization()(d3)
d3 = LeakyReLU(0.2)(d3)
d3 =Dropout(0.2)(d3)

m1 = Add()([d2, d3])
d4 = Dense(n2)(m1)
d4 =BatchNormalization()(d4)
d4 = LeakyReLU(0.2)(d4)
d4 =Dropout(0.2)(d4)

m2 = Add()([d2, d3, d4])
d5 = Dense(256)(m2)
d5 =BatchNormalization()(d5)
d5 = LeakyReLU(0.2)(d5)
d5 =Dropout(0.2)(d5)

d6 = BatchNormalization()(d5)
mask= Dense(129,activation='sigmoid')(d6)

# Use the predicted mask to multiply the unnorm data
decoded= Multiply()([mask,auxiliary_input])

# Weighting factors for frequency bins in loss (energy normalization)
decoded_wgh_factor = Multiply()([decoded,wfac_input])

model = Model(inputs=[input_img, auxiliary_input, wfac_input], outputs=decoded_wgh_factor)
model.summary()

# Training settings
nb_epochs = 100
batch_size = 128
learning_rate = 5e-4
adam_wn = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=adam_wn, loss='mean_squared_error', metrics=['accuracy'])

#####################################################################################
# 3. Fit the model
#####################################################################################
# Stop training after 16 epoches if the vali_loss not decreasing
stop_str = cbs.EarlyStopping(monitor='val_loss', patience=16, verbose=1, mode='auto')
# Reduce learning rate when stop improving lr = lr*factor
reduce_LR = cbs.ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
# Only save the best model
best_weights = "./training results/mask_dnn_baseline_" + SNR_situ + "_weights.h5"
best_weights = os.path.normcase(best_weights)
model_save = cbs.ModelCheckpoint(best_weights, monitor='val_loss', save_best_only=True, mode='auto', save_weights_only=True, period=1)
# Start to fit model
start = time.time()
print("> Training model " + "using Batch-size: " + str(batch_size) + ", Learning_rate: " + str(learning_rate) + "...")
hist = model.fit([x_train_noisy,x_train_noisy_aux,f_train_wfac], x_train_wfac, epochs=nb_epochs, verbose=2, batch_size=batch_size, shuffle=True, initial_epoch=0,
                      callbacks=[reduce_LR, stop_str, model_save],
                      validation_data=[[x_train_noisy_vali,x_train_noisy_vali_aux,f_vali_wfac], x_train_vali_wfac]
                      )
# Save validation loss
ValiLossVec='./training results/mask_dnn_baseline_' + 'validationloss.mat'
ValiLossVec = os.path.normcase(ValiLossVec) # directory
sio.savemat(ValiLossVec, {'Vali_loss_Vec': hist.history['val_loss']})
# Fit complete
print("> Saving Completed, Time : ", time.time() - start)
print('> +++++++++++++++++++++++++++++++++++++++++++++++++++++ ')

