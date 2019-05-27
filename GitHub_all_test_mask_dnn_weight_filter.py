#####################################################################################
# GitHub_all_test_mask_dnn_weight_filter -
# Test stage for WEIGHTING FILTER mask-based DNN, generates the magnitude of FFT coeff.
# for white- and black-box measurements.
# Given data:
#       Input Data: y_norm
#       Auxiliary input: y (noisy speech)
#       Auxiliary input: s (clean speech)
#       Auxiliary input: n (noise speech)
# Output data:
#       s_hat  -> y_norm to generate mask and y as auxiliray input;
#       s_tilt -> y_norm to generate mask and s as auxiliray input;
#       n_tilt -> y_norm to generate mask and n as auxiliray input;
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
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard, LearningRateScheduler
import keras.callbacks as cbs
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.io.wavfile as swave
import math
import time
from tensorflow.python.framework import ops
from keras.backend.tensorflow_backend import set_session
import h5py
from keras.optimizers import Adam
from sklearn import preprocessing
from keras.constraints import maxnorm


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
# 0. Settings
#####################################################################################

# Define string array for all test SNRs
SNR_situ_array = ["-21","-26","-31","-36","-41","-46"] #

# Loop for all test SNRs
for k_snr in range(0, len(SNR_situ_array)):

    # Settings
    SNR_situ = SNR_situ_array[k_snr]
    noi_situ_model_str = "6snrs"
    filter_type_str = "AMR_direct_freqz"
    fram_length = 129
    n1 = 1024
    n2 = 512
    n3 = 512
    drop_out=0.2

    LOOK_BACKWARD = 2
    LOOK_FORWARD = 2
    INPUT_SHAPE = (fram_length*(LOOK_BACKWARD + 1 + LOOK_FORWARD),)
    INPUT_SHAPE2 = (fram_length,)

    #####################################################################################
    # 1. Model define
    #####################################################################################
    input_img = Input(shape=(INPUT_SHAPE))
    auxiliary_input = Input(shape=(INPUT_SHAPE2))
    h_filter_input  = Input(shape=(INPUT_SHAPE2))
    wfac_input = Input(shape=(INPUT_SHAPE2))  # weighting factors for frequency bins energy normalization in loss

    sclae_input=Lambda(lambda x: x /100 )(auxiliary_input)

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

    d6 =BatchNormalization()(d5)
    mask= Dense(129,activation='sigmoid')(d6)

    # Use the predicted mask to multiply the unnorm data
    decoded = Multiply()([mask, auxiliary_input])

    # Filter the enhanced data by weighting filter
    decoded_filt = Multiply()([decoded, h_filter_input])

    # Weighting factors for frequency bins in loss (energy normalization)
    decoded_filt_wgh_factor = Multiply()([decoded_filt, wfac_input])

    model = Model(inputs=[input_img, auxiliary_input, h_filter_input, wfac_input], outputs=decoded_filt_wgh_factor)
    model.summary()

    # Settings
    nb_epochs = 100
    batch_size = 128
    learning_rate = 5e-4
    adam_wn = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam_wn, loss='mean_squared_error', metrics=['accuracy'])
    model.load_weights("./training results/mask_dnn_weight_filter_" + filter_type_str + "_" + noi_situ_model_str+ "_weights.h5")

    #####################################################################################
    # 2. load data
    #####################################################################################

    # Load Input Data: y_norm
    print('> Loading input data: y_norm... ')
    mat_input ="./test data/test_input_y_abs_snr_" + SNR_situ + "_model_" + noi_situ_model_str + "_test_data.mat"
    mat_input = os.path.normcase(mat_input)
    x_test_noisy = sio.loadmat(mat_input)
    x_test_noisy = x_test_noisy['test_input_y']
    x_test_noisy = np.array(x_test_noisy)
    print('  >> Reshaping test input data... ')
    x_test_noisy = reshapeDataMatrix(x_test_noisy, look_backward=LOOK_BACKWARD, look_forward=LOOK_FORWARD)
    print(x_test_noisy.shape)
    x_test_y_norm = np.reshape(x_test_noisy, (x_test_noisy.shape[0], fram_length*(LOOK_BACKWARD + 1 + LOOK_FORWARD)),order='F')
    print('     Input data y_norm shape: %s' % str(x_test_y_norm.shape))

    # load auxiliary input: y
    print('> Loading auxiliary input data: y... ')
    mat_input_aux_y ="./test data/test_input_abs_unnorm_snr_" + SNR_situ + "_model_" + noi_situ_model_str + "_test_data.mat"
    mat_input_aux_y = os.path.normcase(mat_input_aux_y)
    x_test_noisy_aux_y = sio.loadmat(mat_input_aux_y)
    x_test_noisy_aux_y = x_test_noisy_aux_y['test_input_abs_unnorm']
    x_test_y = np.array(x_test_noisy_aux_y)
    print('     Input data y shape: %s' % str(x_test_y.shape))

    # load auxiliary input: s
    print('> Loading auxiliary input data: s... ')
    mat_input_aux ="./test data/test_input_s_snr_" + SNR_situ + "_model_" + noi_situ_model_str + "_test_data.mat"
    mat_input_aux = os.path.normcase(mat_input_aux)
    x_test_noisy_aux = sio.loadmat(mat_input_aux)
    x_test_noisy_aux = x_test_noisy_aux['test_input_s']
    x_test_s = np.array(x_test_noisy_aux)
    print('     Input data s shape: %s' % str(x_test_s.shape))

    # load auxiliary input: n
    print('> Loading auxiliary input data: n... ')
    mat_input_aux_n ="./test data/test_input_n_snr_" + SNR_situ + "_model_" + noi_situ_model_str + "_test_data.mat"
    mat_input_aux_n = os.path.normcase(mat_input_aux_n)
    x_test_noisy_aux_n = sio.loadmat(mat_input_aux_n)
    x_test_noisy_aux_n = x_test_noisy_aux_n['test_input_n']
    x_test_n = np.array(x_test_noisy_aux_n)
    print('     Input data n shape: %s' % str(x_test_n.shape))

    # prepare weights for test: ALL ONES, same dimension as auxiliary input
    x_test_h_ones = np.ones_like(x_test_noisy_aux)
    print('     Weighting filter shape (all ones): %s' % str(x_test_h_ones.shape))

    # Energy normalization as ALL ONES
    x_test_wfac_ones = np.ones_like(x_test_noisy_aux)

    #####################################################################################
    # 3. predict model and save results
    #####################################################################################
    predicted_s_hat = model.predict([x_test_y_norm,x_test_y,x_test_h_ones,x_test_wfac_ones])
    print(predicted_s_hat.shape)
    recon_file = "./test results/mask_dnn_weight_filter_" + filter_type_str + "_s_hat_snr_" + SNR_situ + "_model_" + noi_situ_model_str + "_test_data.mat"
    recon_file = os.path.normcase(recon_file)
    sio.savemat(recon_file, {'test_s_hat':predicted_s_hat})

    predicted_s_tilt = model.predict([x_test_y_norm,x_test_s,x_test_h_ones,x_test_wfac_ones])
    print(predicted_s_tilt.shape)
    recon_file = "./test results/mask_dnn_weight_filter_" + filter_type_str + "_s_tilt_snr_" + SNR_situ + "_model_" + noi_situ_model_str + "_test_data.mat"
    recon_file = os.path.normcase(recon_file)
    sio.savemat(recon_file, {'test_s_tilt':predicted_s_tilt})

    predicted_n_tilt = model.predict([x_test_y_norm,x_test_n,x_test_h_ones,x_test_wfac_ones])
    print(predicted_n_tilt.shape)
    recon_file = "./test results/mask_dnn_weight_filter_" + filter_type_str + "_n_tilt_snr_" + SNR_situ + "_model_" + noi_situ_model_str + "_test_data.mat"
    recon_file = os.path.normcase(recon_file)
    sio.savemat(recon_file, {'test_n_tilt':predicted_n_tilt})

    print('>>> Finishing of SNR: ' + SNR_situ + ', filter type: ' + filter_type_str + '...')

