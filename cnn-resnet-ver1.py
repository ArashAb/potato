import numpy as np
import random
import os

from scipy.io import loadmat
import re
from os.path import isfile, join
from os import listdir

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import MaxPooling1D, Dense, Activation, Reshape, Dropout, Add, \
    Concatenate, UpSampling1D, Cropping1D, ZeroPadding1D, Conv1D, TimeDistributed
from tensorflow.keras.layers import Input, Flatten, SeparableConv1D, BatchNormalization, LSTM, GRU, Bidirectional, \
    SpatialDropout1D, AveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import Callback, ModelCheckpoint

from tensorflow.keras.metrics import Precision, TruePositives, AUC
from pathlib import Path

import logging
l_1_norm = 1e-7
l_2_norm = 1e-7
SEED_VAL = 10


def set_tf_log_level(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)


set_tf_log_level(logging.FATAL)

DROPOUT = 0

learning_rate = 1e-7
checkpoint_path_ = '/home/usd.local/arash.abbasi/uwb-multipurpose/code/checkpoints_cnn_ver_seed_10'
# mypath = '/home/usd.local/arash.abbasi/uwb-snr-estimation/matfile_cm1'
mypath = '/home/usd.local/arash.abbasi/uwb-multipurpose/matfiles'
BATCH_SIZE = 200
NUM_EPOCHS = 2000
PERCENTAGE_DATA_TO_USE = 1

Print_MESSAGE = False

LENGHT_TO_USE = 1 * 500  # LENGTH OF THE CHANNELS ARE DIFFERETN, SOME LONG AND SOME SHORT. THEREFORE, I CHOSE 1000 FOR ALL CHANNELS.


# I TRIED TO GET THE MAx LENGTH FOR ALL CHANNELS BUT MOST ELMENST WOULD BE ZERO AND MODEL IS TOO GOOF THEN

def reading_files():
    cm_list = [1, 2, 3, 4]
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    # num_files = len(onlyfiles)
    filename = join(mypath, 'cm1snr0.mat')
    x = loadmat(filename)
    _, num_channel = x['h'].shape
    num_files = len(onlyfiles)
    len_cms = len(cm_list)
    if Print_MESSAGE:
        print('num_channel ', num_channel)
        print('LENGHT_TO_USE ', LENGHT_TO_USE)
        print('num_files ', num_files)
    h_noisy_total_real = np.zeros((len_cms * 16 * num_channel, LENGHT_TO_USE))
    h_noisy_total_imaginary = np.zeros((len_cms * 16 * num_channel, LENGHT_TO_USE))
    snr_list = []
    nn = 0
    toa_list = []
    channel_type_list = []
    for name in onlyfiles:
        cm = int(re.search('cm(.+?)snr', name).group(1))
        # if cm == 1 or cm == 2:  # or cm == 3 or cm == 4:
        if cm in cm_list:
            snr = float(re.search('snr(.+?).mat', name).group(1))
            filename = join(mypath, name)
            # print('cm ', cm)
            x = loadmat(filename)
            h_noisy_transpose = np.transpose(x['h_noisy'])
            h_t0 = np.asarray(x['t0'])[0]
            _, h_len = h_noisy_transpose.shape
            h_noisy_total_real[nn: nn + num_channel, :LENGHT_TO_USE] = h_noisy_transpose.real[:, : LENGHT_TO_USE]
            h_noisy_total_imaginary[nn: nn + num_channel, :LENGHT_TO_USE] = h_noisy_transpose.imag[:, : LENGHT_TO_USE]
            for i in range(num_channel):
                snr_list.append(int((snr + 10) / 2))
                toa_list.append(h_t0[i])
                # if cm % 2 == 0:# channel is NLOS
                #     channel_type_list.append(0)
                # else:
                #     channel_type_list.append(1)
                channel_type_list.append(int(cm) - 1)
            nn = nn + num_channel

    print('unique channel ', len(np.unique(channel_type_list)))
    num_h, _ = h_noisy_total_real.shape
    print('num_h ', num_h)
    np.random.seed(SEED_VAL)

    indexes = np.random.permutation(num_h)
    num_h_new = int(num_h * PERCENTAGE_DATA_TO_USE)

    h_noisy_total_real = h_noisy_total_real[indexes][: num_h_new]
    h_noisy_total_imaginary = h_noisy_total_imaginary[indexes][: num_h_new]
    snr_array = np.array(snr_list)[indexes][: num_h_new]
    toa_array = np.array(toa_list)[indexes][: num_h_new]
    channel_type_list = np.array(channel_type_list)[indexes][: num_h_new]

    snr_array = to_categorical(snr_array, num_classes=16)
    channel_type_list = to_categorical(channel_type_list, num_classes=len(np.unique(channel_type_list)))

    num_h_new, _ = h_noisy_total_real.shape
    num_h_training = int(num_h_new * .8)
    print('snr_array shape', snr_array.shape)
    print('unique snr ', len(np.unique(snr_list)))

    h_noisy_real_validation = h_noisy_total_real[num_h_training:]
    h_noisy_imaginary_validation = h_noisy_total_imaginary[num_h_training:]
    h_noisy_real_training = h_noisy_total_real[:num_h_training]
    h_noisy_imaginary_training = h_noisy_total_imaginary[:num_h_training]
    train_snr_y = snr_array[:num_h_training]
    validation_snr_y = snr_array[num_h_training:]
    train_toa_y = toa_array[:num_h_training]
    validation_toa_y = toa_array[num_h_training:]

    train_channel_type_y = channel_type_list[:num_h_training]
    validation_channel_type_y = channel_type_list[num_h_training:]

    train_x1 = np.expand_dims(h_noisy_real_training, axis=-1)
    train_x2 = np.expand_dims(h_noisy_imaginary_training, axis=-1)
    train_x = np.concatenate((train_x1, train_x2), axis=0)

    validation_x1 = np.expand_dims(h_noisy_real_validation, axis=-1)
    validation_x2 = np.expand_dims(h_noisy_imaginary_validation, axis=-1)
    validation_x = np.concatenate((validation_x1, validation_x2), axis=0)

    train_snr_y = np.concatenate((train_snr_y, train_snr_y), axis=0)
    validation_snr_y = np.concatenate((validation_snr_y, validation_snr_y), axis=0)

    train_channel_type_y = np.concatenate((train_channel_type_y, train_channel_type_y), axis=0)
    validation_channel_type_y = np.concatenate((validation_channel_type_y, validation_channel_type_y), axis=0)

    train_toa_y = np.concatenate((train_toa_y, train_toa_y), axis=0)
    validation_toa_y = np.concatenate((validation_toa_y, validation_toa_y), axis=0)

    if Print_MESSAGE:
        print('num_h_new ', num_h_new)
        print('indexes ', indexes)
        print('num_h ', num_h)
        print('num_h_training ', num_h_training)
        print('snr_array ', snr_array[:10])
        print('one_hot ', snr_array)
        print('one_hot ', snr_array.shape)
        print('snr_array after ', snr_array[:10])
        print('train_snr_y shape ', train_snr_y.shape)
        print('train_snr_y shape ', train_snr_y[:100])
        print('h_t0 array', h_t0)

    return train_x, validation_x, train_snr_y, validation_snr_y, train_toa_y, \
           validation_toa_y, train_channel_type_y, validation_channel_type_y


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv1D(filters=F1, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=None),
               kernel_regularizer=regularizers.l1_l2(l1=l_1_norm, l2=l_2_norm),
               bias_regularizer=regularizers.l2(l_2_norm),
               activity_regularizer=regularizers.l2(l_2_norm))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv1D(filters=F2, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=None),
               kernel_regularizer=regularizers.l1_l2(l1=l_1_norm, l2=l_2_norm),
               bias_regularizer=regularizers.l2(l_2_norm),
               activity_regularizer=regularizers.l2(l_2_norm))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv1D(filters=F3, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=None),
               kernel_regularizer=regularizers.l1_l2(l1=l_1_norm, l2=l_2_norm),
               bias_regularizer=regularizers.l2(l_2_norm),
               activity_regularizer=regularizers.l2(l_2_norm))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv1D(F1, kernel_size=1, strides=s, name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=None), kernel_regularizer=regularizers.l1_l2(l1=l_1_norm, l2=l_2_norm),
               bias_regularizer=regularizers.l2(l_2_norm),
               activity_regularizer=regularizers.l2(l_2_norm))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv1D(filters=F2, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=None), kernel_regularizer=regularizers.l1_l2(l1=l_1_norm, l2=l_2_norm),
               bias_regularizer=regularizers.l2(l_2_norm),
               activity_regularizer=regularizers.l2(l_2_norm))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv1D(filters=F3, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=None), kernel_regularizer=regularizers.l1_l2(l1=l_1_norm, l2=l_2_norm),
               bias_regularizer=regularizers.l2(l_2_norm),
               activity_regularizer=regularizers.l2(l_2_norm))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv1D(filters=F3, kernel_size=1, strides=s, padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=None),
                        kernel_regularizer=regularizers.l1_l2(l1=l_1_norm, l2=l_2_norm),
                        bias_regularizer=regularizers.l2(l_2_norm),
                        activity_regularizer=regularizers.l2(l_2_norm))(X_shortcut)
    X_shortcut = BatchNormalization(axis=-1, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


# def ResNet50(input_shape=(64, 64, 3), classes=6):
def ResNet50(classes=16):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    # np.random.seed(501)   # 1337
    input_shape = (LENGHT_TO_USE, 1)  # number os steps, number of features --> look at it this way,
    # if you have three axial acceleotor data, each for 200 seconds, then you had (200, 3),
    # now we have a UWB channel with 1000 samples but only 1 feature
    # Define the input as a tensor with shape input_shape
    X_input = Input(shape=input_shape)

    # Zero-Padding
    X = ZeroPadding1D(3)(X_input)

    # Stage 1
    X = Conv1D(64, kernel_size=7, strides=2, name='conv1', kernel_initializer=glorot_uniform(seed=None),
               kernel_regularizer=regularizers.l1_l2(l1=l_1_norm, l2=l_2_norm),
               bias_regularizer=regularizers.l2(l_2_norm),
               activity_regularizer=regularizers.l2(l_2_norm))(X)
    X = BatchNormalization(axis=-1, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(3, strides=2)(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling1D(2, name="avg_pool")(X)
    X = Flatten(name='flatten')(X)




####################################
    X_snr = Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l_1_norm, l2=l_2_norm),
                  kernel_initializer=glorot_uniform(seed=None),
                  bias_regularizer=regularizers.l2(l_2_norm),
                  activity_regularizer=regularizers.l2(l_2_norm))(X)
    X_snr = BatchNormalization(axis=-1, name='bn_dense_1')(X_snr)

    X_snr = Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l_1_norm, l2=l_2_norm),
                  kernel_initializer=glorot_uniform(seed=None),
                  bias_regularizer=regularizers.l2(l_2_norm),
                  activity_regularizer=regularizers.l2(l_2_norm))(X_snr)
    X_snr = BatchNormalization(axis=-1, name='bn_dense_2')(X_snr)

    x_snr = Dense(classes, activation='softmax', name='snr',
                  kernel_regularizer=regularizers.l1_l2(l1=l_1_norm, l2=l_2_norm),
                  kernel_initializer=glorot_uniform(seed=None),
                  bias_regularizer=regularizers.l2(l_2_norm),
                  activity_regularizer=regularizers.l2(l_2_norm))(X_snr)

    ####################################
    X_channel_type = Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l_1_norm, l2=l_2_norm),
                           kernel_initializer=glorot_uniform(seed=None),
                           bias_regularizer=regularizers.l2(l_2_norm),
                           activity_regularizer=regularizers.l2(l_2_norm))(X)
    X_channel_type = BatchNormalization(axis=-1, name='bn_dense_1_channel_type')(X_channel_type)

    X_channel_type = Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l_1_norm, l2=l_2_norm),
                           kernel_initializer=glorot_uniform(seed=None),
                           bias_regularizer=regularizers.l2(l_2_norm),
                           activity_regularizer=regularizers.l2(l_2_norm))(X_channel_type)

    X_channel_type = BatchNormalization(axis=-1, name='bn_dense_2_channel_type')(X_channel_type)

    x_channel_type = Dense(4, activation='softmax', name='channel_type',
                           kernel_regularizer=regularizers.l1_l2(l1=l_1_norm, l2=l_2_norm),
                           kernel_initializer=glorot_uniform(seed=None),
                           bias_regularizer=regularizers.l2(l_2_norm),
                           activity_regularizer=regularizers.l2(l_2_norm))(X_channel_type)

    ####################################
    X_toa = Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l_1_norm, l2=l_2_norm),
                  kernel_initializer=glorot_uniform(seed=None),
                  bias_regularizer=regularizers.l2(l_2_norm),
                  activity_regularizer=regularizers.l2(l_2_norm))(X)
    X_toa = BatchNormalization(axis=-1, name='bn_dense_1_toa')(X_toa)

    X_toa = Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l_1_norm, l2=l_2_norm),
                  kernel_initializer=glorot_uniform(seed=None),
                  bias_regularizer=regularizers.l2(l_2_norm),
                  activity_regularizer=regularizers.l2(l_2_norm))(X_toa)
    X_toa = BatchNormalization(axis=-1, name='bn_dense_2_toa')(X_toa)

    x_toa = Dense(1, activation='linear', name='toa',
                  kernel_regularizer=regularizers.l1_l2(l1=l_1_norm, l2=l_2_norm),
                  kernel_initializer=glorot_uniform(seed=None),
                  bias_regularizer=regularizers.l2(l_2_norm),
                  activity_regularizer=regularizers.l2(l_2_norm))(X_toa)

    ####################################
    model = Model(inputs=X_input, outputs=[x_snr, x_toa, x_channel_type])
    # model.summary()

    return model


def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.


    print('Hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee ')
    checkpoint_path = checkpoint_path_
    print('checkpoint_path ', checkpoint_path)

    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    checkpoints = [checkpoint_path + '/' + name
                   for name in os.listdir(checkpoint_path)]

    print('checkpoints ', checkpoints)
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print('Restoring from', latest_checkpoint)
        return load_model(latest_checkpoint)
    else:
        print('Creating a new model')
        return ResNet50()


def main():
    # model = model_fun()

    model = make_or_restore_model()

    # model = ResNet50()
    train_x, validation_x, train_snr_y, validation_snr_y, train_toa_y, validation_toa_y, \
        train_channel_type_y, validation_channel_type_y = reading_files()

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss={'snr': 'categorical_crossentropy', 'toa': 'mse',
                        'channel_type': 'categorical_crossentropy'},
                  loss_weights={'snr': 1 / 3, 'toa': 1 / 3, 'channel_type': 1 / 3},
                  metrics={'snr': ['categorical_accuracy', Precision(), AUC()], 'toa': ['mse', 'mae'],
                           'channel_type': ['categorical_accuracy', Precision(), AUC()]})


    # Binary crossentropy is for multi - label classifications,
    # whereas categorical cross entropy is for multi-class classification where each example belongs to a single class.
    # H = model.fit(train_x, [train_snr_y, train_toa_y], batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)

    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path_ + "/check.hdf5",
                                                monitor='val_loss', verbose=0,
                                                save_best_only=True, save_weights_only=False, mode='auto',
                                                save_freq="epoch")

    H = model.fit(train_x, [train_snr_y, train_toa_y, train_channel_type_y],
                  validation_data=[validation_x, [validation_snr_y, validation_toa_y, validation_channel_type_y]],
                  callbacks=[model_checkpoint_callback],
                  use_multiprocessing=True, workers=6, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=0)

    # H = model.fit(train_x, [train_snr_y, train_toa_y, train_channel_type_y],
    #               validation_data=[validation_x, [validation_snr_y, validation_toa_y, validation_channel_type_y]],
    #               use_multiprocessing=True, workers=6, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=2)
    # print('H.history.keys() ', H.history.keys())
    for key_ in H.history.keys():
        print(key_ + '=', [float('%.3f' % elem) for elem in H.history[key_]], end=" ")
    print()


if __name__ == '__main__':
    main()
