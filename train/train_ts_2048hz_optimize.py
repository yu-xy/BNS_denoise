import gc
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import keras.backend as K
import keras.models
import numpy as np
from scipy import signal
from keras.layers import Dense, Conv1D, Conv2D, ELU, Flatten, Dropout, BatchNormalization, AveragePooling1D,MaxPooling1D, MaxPooling2D, Activation, Add, UpSampling1D, UpSampling2D, concatenate, Reshape
from keras.models import Model
from keras.layers import Input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizer_v2.adam import Adam
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from matplotlib import pyplot as plt
import matplotlib
import tensorflow as tf
import yaml
import sys
sys.path.append('../tools')
from transformer_seunet_ts2 import EncoderModel_6
from model import unet3, seunet, unet_notse
from load_data_yxy import get_ds_slicetrain_batch_iter, data_sample
from keras.callbacks import Callback



# matplotlib.use('TkAgg')

class data_sample():
    def __init__(self, noiseE1, noiseE2, noiseE3, mass1_list, mass2_list, spin1z_list, spin2z_list, right_ascension_list,
                 declination_list, signal_E1_list, signal_E2_list, signal_E3_list, snr_E1_list, snr_E2_list, snr_E3_list):
        self.noiseE1 = noiseE1
        self.noiseE2 = noiseE2
        self.noiseE3 = noiseE3
        self.mass1_list = mass1_list
        self.mass2_list = mass2_list
        self.spin1z_list = spin1z_list
        self.spin2z_list = spin2z_list
        self.right_ascension_list = right_ascension_list
        self.declination_list = declination_list
        self.signal_E1_list = signal_E1_list
        self.signal_E2_list = signal_E2_list
        self.signal_E3_list = signal_E3_list
        self.snr_E1_list = snr_E1_list
        self.snr_E2_list = snr_E2_list
        self.snr_E3_list = snr_E3_list
    def print(self):
        print('mass1='+str(mass1_list))
        print('mass2=' + str(mass2_list))
        print('spin1z='+str(spin1z_list))
        print('spin2z='+str(spin2z_list))
        print('right_ascension='+str(right_ascension_list))
        print('declination='+str(declination_list))
    def help(self):
        print('noiseE1, noiseE2, noiseE3 are 128 s length noise of E1, E2 and E3')
        print('signal_E1_list, signal_E2_list and signal_E3_list are signals, and each have 1 samples')
        print('mass1_list, mass2_list are masses, and each have 1 masses')
        print('right_ascension_list and declination_list are the directions of the source of the signal')

class LearningRatePrinter(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        print(f"Learning rate at epoch {epoch + 1}: {K.get_value(lr)}")

def model_denoise_iter(model, batch_iter):
    for x, y in batch_iter:
        pre = model.predict(x)
        yield (pre,y)

def normal_train():
    model = unet3()
    # model = keras.models.load_model('../model/train_2048_ts_optimize_snr20-40_20s_0.0120328_38.h5')
    model.summary()
    model_denoise = keras.models.load_model('../model/train_2048_new_bottomtrans_reseunet_snr20-40_20s_0.0264762_24+18.h5', \
                                        custom_objects={"EncoderModel_6": EncoderModel_6})
    # optimizer = Adam(lr=1e-4)
    # model.compile(optimizer=optimizer, loss='mean_squared_error')
    # K.set_value(model.optimizer.lr,1e-4)
    train_datapath = '../../../../teledata/8192-bns-100s-256/train_data'
    valid_datapath = '../../../../teledata/8192-bns-100s-256/valid_data'
    batch_size = 32
    sample_length = 20
    snr_list=[20,15,12,10,12,15]
    peak_range=(0.6,0.9)
    sample_distance=[1,4]
    freq=8192
    snr_change_time=2000
    wn=[2 * 1024 / freq]
    train_iter = get_ds_slicetrain_batch_iter(batch_size=batch_size,sample_length=sample_length,snr_list=snr_list,\
                                              sample_distance=sample_distance,freq=freq, snr_change_time=snr_change_time,\
                                              datapath=train_datapath,wn=wn)
    valid_iter = get_ds_slicetrain_batch_iter(batch_size=batch_size,sample_length=sample_length,snr_list=snr_list,\
                                             sample_distance=sample_distance,freq=freq, snr_change_time=snr_change_time,\
                                             datapath=valid_datapath, wn=wn)
    # check_point = ModelCheckpoint('train_2048_ts_bottomtrans_reseunet_snr20-40_5s_{val_loss:.7f}_{epoch:02d}epoch.h5', monitor='val_loss',\
    #                               verbose=1, save_best_only=True, mode='min')
    check_point = ModelCheckpoint(filepath='train_2048_ts_optimize_snr20-40_20s_{val_loss:.7f}_{epoch:02d}+38.h5')
    Reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', epsilon=0.000001)
    
    lr_printer = LearningRatePrinter()
    
    train_opt_iter = model_denoise_iter(model_denoise, train_iter)
    valid_opt_iter = model_denoise_iter(model_denoise, valid_iter)
    
    history = model.fit_generator(
        generator=train_opt_iter,
        steps_per_epoch=240000//batch_size,
        epochs=100,
        verbose=1,
        # validation_data=get_train_batch_origin(data_path=val_path, batch_size=batch_size,config_path=val_config_file),
        validation_data=valid_opt_iter,
        validation_steps=30000//batch_size,
        callbacks=[check_point, Reduce, lr_printer],
        workers=1
    )
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def overlap(a, b):
    return np.sum(a*b, axis=0)/np.sqrt(np.sum(a*a, axis=0))/np.sqrt(np.sum(b*b, axis=0))

if __name__ == '__main__':

    normal_train()


