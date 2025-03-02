import gc
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import numpy as np
from scipy import signal as sp_signal

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


def file_iter(directory):
    file_list = os.listdir(directory)
    while True:
        for dir_item in file_list:
            if dir_item.endswith('.pkl'):
                yield os.path.join(directory,dir_item)

def sample_iter(datapath):
    data = [1,2,3]
    data_file_iter = file_iter(datapath)
    print(data_file_iter)
    for file in data_file_iter:
        # print(file)
        with open(file,'rb') as f:
            del data
            gc.collect()
            data = pickle.load(f)
        for sample in data:
            yield sample

# E1 Signal reading
def denoising_sample_cut_iter_forE1(sample_length,snr_list, peak_range,freq,snr_change_time,datapath):
    data_iter_sample = sample_iter(datapath)
    noise_rand_begin = 8192
    noise_rand_end = int(255*8192-sample_length*freq)
    gen_num = 0
    obj_snr_index = 0
    for data_sample_ in data_iter_sample:
        for signal, snr_s in zip(data_sample_.signal_E1_list,data_sample_.snr_E1_list):
            peak = random.uniform(peak_range[0],peak_range[1])

            # change snr range
            if gen_num > 0 and gen_num % snr_change_time == 0:
                obj_snr_index = (obj_snr_index + 1) % len(snr_list)
                # print('next snr is:', str(snr_list[obj_snr_index]))
            snr = random.uniform(snr_list[obj_snr_index],snr_list[int((obj_snr_index+1) % len(snr_list))])
            snr_norm = snr/snr_s

            peak_num = int(peak*sample_length*freq)
            signal_peak = np.argmax(signal)
            all_num = sample_length*freq
            # print(all_num)
            signal_after_peak = np.size(signal)-signal_peak
            after_peak = all_num-peak_num
            return_data = np.zeros(all_num)
            if(peak_num<signal_peak):
                return_data[:peak_num]=signal[signal_peak-peak_num:signal_peak]*snr_norm
            else:
                return_data[peak_num-signal_peak:peak_num]=signal[0:signal_peak]*snr_norm
            if(signal_after_peak<after_peak):
                return_data[peak_num:peak_num+signal_after_peak]=signal[signal_peak:signal_peak+signal_after_peak]*snr_norm
            else:
                return_data[peak_num:]=signal[signal_peak:signal_peak+after_peak]*snr_norm

            noise_begin = random.randint(noise_rand_begin,noise_rand_end)
            noise = data_sample_.noiseE1[noise_begin:noise_begin+all_num]
            gen_num = gen_num + 1
            yield return_data+noise, return_data

# E1-3 Signal reading
def denoising_sample_cut_iter_forE1_E3(sample_length,snr_list, peak_range,freq,snr_change_time,datapath):
    data_iter_sample = sample_iter(datapath)
    noise_rand_begin = 8192
    noise_rand_end = int(255*8192-sample_length*freq)
    gen_num = 0
    obj_snr_index = 0
    for data_sample_ in data_iter_sample:
        for signal, snr_s in zip(data_sample_.signal_E1_list,data_sample_.snr_E1_list):
            peak = random.uniform(peak_range[0],peak_range[1])

            # change snr range
            if gen_num > 0 and gen_num % snr_change_time == 0:
                obj_snr_index = (obj_snr_index + 1) % len(snr_list)
                # print('next snr is:', str(snr_list[obj_snr_index]))
            snr = random.uniform(snr_list[obj_snr_index],snr_list[int((obj_snr_index+1) % len(snr_list))])
            snr_norm = snr/snr_s

            peak_num = int(peak*sample_length*freq)
            signal_peak = np.argmax(signal)
            all_num = sample_length*freq
            # print(all_num)
            signal_after_peak = np.size(signal)-signal_peak
            after_peak = all_num-peak_num
            return_data = np.zeros(all_num)
            if(peak_num<signal_peak):
                return_data[:peak_num]=signal[signal_peak-peak_num:signal_peak]*snr_norm
            else:
                return_data[peak_num-signal_peak:peak_num]=signal[0:signal_peak]*snr_norm
            if(signal_after_peak<after_peak):
                return_data[peak_num:peak_num+signal_after_peak]=signal[signal_peak:signal_peak+signal_after_peak]*snr_norm
            else:
                return_data[peak_num:]=signal[signal_peak:signal_peak+after_peak]*snr_norm

            noise_begin = random.randint(noise_rand_begin,noise_rand_end)
            noise = data_sample_.noiseE1[noise_begin:noise_begin+all_num]
            gen_num = gen_num + 1
            yield return_data+noise, return_data
        for signal, snr_s in zip(data_sample_.signal_E2_list,data_sample_.snr_E2_list):
            peak = random.uniform(peak_range[0],peak_range[1])

            # change snr range
            if gen_num > 0 and gen_num % snr_change_time == 0:
                obj_snr_index = (obj_snr_index + 1) % len(snr_list)
                # print('next snr is:', str(snr_list[obj_snr_index]))
            snr = random.uniform(snr_list[obj_snr_index],snr_list[int((obj_snr_index+1) % len(snr_list))])
            snr_norm = snr/snr_s

            peak_num = int(peak*sample_length*freq)
            signal_peak = np.argmax(signal)
            all_num = sample_length*freq
            # print(all_num)
            signal_after_peak = np.size(signal)-signal_peak
            after_peak = all_num-peak_num
            return_data = np.zeros(all_num)
            if(peak_num<signal_peak):
                return_data[:peak_num]=signal[signal_peak-peak_num:signal_peak]*snr_norm
            else:
                return_data[peak_num-signal_peak:peak_num]=signal[0:signal_peak]*snr_norm
            if(signal_after_peak<after_peak):
                return_data[peak_num:peak_num+signal_after_peak]=signal[signal_peak:signal_peak+signal_after_peak]*snr_norm
            else:
                return_data[peak_num:]=signal[signal_peak:signal_peak+after_peak]*snr_norm

            noise_begin = random.randint(noise_rand_begin,noise_rand_end)
            noise = data_sample_.noiseE2[noise_begin:noise_begin+all_num]
            gen_num = gen_num + 1
            yield return_data+noise, return_data
        for signal, snr_s in zip(data_sample_.signal_E3_list,data_sample_.snr_E3_list):
            peak = random.uniform(peak_range[0],peak_range[1])

            # change snr range
            if gen_num > 0 and gen_num % snr_change_time == 0:
                obj_snr_index = (obj_snr_index + 1) % len(snr_list)
                # print('next snr is:', str(snr_list[obj_snr_index]))
            snr = random.uniform(snr_list[obj_snr_index],snr_list[int((obj_snr_index+1) % len(snr_list))])
            snr_norm = snr/snr_s

            peak_num = int(peak*sample_length*freq)
            signal_peak = np.argmax(signal)
            all_num = sample_length*freq
            # print(all_num)
            signal_after_peak = np.size(signal)-signal_peak
            after_peak = all_num-peak_num
            return_data = np.zeros(all_num)
            if(peak_num<signal_peak):
                return_data[:peak_num]=signal[signal_peak-peak_num:signal_peak]*snr_norm
            else:
                return_data[peak_num-signal_peak:peak_num]=signal[0:signal_peak]*snr_norm
            if(signal_after_peak<after_peak):
                return_data[peak_num:peak_num+signal_after_peak]=signal[signal_peak:signal_peak+signal_after_peak]*snr_norm
            else:
                return_data[peak_num:]=signal[signal_peak:signal_peak+after_peak]*snr_norm

            noise_begin = random.randint(noise_rand_begin,noise_rand_end)
            noise = data_sample_.noiseE3[noise_begin:noise_begin+all_num]
            gen_num = gen_num + 1
            yield return_data+noise, return_data


            
# Random peak timing signal packaging
def get_train_batch_iter(batch_size,sample_length,snr_list, peak_range,freq, snr_change_time, datapath):
    my_denoising_iter = denoising_sample_cut_iter_forE1_E3(sample_length,snr_list, peak_range,freq,snr_change_time,datapath)
    count = 1
    batch_x = []
    batch_y = []
    for strain, signal in my_denoising_iter:
        # print(strain.shape)
        if count == 1:
            batch_x = strain / np.max(strain)
            batch_y = signal / np.max(signal)
        else:
            batch_x = np.concatenate((batch_x, strain / np.max(strain)))
            batch_y = np.concatenate((batch_y, signal / np.max(signal)))
        if count == batch_size:
            yield (batch_x.reshape(-1, sample_length*freq, 1), batch_y.reshape(-1, sample_length *freq, 1))
        count = count + 1
        if count > batch_size:
            count = 1
            

# Reading of E1-E3 slice signals
# sample_distance:The distance between the tail of the sliced signal and the peak of the original signal
def cut_slice_iter_forE1_E3(sample_length, snr_list, sample_distance, freq, snr_change_time, datapath):
    data_iter_sample = sample_iter(datapath)
    noise_rand_begin = 8192
    noise_rand_end = int(255*8192-sample_length*freq)
    gen_num = 0
    obj_snr_index = 0
    for data_sample_ in data_iter_sample:
        for signal, snr_s in zip(data_sample_.signal_E1_list,data_sample_.snr_E1_list):
            peak = 1

            # change snr range
            if gen_num > 0 and gen_num % snr_change_time == 0:
                obj_snr_index = (obj_snr_index + 1) % len(snr_list)
                # print('next snr is:', str(snr_list[obj_snr_index]))
            snr = random.uniform(snr_list[obj_snr_index],snr_list[int((obj_snr_index+1) % len(snr_list))])
            snr_norm = snr/snr_s
            data_end = random.randint(sample_distance[0]*freq, sample_distance[1]*freq)
            signal_peak = np.argmax(signal)
            sample_num = sample_length*freq
            return_data = np.zeros(sample_num)
            if (signal_peak-data_end-sample_num) >= 0:
                return_data[:] = signal[signal_peak-data_end-sample_num+1:signal_peak-data_end+1]*snr_norm
            else:
                return_data[sample_num-1-signal_peak+data_end:] = signal[0:signal_peak-data_end+1]*snr_norm

            noise_begin = random.randint(noise_rand_begin,noise_rand_end)
            noise = data_sample_.noiseE1[noise_begin:noise_begin+sample_num]
            gen_num = gen_num + 1
            yield return_data+noise, return_data
        for signal, snr_s in zip(data_sample_.signal_E2_list,data_sample_.snr_E2_list):
            peak = 1

            # change snr range
            if gen_num > 0 and gen_num % snr_change_time == 0:
                obj_snr_index = (obj_snr_index + 1) % len(snr_list)
                # print('next snr is:', str(snr_list[obj_snr_index]))
            snr = random.uniform(snr_list[obj_snr_index],snr_list[int((obj_snr_index+1) % len(snr_list))])
            snr_norm = snr/snr_s
            data_end = random.randint(sample_distance[0]*freq, sample_distance[1]*freq)
            signal_peak = np.argmax(signal)
            sample_num = sample_length*freq
            return_data = np.zeros(sample_num)
            if (signal_peak-data_end-sample_num) >= 0:
                return_data[:] = signal[signal_peak-data_end-sample_num+1:signal_peak-data_end+1]*snr_norm
            else:
                return_data[sample_num-1-signal_peak+data_end:] = signal[0:signal_peak-data_end+1]*snr_norm

            noise_begin = random.randint(noise_rand_begin,noise_rand_end)
            noise = data_sample_.noiseE2[noise_begin:noise_begin+sample_num]
            gen_num = gen_num + 1
            yield return_data+noise, return_data
        for signal, snr_s in zip(data_sample_.signal_E3_list,data_sample_.snr_E3_list):
            peak = 1

            # change snr range
            if gen_num > 0 and gen_num % snr_change_time == 0:
                obj_snr_index = (obj_snr_index + 1) % len(snr_list)
                # print('next snr is:', str(snr_list[obj_snr_index]))
            snr = random.uniform(snr_list[obj_snr_index],snr_list[int((obj_snr_index+1) % len(snr_list))])
            snr_norm = snr/snr_s
            data_end = random.randint(sample_distance[0]*freq, sample_distance[1]*freq)
            signal_peak = np.argmax(signal)
            sample_num = sample_length*freq
            return_data = np.zeros(sample_num)
            if (signal_peak-data_end-sample_num) >= 0:
                return_data[:] = signal[signal_peak-data_end-sample_num+1:signal_peak-data_end+1]*snr_norm
            else:
                return_data[sample_num-1-signal_peak+data_end:] = signal[0:signal_peak-data_end+1]*snr_norm

            noise_begin = random.randint(noise_rand_begin,noise_rand_end)
            noise = data_sample_.noiseE3[noise_begin:noise_begin+sample_num]
            gen_num = gen_num + 1
            yield return_data+noise, return_data

# Packaging of sliced timing signals
def get_slicetrain_batch_iter(batch_size,sample_length,snr_list, sample_distance,freq, snr_change_time, datapath):
    my_denoising_iter = cut_slice_iter_forE1_E3(sample_length,snr_list, sample_distance,freq,snr_change_time,datapath)
    count = 1
    batch_x = []
    batch_y = []
    for strain, signal in my_denoising_iter:
        # print(strain.shape)
        if count == 1:
            batch_x = strain / np.max(strain)
            batch_y = signal / np.max(signal)
        else:
            batch_x = np.concatenate((batch_x, strain / np.max(strain)))
            batch_y = np.concatenate((batch_y, signal / np.max(signal)))
        if count == batch_size:
            yield (batch_x.reshape(-1, sample_length*freq, 1), batch_y.reshape(-1, sample_length *freq, 1))
        count = count + 1
        if count > batch_size:
            count = 1
            
# Low pass filter
def low_filt(sig, N=8, wn=[2 * 256 / 8192], type = 'lowpass' ):
    # N = 8 # filter order
    # wn=[2 * 256 / 8192] # Normalized cut-off frequency
    # type = 'lowpass' # filter type
    b, a = sp_signal.butter(N=N, Wn=wn, btype=type, analog=False, output='ba')
    filtedData = sp_signal.filtfilt(b, a, sig)
    return filtedData
            
# Downsampling and packaging of sliced temporal signals
def get_ds_slicetrain_batch_iter(batch_size,sample_length,wn,snr_list, sample_distance,freq, snr_change_time, datapath):
    my_denoising_iter = cut_slice_iter_forE1_E3(sample_length,snr_list, sample_distance,freq,snr_change_time,datapath)
    count = 1
    batch_x = []
    batch_y = []
    for strain, signal in my_denoising_iter:
        strain = low_filt(strain, wn=wn)
        strain = strain[::int(sample_length/5)]
        signal = low_filt(signal, wn=wn)
        signal = signal[::int(sample_length/5)]
        if count == 1:
            batch_x = strain / np.max(strain)
            batch_y = signal / np.max(signal)
        else:
            batch_x = np.concatenate((batch_x, strain / np.max(strain)))
            batch_y = np.concatenate((batch_y, signal / np.max(signal)))
        if count == batch_size:
            yield (batch_x.reshape(-1, 5*freq, 1), batch_y.reshape(-1, 5*freq, 1))
        count = count + 1
        if count > batch_size:
            count = 1
            
# The sliced timing signal is downsampled and packaged, but not filtered
def get_nfilt_ds_slicetrain_batch_iter(batch_size,sample_length,snr_list, sample_distance,freq, snr_change_time, datapath):
    my_denoising_iter = cut_slice_iter_forE1_E3(sample_length,snr_list, sample_distance,freq,snr_change_time,datapath)
    count = 1
    batch_x = []
    batch_y = []
    for strain, signal in my_denoising_iter:
        strain = strain[::int(sample_length/5)]
        signal = signal[::int(sample_length/5)]
        if count == 1:
            batch_x = strain / np.max(strain)
            batch_y = signal / np.max(signal)
        else:
            batch_x = np.concatenate((batch_x, strain / np.max(strain)))
            batch_y = np.concatenate((batch_y, signal / np.max(signal)))
        if count == batch_size:
            yield (batch_x.reshape(-1, 5*freq, 1), batch_y.reshape(-1, 5*freq, 1))
        count = count + 1
        if count > batch_size:
            count = 1
        