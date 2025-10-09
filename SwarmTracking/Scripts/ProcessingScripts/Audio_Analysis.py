#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Created: 2025-10-9
Author: Alberto Doimo
email: alberto.doimo@uni-konstanz.de

Description:

Audio analysis and save of events for video tracking.
Uses AudioProcessor.py to process, same as the robots

"""
#%%

import numpy as np
import sounddevice as sd
import soundfile as sf
import scipy.signal as signal
import pandas as pd
import os
from AudioProcessor import AudioProcessor
from Utils_SwarmTracking import *
from functions import *
from functions.das_v2 import das_filter

ips = [238, 240, 241]
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # project directory
input_dir = "Data/IntermediateData/"
date_dir = '2025-10-02/'
robot_rec_dir = '2025-10-02_18-45-28/'

file_name = '2025-09-24_20-04-23_upsampled'

trim_dir = os.path.join(project_dir, input_dir, date_dir, robot_rec_dir, "hp_filtered_trimmed_audio") # trimmed audio directory

# Get the index of the USB card
usb_fireface_index = None

# Define the variables for the audio processing

trigger_level =  70 # dB SPL
critical_level = 75 # dB SPL
c = 343   # speed of sound
fs = 48000

rec_samplerate = 48000
input_buffer_time = 0.04 # seconds
block_size = int(input_buffer_time*fs)  #used for the shared queue from which the doa is computed, not anymore for the output stream
channels = 5
mic_spacing = 0.018 #m
ref = channels//2 #central mic in odd array as ref
#print('ref=',ref) 
#ref= 0 #left most mic as reference

# Possible algorithms for computing DOA: CC, DAS
method = 'DAS'

# Parameters for the DAS algorithm
theta_das = np.linspace(-90, 90, 61) # angles resolution for DAS spectrum
N_peaks = 1 # Number of peaks to detect in DAS spectrum

# Parameters for the chirp signal
duration_out = 20e-3  # Duration in seconds
silence_post = 110 # [ms] can probably pushed to 20
amplitude = 0.5 # Amplitude of the chirp

t = np.linspace(0, duration_out, int(fs*duration_out))
start_f, end_f = 24e3, 2e3
sweep = signal.chirp(t, start_f, t[-1], end_f)
sweep *= signal.windows.tukey(sweep.size, 0.2)
sweep *= 0.8

silence_samples_post = int(silence_post * fs/1000)
silence_vec_post = np.zeros((silence_samples_post, ))
post_silence_sig = np.concatenate((sweep, silence_vec_post))
full_sig = post_silence_sig

stereo_sig = np.hstack([full_sig.reshape(-1, 1), full_sig.reshape(-1, 1)])
data = amplitude * np.float32(stereo_sig)

out_blocksize = int(len(data))  # Length of the output signal
print('out_blocksize =', out_blocksize)

# plot and save data 
# plt.figure(figsize=(10, 4))
# plt.plot(np.arange(len(full_sig)) / fs, data[:, 0], label='Left Channel')
# plt.plot(np.arange(len(full_sig)) / fs, data[:, 1], label='Right Channel')
# plt.title('Chirp Signal')
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.tight_layout()
# plt.savefig('chirp_signal.png')
# plt.close()

# Calculate highpass and lowpass frequencies based on the array geometry
auto_hipas_freq = int(343/(2*(mic_spacing*(channels-1))))
print('HP frequency:', auto_hipas_freq)
auto_lowpas_freq = int(343/(2*mic_spacing))
print('LP frequency:', auto_lowpas_freq)
highpass_freq, lowpass_freq = [auto_hipas_freq ,auto_lowpas_freq]
#highpass_freq, lowpass_freq = [200 ,20e3]
freq_range = [start_f, end_f]

cutoff = auto_hipas_freq # [Hz] highpass filter cutoff frequency
sos = signal.butter(1, cutoff, 'hp', fs=fs, output='sos')

# Set the path for the sensitivity CSV file relative to the current script location
script_dir = os.path.dirname(os.path.abspath(__file__))
sensitivity_path = os.path.join(project_dir, input_dir, 'Knowles_SPH0645LM4H-B_sensitivity.csv')
sensitivity = pd.read_csv(sensitivity_path)

analyzed_buffer_time = 0.01
block_size_analyzed_buffer = int(analyzed_buffer_time * fs)

in_sig = np.zeros(block_size_analyzed_buffer, dtype=np.float32)  # Initialize the buffer for the audio input stream
print('in_sig shape:', np.shape(in_sig))
centrefreqs, freqrms = calc_native_freqwise_rms(in_sig, fs)

freqs = np.array(sensitivity.iloc[:, 0])  # first column contains frequencies
sens_freqwise_rms = np.array(sensitivity.iloc[:, 1])  # Last column contains sensitivity values 
interp_sensitivity = interpolate_freq_response([freqs, sens_freqwise_rms],
                centrefreqs)

frequency_band = [2e3, 20e3] # min, max frequency to do the compensation Hz
tgtmic_relevant_freqs = np.logical_and(centrefreqs>=frequency_band[0],
                            centrefreqs<=frequency_band[1])

# args = {}
# args.buffer = np.zeros((block_size, channels))

# # Set initial parameters for the audio processing
# args.samplerate = fs
# args.rec_samplerate = rec_samplerate
# args.angle = 0
# # Create instances of the AudioProcessor and RobotMove classes
# audio_processor = AudioProcessor(fs, channels, block_size, analyzed_buffer_time, data, args, trigger_level, critical_level, mic_spacing, ref, highpass_freq, lowpass_freq, theta_das, N_peaks,
#                                     usb_fireface_index, args.subtype, interp_sensitivity, tgtmic_relevant_freqs, args.filename, args.rec_samplerate, sos, sweep)
# # robot_move = RobotMove(speed, turn_speed, left_sensor_threshold, right_sensor_threshold, critical_level, trigger_level, ground_sensors_bool = True)
# doa_calc = AudioProcessor.update_das()

# inputstream_thread = audio_processor.input_stream
# inputstream_thread.start()

print(f"\n !! Verify parameter matching between this signal and the one used in the experiment !! \n")
print(f" Current signal characteristics: \n Fs: {fs},\n Range: {start_f} - {end_f} Hz,\n Amplitude: {amplitude},\n Duration: {duration_out} [s],\n Silence after call: {silence_post} [ms]\n")
# user_input = input("Do you want to continue? (y/n): ").strip().lower()
# if user_input != 'y':
#     print("Operation cancelled by user.")
#     exit(0)
#%%

# Match filtering and peak detection on channel 3 of each IP
for ip in ips:
    ip_folder = os.path.join(trim_dir, f"ip_{ip}")
    # Define the time for trimming 
    for fname in os.listdir(ip_folder):
        if fname.lower().startswith('mic_') and fname.lower().endswith('.wav'):
            mic_num = fname.split('_')[1].split('.')[0]  # Extract mic number
            fpath = os.path.join(ip_folder, fname)
            audio_data, fs_file = sf.read(fpath)
            peak_times = pd.read_csv(os.path.join(project_dir, input_dir, date_dir, robot_rec_dir, f"ip_{ip}_call_times.csv"), header=None).values.flatten()
            print(f"IP: {ip} Mic: {mic_num}")
            # Run over peak times
            for i in range(len(peak_times) - 1):
                peak_time = peak_times[i]
                next_peak_time = peak_times[i + 1]
                
                intercall_audio = audio_data[int(peak_time*fs_file):int(next_peak_time*fs_file)]
                # print(f"IP: {ip} Mic: {mic_num} Processing peak at {peak_time:.3f}s, next peak at {next_peak_time:.3f}s, buffer length: {len(intercall_audio)/fs_file:.3f} secs")

                # buffer = intercall_audio[int(0.09*fs_file):]

                # update_das(buffer, fs, sos, ref, analyzed_buffer_time, tgtmic_relevant_freqs, interp_sensitivity, das_filter, channels, mic_spacing, highpass_freq, lowpass_freq, theta_das, critical_level, trigger_level, N_peaks)


# %%
