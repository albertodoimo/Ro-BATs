#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Created: 2025-10-6
Author: Alberto Doimo
email: alberto.doimo@uni-konstanz.de

Description:

Robat Calls identification:
This scripts identify the emission calls (chirps) from the audio recordings of the robots.
Save as csv file for each ip

"""

#%%
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import soundfile as sf
import os 
from Utils_SwarmTracking import *
import pandas as pd 

# %%

# Define the directories
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # project directory
input_dir = "./Data"
output_dir = "./Data/IntermediateData"
os.makedirs(output_dir, exist_ok=True)

# Chirp signal template
fs_call_sig = 48000 # Sampling frequency
duration_out = 20e-3  # Duration in seconds
silence_post = 110 # [ms] can probably pushed to 20
amplitude = 0.5 # Amplitude of the chirp

t = np.linspace(0, duration_out, int(fs_call_sig*duration_out))
start_f, end_f = 24e3, 2e3
sweep = signal.chirp(t, start_f, t[-1], end_f)
sweep *= signal.windows.tukey(sweep.size, 0.2)
sweep *= 0.8

silence_samples_post = int(silence_post * fs_call_sig/1000)
silence_vec_post = np.zeros((silence_samples_post, ))
post_silence_sig = np.concatenate((sweep, silence_vec_post))
full_sig = post_silence_sig

stereo_sig = np.hstack([full_sig.reshape(-1, 1), full_sig.reshape(-1, 1)])
call_sig = amplitude * full_sig

print(f"\n !! Verify parameter matching between this signal and the one used in the experiment !! \n")
print(f" Current signal characteristics: \n Fs: {fs_call_sig},\n Range: {start_f} - {end_f} Hz,\n Amplitude: {amplitude},\n Duration: {duration_out} [s],\n Silence after call: {silence_post} [ms]\n")
# user_input = input("Do you want to continue? (y/n): ").strip().lower()
# if user_input != 'y':
#     print("Operation cancelled by user.")
#     exit(0)

# Plot the call_sig signal
# Time-domain plot
plt.figure(figsize=(15, 8))
plt.subplot( 2, 1, 1)
plt.plot(np.linspace(0,len(call_sig)/fs_call_sig, len(call_sig)), call_sig)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude', fontsize=12)
plt.grid()
plt.tight_layout()
# Spectrogram plot
plt.subplot( 2, 1, 2)
plt.specgram(call_sig, Fs=fs_call_sig, NFFT=64, noverlap=32, cmap='viridis')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.yticks(np.arange(0, fs_call_sig/2+1, 1000))
plt.suptitle('Call signal', fontsize=20)
plt.tight_layout()
plt.show()

#%%

# Load the recordings
ips = [238, 240, 241]
date_dir = '2025-10-02/'
robot_rec_dir = '2025-10-02_18-45-28/'

trim_dir = os.path.join(output_dir, date_dir, robot_rec_dir, "hp_filtered_trimmed_audio") # trimmed audio directory

# Match filtering and peak detection on channel 3 of each IP
for ip in ips:
    ip_folder = os.path.join(trim_dir, f"ip_{ip}")
    for fname in os.listdir(ip_folder):
        if fname.lower() == 'mic_3.wav':

            fpath = os.path.join(ip_folder, fname)
            audio_data, fs_file = sf.read(fpath)

            # Match filtering 
            matched = matched_filter(audio_data, call_sig)

            # Normalize the matched filter output
            matched = matched / np.max(np.abs(matched))
            peaks = detect_peaks(matched, fs_file)
            peak_times = (peaks / fs_file)
            print(f"Detected peaks: {len(peaks)} in IP {ip} at {peak_times}, file {fname}")
            # Save peaks to CSV
            pd.DataFrame(peak_times).to_csv(os.path.join(output_dir, date_dir, robot_rec_dir, f"ip_{ip}_call_times.csv"), index=False, header=False)

            # plot results
            plt.figure(figsize=(15, 8))

            ax1 = plt.subplot(2, 1, 1)
            ax2 = plt.subplot(2, 1, 2, sharex=ax1)

            ax1.plot(np.linspace(0, len(matched) / fs_file, len(matched)), matched)
            ax1.plot(peaks / fs_file, matched[peaks], 'ro')
            ax1.set_title('Matched Filter Output')
            ax1.set_xlabel('Time [s]')
            ax1.set_ylabel('Amplitude')

            ax2.plot(np.linspace(0, len(audio_data) / fs_file, len(audio_data)), audio_data)
            ax2.plot(peaks / fs_file, audio_data[peaks], 'ro')
            ax2.set_title('Original Audio with Detected Peaks')
            ax2.set_xlabel('Time [s]')
            ax2.set_ylabel('Amplitude')

            plt.setp(ax1.get_xticklabels(), visible=False)  # Hide x labels on the first plot

            plt.tight_layout()
            plt.show()

# %%


csv = pd.read_csv(os.path.join(output_dir, date_dir, robot_rec_dir, "ip_238_call_times.csv"), header=None)
print(csv.shape)
csv.T
 
pd.DataFrame.to_csv(csv.T, 'test.csv', index=False, header=False)

# %%