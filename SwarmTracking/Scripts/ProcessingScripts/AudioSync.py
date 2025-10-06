#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Created: 2025-10-1
Author: Alberto Doimo
email: alberto.doimo@uni-konstanz.de

Description:

This scripts sync audio files from sync_sig template.

Notes:

chirp was created at 48khz and reproduced probably at 192khz from rme 802, 
so this is why i see a ~100 ms shift between first and second sweep 
in the match filtering 

the first rising edge is on the second period of the square wave,
there's a possible delay of 1/15 fps (~66 ms) between the camera and the audio recording.

"""

#%%
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import soundfile as sf
import os 
from Utils_SwarmTracking import *

#%%
 
# Define the directories
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # project directory
input_dir = "./Data"
output_dir = "./Data/IntermediateData"
os.makedirs(output_dir, exist_ok=True)

# Upload the sync_sig signal
sync_sig, fs_sync_sig = sf.read(output_dir + '/filtered_sweep.wav')

# Plot the sync_sig signal
# Time-domain plot
plt.subplot( 2, 1, 1)
plt.plot(np.linspace(0,len(sync_sig)/fs_sync_sig, len(sync_sig)), sync_sig)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude', fontsize=12)
plt.grid()
# Spectrogram plot
plt.subplot( 2, 1,2)
plt.specgram(sync_sig, Fs=fs_sync_sig, NFFT=256, noverlap=64, cmap='viridis')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.yticks(np.arange(0, fs_sync_sig/2+1, 1000))
plt.suptitle('Filtered synchronisation sync_sig signal', fontsize=20)
plt.tight_layout()
plt.show()

# %%

# Load the recordings
ips = [238, 240, 241]
date_dir = '2025-10-02/'
robot_rec_dir = '2025-10-02_18-45-28/'
list_of_files = os.listdir(input_dir + '/InputData/' + date_dir + robot_rec_dir)
wav_files = [f for f in list_of_files if f.lower().endswith('.wav')]
print(f"Found {len(wav_files)} .wav files in the directory.")

# For each IP, find the corresponding file, load it, and save each channel in a separate subfolder
for ip in ips:
    # Find the file for this IP
    file_ip = next((f for f in wav_files if str(ip) in f), None)
    if file_ip is None:
        print(f"No file found for IP {ip}")
        continue

    print(f"Selected file for IP {ip}: {file_ip}")

    # Load the audio file
    orig_audio, fs = sf.read(os.path.join(input_dir + '/InputData/' + date_dir + robot_rec_dir, file_ip))
    if orig_audio.ndim == 1:
        orig_audio = orig_audio[:, np.newaxis]
    num_channels = orig_audio.shape[1]
    channels = [orig_audio[:, i] for i in range(num_channels)]

    # Prepare output subfolder for this IP
    ip_folder = os.path.join(output_dir, date_dir, robot_rec_dir, "multich_audio_separation", f"ip_{ip}")
    os.makedirs(ip_folder, exist_ok=True)

    # Save each channel as a separate WAV file
    for i, channel_data in enumerate(channels):
        output_file = os.path.join(ip_folder, f"mic_{i + 1}.wav")
        sf.write(output_file, channel_data, fs)
        print(f"Saved channel {i + 1} for IP {ip} to {output_file}")

# %%
# Apply filter to each separated channel file

# Design the bandpass filter
low_cutoff = 80  # low cutoff frequency in Hz
high_cutoff = 2000  # high cutoff frequency in Hz
order = 16  # filter order
sos = signal.butter(order, [low_cutoff, high_cutoff], 'bandpass', fs=fs, output='sos', analog=False)
w, h = signal.sosfreqz(sos, worN=2000, fs=fs)

# Design the highpass filter
sos2 = signal.butter(order, 4000, 'highpass', fs=fs, output='sos', analog=False)
w2, h2 = signal.sosfreqz(sos2, worN=2000, fs=fs)

# Compute frequency response for plotting
plt.figure(figsize=(15, 6))
plt.plot(w, 20 * np.log10(abs(h)), label='Bandpass filter')
plt.plot(w2, 20 * np.log10(abs(h2)), label='Highpass filter')
plt.title('Butterworth filter frequency response ')
plt.xlabel('Frequency [Hz]')
plt.xticks(np.linspace(0, 5000, 26), rotation=45)
plt.xlim(0, 5000)
plt.ylabel('Amplitude [dB]')
plt.ylim(-100, 5)
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(low_cutoff, color='red') # low cutoff frequency
plt.axvline(high_cutoff, color='red') # high cutoff frequency
plt.legend(loc='best', frameon=False)
plt.show()

# Directory containing separated channel files
for ip in ips:
    ip_folder = os.path.join(output_dir, date_dir, robot_rec_dir, "multich_audio_separation", f"ip_{ip}")
    ip_folder_lp = os.path.join(output_dir, date_dir, robot_rec_dir, "lp_filtered_audio", f"ip_{ip}")
    os.makedirs(ip_folder_lp, exist_ok=True)
    for fname in os.listdir(ip_folder):
        if fname.lower().endswith('.wav'):
            fpath = os.path.join(ip_folder, fname)
            data, fs = sf.read(fpath)
            filtered = signal.sosfiltfilt(sos, data)
            sf.write(os.path.join(ip_folder_lp, fname), filtered, fs)
            print(f"Applied BP filter to {os.path.join(ip_folder_lp, fname)}")


# %%

# Match filtering and peak detection on channel 3 of each IP
for ip in ips:
    ip_folder = os.path.join(output_dir, date_dir, robot_rec_dir, "multich_audio_separation", f"ip_{ip}")
    os.makedirs(ip_folder, exist_ok=True)
    for fname in os.listdir(ip_folder):
        if fname.lower() == 'mic_3.wav':
            fpath = os.path.join(ip_folder, fname)
            audio_data, fs_file = sf.read(fpath)

            # Match filtering channel 3 
            matched = matched_filter(audio_data, sync_sig)

            # Normalize the matched filter output
            matched = matched / np.max(np.abs(matched))
            peaks = detect_peaks(matched, fs_file)
            print(f"Detected peaks: {len(peaks)} in IP {ip} at {peaks/ fs_file}, file {fname}")

            # Save peaks to CSV
            # Extract times from peak indices
            peak_times = peaks / fs_file
            # Save all peak times for this IP to CSV (append mode)
            csv_path = os.path.join(output_dir, date_dir, robot_rec_dir)
            save_data_to_csv([ip, peak_times], f"ip_{ip}_sync_times.csv", csv_path)

            # plot results
            plt.figure(figsize=(18, 8))

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
            ax2.grid()

            plt.setp(ax1.get_xticklabels(), visible=False)  # Hide x labels on the first plot

            plt.tight_layout()
            plt.show()

    # Trim audio files at the first detected peak

    trim_dir = os.path.join(output_dir, date_dir, robot_rec_dir, "trimmed_audio") # create trimmed audio directory
    os.makedirs(trim_dir, exist_ok=True)
    ip_folder_trim = os.path.join(trim_dir, f"ip_{ip}") # create subfolder for each IP
    os.makedirs(ip_folder_trim, exist_ok=True)
    for f_name in os.listdir(ip_folder):
        if f_name.lower().endswith('.wav'):
            fpath = os.path.join(ip_folder, f_name)
            data, fs = sf.read(fpath)
            # Use previously detected peaks if available, otherwise skip
            # Only trim if peaks were detected for this file
            if 'peaks' in locals() and len(peaks) > 0:
                cut_audio = data[peaks[0]:]
                fpath_trim = os.path.join(ip_folder_trim, f_name)
                sf.write(fpath_trim, cut_audio, fs)
                print(f"Trimmed {fpath_trim} \n at first peak (time {peaks[0] / fs})")
            else:
                print(f"No peaks found in {fpath}, file not trimmed.")

    ip_folder_hp = os.path.join(output_dir, date_dir, robot_rec_dir, "hp_filtered_trimmed_audio", f"ip_{ip}")
    os.makedirs(ip_folder_hp, exist_ok=True)
    print('\n')
    for fname in os.listdir(ip_folder_trim):
        if fname.lower().endswith('.wav'):
            fpath = os.path.join(ip_folder_trim, fname)
            data, fs = sf.read(fpath)
            filtered = signal.sosfiltfilt(sos2, data)
            sf.write(os.path.join(ip_folder_hp, fname), filtered, fs)
            print(f"Applied HP filter to {os.path.join(ip_folder_hp, fname)}")
    print('\n')



# %%
