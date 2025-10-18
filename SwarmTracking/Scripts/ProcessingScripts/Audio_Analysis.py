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

# --- Configuration and Paths ---
ips = [238, 240, 241]
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Project directory
input_dir = "./Data/IntermediateData/"
date_dir = '2025-10-02/'
robot_rec_dir = '2025-10-02_18-45-28/'

trim_dir = os.path.join(project_dir, input_dir, date_dir, robot_rec_dir, "trimmed_audio")  # Trimmed audio directory

# --- Audio Parameters ---
trigger_level = 70  # dB SPL
critical_level = 75  # dB SPL
c = 343  # Speed of sound
fs = 48000
rec_samplerate = 48000
input_buffer_time = 0.04  # seconds
block_size = int(input_buffer_time * fs)
channels = 5
mic_spacing = 0.018  # meters
ref = channels // 2  # Central mic as reference

# --- DOA Algorithm Parameters ---
method = 'DAS'
theta_das = np.linspace(-90, 90, 61)  # Angles for DAS spectrum
N_peaks = 1  # Number of peaks to detect

# --- Chirp Signal Parameters ---
duration_out = 20e-3
silence_post = 110  # ms
amplitude = 0.5

t = np.linspace(0, duration_out, int(fs * duration_out))
start_f, end_f = 24e3, 2e3
sweep = signal.chirp(t, start_f, t[-1], end_f)
sweep *= signal.windows.tukey(sweep.size, 0.2)
sweep *= 0.8

silence_samples_post = int(silence_post * fs / 1000)
silence_vec_post = np.zeros((silence_samples_post,))
post_silence_sig = np.concatenate((sweep, silence_vec_post))
full_sig = post_silence_sig

stereo_sig = np.hstack([full_sig.reshape(-1, 1), full_sig.reshape(-1, 1)])
data = amplitude * np.float32(stereo_sig)

out_blocksize = int(len(data))
print('out_blocksize =', out_blocksize)

# --- Frequency Calculations ---
auto_hipas_freq = int(343 / (2 * (mic_spacing * (channels - 1))))
print('HP frequency:', auto_hipas_freq)
auto_lowpas_freq = int(343 / (2 * mic_spacing))
print('LP frequency:', auto_lowpas_freq)
highpass_freq, lowpass_freq = [auto_hipas_freq, auto_lowpas_freq]
freq_range = [start_f, end_f]

cutoff = auto_hipas_freq
sos = signal.butter(1, cutoff, 'hp', fs=fs, output='sos')

# --- Mics Sensitivity ---
script_dir = os.path.dirname(os.path.abspath(__file__))
sensitivity_path = os.path.join(project_dir, input_dir, 'Knowles_SPH0645LM4H-B_sensitivity.csv')
sensitivity = pd.read_csv(sensitivity_path)

analyzed_buffer_time = 0.01
block_size_analyzed_buffer = int(analyzed_buffer_time * fs)

in_sig = np.zeros(block_size_analyzed_buffer, dtype=np.float32)
print('in_sig shape:', np.shape(in_sig))
centrefreqs, freqrms = calc_native_freqwise_rms(in_sig, fs)

freqs = np.array(sensitivity.iloc[:, 0])
sens_freqwise_rms = np.array(sensitivity.iloc[:, 1])
interp_sensitivity = interpolate_freq_response([freqs, sens_freqwise_rms], centrefreqs)

frequency_band = [2e3, 20e3]
tgtmic_relevant_freqs = np.logical_and(centrefreqs >= frequency_band[0], centrefreqs <= frequency_band[1])

# --- Parameter Summary ---
print(f"\n !! Verify parameter matching between this signal and the one used in the experiment !! \n")
print(f" Current signal characteristics: ")
print(f"  - Sampling rate (Fs): {fs} Hz")
print(f"  - Frequency sweep range: {start_f} Hz to {end_f} Hz")
print(f"  - Amplitude: {amplitude}")
print(f"  - Duration: {duration_out} s ({duration_out*1000:.1f} ms)")
print(f"  - Silence after call: {silence_post} ms ({silence_samples_post} samples)")
print(f"  - Output block size: {out_blocksize} samples")
print(f"  - Channels: {channels}")
print(f"  - Highpass filter cutoff: {highpass_freq} Hz")
print(f"  - Lowpass filter cutoff: {lowpass_freq} Hz")
print(f"  - Analyzed buffer time: {analyzed_buffer_time} s ({block_size_analyzed_buffer} samples)")
print(f"  - Trigger level: {trigger_level} dB SPL")
print(f"  - Critical level: {critical_level} dB SPL")
print(f"  - Robot recording directory: {robot_rec_dir}")

#%%

# --- Stack Multi-channel Audio for DOA calculation ---
for ip in ips:
    ip_folder = os.path.join(trim_dir, f"ip_{ip}")
    if not os.path.exists(ip_folder):
        print(f"Warning: Folder {ip_folder} does not exist. Skipping IP {ip}.")
        continue
    mic_data = []
    mic_files = []
    for mic_idx in range(channels):
        mic_fname = f"mic_{mic_idx+1}.wav"
        mic_fpath = os.path.join(ip_folder, mic_fname)
        if os.path.exists(mic_fpath):
            mic_audio, _ = sf.read(mic_fpath)
            mic_data.append(mic_audio)
            mic_files.append(mic_fname)
        else:
            print(f"Warning: File {mic_fpath} does not exist. Skipping mic {mic_idx} for IP {ip}.")
    if len(mic_data) != channels:
        print(f"Warning: Not all channels found for IP {ip}. Found {len(mic_data)} channels. Skipping.")
        continue

    mic_data = np.stack(mic_data, axis=-1)
    multichannel_path = os.path.join(ip_folder, f"ip_{ip}_multichannel.wav")
    sf.write(multichannel_path, mic_data, fs)
    print(f"Saved multi-channel audio for IP {ip} to {multichannel_path}, shape: {mic_data.shape}, channel order: {mic_files}")

#  Apply DAS algorithms as in the robats to extract DOA ---
for ip in ips:
    results = []
    print(f"IP: {ip}")
    peak_times_path = os.path.join(project_dir, input_dir, date_dir, robot_rec_dir, f"ip_{ip}_call_times.csv")
    if not os.path.exists(peak_times_path):
        print(f"Warning: Peak times file {peak_times_path} does not exist. Skipping.")
        continue
    peak_times = pd.read_csv(peak_times_path)
    if 'Call_time' in peak_times.columns:
        peak_times = peak_times['Call_time'].values.flatten()
    else:
        peak_times = peak_times.values.flatten()
    if len(peak_times) < 2:
        print(f"Warning: Not enough peak times in {peak_times_path} to process. Skipping.")
        continue

    for i in range(len(peak_times) - 1):
        peak_time = peak_times[i]
        next_peak_time = peak_times[i + 1]
        intercall_audio = mic_data[int((peak_time + duration_out) * fs):int((next_peak_time - 0.01) * fs), :]

        ts = silence_post / 1000 - input_buffer_time
        if intercall_audio.shape[0] < int(ts * fs):
            print(f"Skipping: intercall_audio too short for ts={ts:.3f}s (length={intercall_audio.shape[0]/fs:.3f}s)")
            continue
        # Take a chunk of length input_buffer_time starting from ts for all channels
        start_idx = int(ts * fs)
        end_idx = start_idx + int(input_buffer_time*fs)
        buffer = intercall_audio[start_idx:end_idx, :]

        # # Plot waveform and spectrogram for the reference channel
        import matplotlib.pyplot as plt

        ref_channel_audio = buffer[:, ref]

        plt.figure(figsize=(12, 6))

        # Waveform
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(ref_channel_audio.size) / fs, ref_channel_audio)
        plt.title(f'Waveform (IP {ip}, Call {i}, Ref Channel)')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')

        # Spectrogram
        plt.subplot(2, 1, 2)
        f, t_spec, Sxx = signal.spectrogram(ref_channel_audio, fs=fs, nperseg=128, noverlap=64)
        plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud')
        plt.title('Spectrogram')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.colorbar(label='Power [dB]')
        plt.tight_layout()
        plt.show()


        peak_angles = [None]
        dB_SPL_level = None
        peak_angles, dB_SPL_level = update_das(
            buffer, fs, sos, ref, analyzed_buffer_time, tgtmic_relevant_freqs,
            interp_sensitivity, das_filter, channels, mic_spacing,
            highpass_freq, lowpass_freq, theta_das, critical_level,
            trigger_level, N_peaks
        )

        results.append([peak_time, peak_angles, dB_SPL_level[0]])

        results_df = pd.DataFrame(results, columns=['Call_time', 'DOA_angle', 'dB_SPL_level'])
        results_df.to_csv(
            os.path.join(project_dir, input_dir, date_dir, robot_rec_dir, f"ip_{ip}_audio_tracking.csv"),
            index=False, header=True
        )
    print(f"Saved results for IP {ip} to ip_{ip}_audio_tracking.csv")

# %%
