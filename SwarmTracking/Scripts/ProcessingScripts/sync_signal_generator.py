#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Created: 2025-10-1
Author: Alberto Doimo
email: alberto.doimo@uni-konstanz.de

Description:

Audio signal generator for robot and camera synchronization.

"""

import numpy as np
import soundfile as sf
import csv
import os
import matplotlib.pyplot as plt
import scipy.signal as signal
from Utils_SwarmTracking import *

if __name__ == "__main__":

    # Example parameters
    fps = 15  # minimum hardware sampling frequency
    P_min = 2*1/fps   # Minimum period in sec
    P_max = 4*P_min  # Maximum period in sec

    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 
    output_dir = "./Data/IntermediateData/"
    os.makedirs(output_dir, exist_ok=True)

    # pseudo-random waiting periods
    periods = generate_pseudo_random_signal(P_min, P_max, 20)

    # Save wait periods to CSV
    with open(output_dir + "wait_periods.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["wait_period_seconds"])
        for period in periods:
            writer.writerow([period])

    # Audio settings
    total_duration = 30*60  # seconds 
    interval = 60  # seconds interval between start of the segments
    fs = 48000 # audio sample rate
    freq = 15  # Hz square wave frequency

    ###################################################################
    # # Generate audio file with alternating white noise based on periods

    # _, noise = generate_alternating_white_noise_audio(periods, fs)

    # # Apply low-pass filter to the noise to avoid interference with robots DOA
    # sos = signal.butter(8, 2000, 'low', fs=fs, output='sos')
    # filtered_noise = signal.sosfiltfilt(sos, noise)
    # # Check if an .wav file already exists in the directory
    # existing_wav = [f for f in os.listdir(output_dir) if f.lower().endswith('.wav')]

    # if existing_wav:
    #     print(f"\nExisting .wav file found in {output_dir}: \n{existing_wav[0]}\n")
    #     user_input = input("A .wav file already exists. Do you want to overwrite it and continue? (y/n): ").strip().lower()
    #     if user_input != 'y':
    #         print("Operation cancelled by user.")
    #         exit(0)
    #     else:
    #         sf.write(output_dir + "filtered_noise.wav", filtered_noise, fs)

    # # Repeat the one-minute noise + silence pattern to fill the total duration
    # one_minute_noise_signal = np.concatenate((filtered_noise, np.zeros((int(fs * interval) - filtered_noise.shape[0]))))
    # sync_signal = np.tile(one_minute_noise_signal, total_duration // interval)

    ###################################################################
    # Generate audio file with linear frequency sweeps
    
    frequencies = (20, 4000)  
    silence_dur = 500  # milliseconds of silence
    sweep_signal, sig1, sig2 = generate_sweeps(frequencies, duration=2, fs=fs, silence_dur=silence_dur)

    # Apply low-pass filter to avoid interference with robots DOA
    sos = signal.butter(8, 2000, 'low', fs=fs, output='sos')
    filtered_sweep = signal.sosfiltfilt(sos, sweep_signal)

    # Check if an .wav file already exists in the directory
    existing_wav = [f for f in os.listdir(output_dir) if f.lower().endswith('.wav')]

    if existing_wav:
        print(f"\nExisting .wav file found in {output_dir}: \n{existing_wav[0]}\n")
        user_input = input("A .wav file already exists. Do you want to overwrite it and continue? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Operation cancelled by user.")
            exit(0)
        else:
            sf.write(output_dir + "filtered_sweep.wav", filtered_sweep, fs)

    # Repeat the one-minute silence pattern to fill the total duration
    sweep_template = np.concatenate((filtered_sweep, np.zeros((int(fs * interval) - filtered_sweep.shape[0]))))
    sync_signal = np.tile(sweep_template, total_duration // interval)

    ###################################################################
    # generate 15 Hz square wave
    t = np.linspace(0, total_duration, int(fs * total_duration), endpoint=False)
    square_wave = 0.8 * signal.square(2 * np.pi * freq * t)

    # Stack into stereo: L=square wave, R=sync signal
    stereo = np.stack([square_wave, sync_signal], axis=1)

    # Save to WAV file
    existing_wav = [f for f in os.listdir(output_dir) if f.lower().endswith('.wav')]

    if existing_wav:
        print(f"\nExisting .wav file found in {output_dir}: \n{existing_wav[0]}\n")
        user_input = input("A .wav file already exists. Do you want to overwrite it and continue? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Operation cancelled by user.")
            exit(0)
        else:
            sf.write(output_dir + f"{freq}Hz_tracking_sync_signal.wav", stereo, fs)


