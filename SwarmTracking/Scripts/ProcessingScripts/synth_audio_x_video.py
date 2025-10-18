#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Created: 2025-10-15
Author: Alberto Doimo
email: alberto.doimo@uni-konstanz.de

Description:
Generates a synthetic audio file by placing chirp signals at specified call times loaded from CSV files.

"""
#%%
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile
import os

# Parameters
fs = 48000  # Sampling frequency
duration_chirp = 20e-3  # Chirp duration (s)
silence_post = 20e-3  # Silence after chirp (s)
amplitude = 0.5  # Chirp amplitude
start_f, end_f = 24e3, 2e3  # Frequency sweep (Hz)
# Set the project directory (three levels up from this script)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # project directory

# Directories
# Intermediate data directory
input_dir = "Data/IntermediateData/"
# Date directory
date_dir = '2025-10-02/'
# Directory for robot recordings (date and time)
robot_rec_dir = '2025-10-02_18-45-28/'

# LOAD CALL TIMES
files = [
    "ip_238_call_times.csv",
    "ip_240_call_times.csv",
    "ip_241_call_times.csv"
]

# Read all CSVs and combine times
call_times = []
for f in files:
    df = pd.read_csv(os.path.join(project_dir, input_dir, date_dir, robot_rec_dir,  f))
    # Assuming a single column of call times
    call_times.extend(df.iloc[:, 0].values)

call_times = np.unique(np.sort(call_times))
print(f"Loaded {len(call_times)} call times from {len(files)} files.")

# CHIRP GENERATION
t = np.linspace(0, duration_chirp, int(fs * duration_chirp), endpoint=False)
sweep = signal.chirp(t, f0=start_f, f1=end_f, t1=t[-1], method='linear')
sweep *= signal.windows.tukey(sweep.size, 0.2)
sweep *= 0.8  # taper amplitude
chirp_sig = sweep
chirp_sig *= amplitude

# BUILD FULL SIGNAL
total_duration = call_times[-1] + duration_chirp 
num_samples = int(np.ceil(total_duration * fs))
audio = np.zeros(num_samples)  # mono zeros

chirp_len = len(chirp_sig)

for t_call in call_times:
    start_idx = int(t_call * fs)
    end_idx = start_idx + chirp_len
    if end_idx <= num_samples:
        audio[start_idx:end_idx] = chirp_sig

# Clip in case of overlap
audio = np.clip(audio, -0.5, 0.5)

# SAVE OUTPUT 
wavfile.write(os.path.join(project_dir, input_dir, "combined_call_chirps.wav"), fs, audio.astype(np.float32))
print("WAV file saved as combined_call_chirps.wav")

# %%
