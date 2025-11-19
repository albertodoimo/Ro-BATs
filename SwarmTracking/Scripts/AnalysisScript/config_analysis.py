#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Created: 2025-11-15
Author: Alberto Doimo
email: alberto.doimo@uni-konstanz.de

Description:
Define parameters and configurations for the processing pipeline
"""
#%%
import os
import numpy as np
# # Single experiment date and time
experiment_date = "2025-11-17"  
experiment_time = "14-01-30"

#%%
# Project directories
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Extract experiment dates from InputData directory
input_base_dir = "./Data/InputData/"
experiment_dates = [d for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]

# List available experiment times in the input data directory
input_data_dir = f"./Data/InputData/{experiment_date}/"
experiment_times = [d.split('_')[1] for d in os.listdir(input_data_dir) if os.path.isdir(os.path.join(input_data_dir, d))]

input_dir = "./Data"
orig_dir = f"./Data/InputData/{experiment_date}/{experiment_date}_{experiment_time}/"
temporary_frames_dir = f"./Data/IntermediateData/Renamed_video_frames/{experiment_date}_{experiment_time}/"
interm_data_dir = "./Data/IntermediateData/"
date_time_folder = os.path.join(interm_data_dir, experiment_date, experiment_time)
file_name = f"{experiment_date}_{experiment_time}"
output_video_path = f"./Data/IntermediateData/{file_name}.mp4"
output_video_path_upsampled = f"./Data/IntermediateData/{file_name}_upsampled.mp4"
tracked_video_path = os.path.join(interm_data_dir, file_name + '_tracked.mp4')
audio_csv_path = os.path.join(interm_data_dir, file_name + '_tracking_data.csv')
analysis_data_dir = "./Data/AnalysisData/"
output_path = "./Output/"

#%%
# Expertiment parameters
experiment_type = "multi_dispersion" # multi_dispersion , multi_attraction, single_attraction, single_repulsion
num_robots = 1
ips = [241] # Robot IP addresses

# Obstacles markers IDs
obst_ids = [13,14,0,8,9]  # IDs of the ArUco markers for the arena corners
obst_ids = [13 ]  # IDs of the ArUco markers for the arena corners

# Aruco markers pair corresponding to each robot
marker_pairs = [(4, 5), (6, 7), (10, 11)]
robot_names = {"241": (4, 5), "240": (6, 7), "238": (10, 11)}

# Color map for different robot IDs
color_map = {
    "241": (255, 255, 0),    # Cyan
    "240": (0, 255, 0),    # Green
    "238": (0, 0, 255),    # Red
            }

# Chirp signal template
start_f, end_f = 9e3, 4e3
fs_call_sig = 48000 # Sampling frequency
duration_out = 20e-3  # Duration in seconds
silence_post = 200 # [ms]
amplitude = 1 # Amplitude of the chirp

# --- Audio Parameters ---
trigger_level = 65  # dB SPL
critical_level = 85  # dB SPL
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


# Arena dimensions in meters
arena_height = 1.40 # m
arena_width = 2.00 # m

# plots

label_fontsize = 9
tick_fontsize = 7
fig_width = 12.2  # cm: 12.2 cm for llncs template
fig_height = 9  # cm