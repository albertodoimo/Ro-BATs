#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Created: 2025-11-16
Author: Alberto Doimo
email: alberto.doimo@uni-konstanz.de

Description:
Check the npy file and display its content
"""
#%%
import numpy as np
import os
import glob
from config import experiment_date, experiment_time, ips

# Build the directory path
robot_ip = f"134.34.226.{ips[0]}"
data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Data", "InputData", experiment_date, f"{experiment_date}_{experiment_time}"
)

# Search for timestamp file (might have slightly different time suffix)
pattern = os.path.join(data_dir, f"TIMESTAMPS_{robot_ip}_*.npy")
timestamp_files = glob.glob(pattern)

if not timestamp_files:
    print(f"ERROR: No timestamp files found matching pattern: {pattern}")
    exit(1)

timestamp_path = timestamp_files[0]
timestamp_file = os.path.basename(timestamp_path)

print(f"Loading timestamp file: {timestamp_file}")
print(f"Full path: {timestamp_path}")
print("-" * 80)

# Load and print the data
if os.path.exists(timestamp_path):
    timestamps = np.load(timestamp_path, allow_pickle=True)
    print(f"\nTimestamp array shape: {timestamps.shape}")
    print(f"Data type: {timestamps.dtype}")
    print(f"\nAll timestamps:")
    print(timestamps)
    print(f"\nTotal number of timestamps: {len(timestamps)}")
    
    # If timestamps are numeric, calculate some statistics
    if np.issubdtype(timestamps.dtype, np.number):
        print(f"\nStatistics:")
        print(f"  Min: {np.min(timestamps)}")
        print(f"  Max: {np.max(timestamps)}")
        print(f"  Mean: {np.mean(timestamps)}")
        print(f"  Std: {np.std(timestamps)}")
        
        # Calculate time differences if there are multiple timestamps
        if len(timestamps) > 1:
            diffs = np.diff(timestamps)
            print(f"\nTime differences between consecutive timestamps:")
            print(f"  Min diff: {np.min(diffs)}")
            print(f"  Max diff: {np.max(diffs)}")
            print(f"  Mean diff: {np.mean(diffs)}")
            print(f"  Std diff: {np.std(diffs)}")
else:
    print(f"\nERROR: File not found at {timestamp_path}")

# %%
