#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Created: 2025-10-15
Author: Alberto Doimo
email: alberto.doimo@uni-konstanz.de

Description:
plots the distance as function of dB SPL for all robots and all frames

To BE REVISED ACCURATELY!!!
"""
#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

print('!! CHECK THIS SCRIPT CAREFULLY !!')

# Load dataset
# Get project directory and input file path
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
input_dir = "Data/IntermediateData/"
file_name = os.path.join(project_dir, input_dir, "2025-10-02_18-45-28_upsampled_tracking_data.csv")

# Load CSV data into DataFrame
df = pd.read_csv(file_name)

# Ensure required columns exist 
required_cols = ['frame', 'robot_id', 'center_x_m', 'center_y_m', 'dB_SPL_level']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Compute closest distance per frame 
closest_distances = []
for frame, group in df.groupby('frame'):
    # Skip frames with less than 2 robots
    if len(group) < 2:
        print(f"Skipping frame {frame} with {len(group)} robots.")
        continue

    # Get robot positions for this frame
    positions = group[['center_x_m', 'center_y_m']].to_numpy()
    # Compute pairwise Euclidean distances
    dist_mat = distance_matrix(positions, positions)
    # Ignore self-distances by setting diagonal to infinity
    np.fill_diagonal(dist_mat, np.inf)

    # Find indices of the closest pair
    min_idx = np.unravel_index(np.argmin(dist_mat), dist_mat.shape)
    r1, r2 = group.iloc[min_idx[0]], group.iloc[min_idx[1]]
    min_dist = dist_mat[min_idx]

    # Compute mean dB SPL level for the closest pair
    mean_db = (r1['dB_SPL_level'] + r2['dB_SPL_level']) / 2
    closest_distances.append({'frame': frame, 'distance': min_dist, 'mean_dB_SPL': mean_db})

# Convert to DataFrame 
dist_df = pd.DataFrame(closest_distances)

# Results 
if dist_df.empty:
    print("Not enough data to compute distances.")
else:
    print(f"Computed closest distances for {len(dist_df)} frames.")
    print("Example:")
    print(dist_df.head())

    # Plot 
    plt.figure(figsize=(8, 6))
    plt.scatter(dist_df['mean_dB_SPL'], dist_df['distance'], alpha=0.7, marker='.')
    plt.xlabel("Mean dB SPL Level")
    plt.ylabel("Closest Euclidean Distance (m)")
    plt.title("Closest Robot Distance vs. dB SPL Level")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# %%
