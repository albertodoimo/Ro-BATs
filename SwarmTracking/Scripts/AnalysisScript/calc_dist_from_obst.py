#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Created: 2025-11-16
Author: Alberto Doimo
email: alberto.doimo@uni-konstanz.de

Description:

Calculate distance from robot to closest obstacle for each frame 
"""

import pandas as pd
import numpy as np
import sys
import os
from config_analysis import *  
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
# -------------------------------------------
# Load CSV
# -------------------------------------------

# -------------------------------------------
# Get experiment date and time from config
# -------------------------------------------
# Check if experiment_date and experiment_time are defined in config
# Get list of available CSV files in the intermediate data directory
available_files = [f for f in os.listdir(interm_data_dir) if f.endswith('_tracking_data.csv')]

if not available_files:
    print("No tracking data files found in the intermediate data directory.")
    sys.exit(1)

# Extract date and time from filenames
experiments = []
for f in available_files:
    # Remove '_tracking_data.csv' suffix
    name = f.replace('_tracking_data.csv', '')
    # Split by underscore to get date and time
    parts = name.split('_')
    if len(parts) >= 2:
        experiments.append({'date': parts[0], 'time': parts[1], 'file': f})

if not experiments:
    print("No valid experiment files found.")
    sys.exit(1)

# Sort experiments by date and time
experiments.sort(key=lambda x: (x['date'], x['time']))

# Display options to user
print("\nAvailable experiments:")
for idx, exp in enumerate(experiments, 1):
    print(f"{idx}. {exp['date']}   {exp['time']}")

# Get user selection
while True:
    try:
        choice = int(input(f"\nSelect experiment (1-{len(experiments)}): "))
        if 1 <= choice <= len(experiments):
            selected_exp = experiments[choice - 1]
            break
        else:
            print(f"Please enter a number between 1 and {len(experiments)}")
    except ValueError:
        print("Please enter a valid number")

experiment_date = selected_exp['date']
experiment_time = selected_exp['time']
audio_csv_path = os.path.join(interm_data_dir, selected_exp['file'])
df = pd.read_csv(audio_csv_path)

print(f"Processing experiment: {experiment_date} at {experiment_time}")
# -------------------------------------------
# Identify obstacle coordinate pairs
# -------------------------------------------
obst_pairs = []
for col in df.columns:
    if col.startswith("obst_x_"):
        idx = col.split("_")[-1]
        y_col = f"obst_y_{idx}"
        if y_col in df.columns:
            obst_pairs.append((col, y_col))

# -------------------------------------------
# Compute distance to each obstacle or robot for all IDs
# -------------------------------------------
# Find all robot IDs
all_robot_ids = []
for col in df.columns:
    if col.startswith("x_"):
        robot_id = col.split("_", 1)[-1]
        y_col = f"y_{robot_id}"
        if y_col in df.columns:
            all_robot_ids.append(robot_id)

# Compute distances for each robot
all_robot_distances = {}
for ips in all_robot_ids:
    x = df[f"x_{ips}"]
    y = df[f"y_{ips}"]
    
    distances = []
    for ox, oy in obst_pairs:
        # Check if obstacle coordinates are NaN, if so compute distance to other robots
        obst_x = df[ox]
        obst_y = df[oy]
        
        # Create a distance array
        d = np.full(len(df), np.nan)
        
        # Where obstacle exists, compute distance to obstacle
        valid_obst = obst_x.notna() & obst_y.notna()
        d[valid_obst] = np.sqrt((obst_x[valid_obst] - x[valid_obst])**2 + 
                                 (obst_y[valid_obst] - y[valid_obst])**2)
        
        # Where obstacle is NaN, compute distance to other robots
        invalid_obst = ~valid_obst
        if invalid_obst.any():
            # Find other robot columns
            other_robot_ids = [rid for rid in all_robot_ids if rid != ips]
            for other_ips in other_robot_ids:
                other_x = df[f"x_{other_ips}"]
                other_y = df[f"y_{other_ips}"]
                valid_robot = other_x.notna() & other_y.notna() & invalid_obst
                if valid_robot.any():
                    robot_dist = np.sqrt((other_x[valid_robot] - x[valid_robot])**2 + 
                                        (other_y[valid_robot] - y[valid_robot])**2)
                    d[valid_robot] = np.where(np.isnan(d[valid_robot]), robot_dist, 
                                             np.minimum(d[valid_robot], robot_dist))
        
        distances.append(d)
    
    all_robot_distances[ips] = distances

# Store nearest obstacle distance for each robot
for ips, distances in all_robot_distances.items():
    if distances:
        nearest = np.nanmin(np.vstack(distances), axis=0)
        df[f"nearest_obstacle_distance_{ips}"] = nearest

# -------------------------------------------
# Compute polygon perimeter between robot centers
# -------------------------------------------

perimeters = np.full(len(df), np.nan)
normalized_perimeters = np.full(len(df), np.nan)

if len(all_robot_ids) >= 3:
    for idx in range(len(df)):
        # Collect all robot positions for this frame
        positions = []
        for robot_id in all_robot_ids:
            x_val = df.loc[idx, f"x_{robot_id}"]
            y_val = df.loc[idx, f"y_{robot_id}"]
            if pd.notna(x_val) and pd.notna(y_val):
                positions.append([x_val, y_val])
        
        # Need at least 3 robots to form a polygon
        if len(positions) >= 3:
            positions = np.array(positions)
            try:
                hull = ConvexHull(positions) # Compute the convex hull of the obstacle positions
                        # The convex hull is the smallest convex polygon/polyhedron that encloses all points
                        # This creates a boundary around the obstacles for distance calculations

                perimeter = hull.area  # In 2D, 'area' is actually the perimeter
                perimeters[idx] = perimeter
                # Normalize by number of robots
                normalized_perimeters[idx] = perimeter / len(positions)
            except:
                print(f"ConvexHull computation failed at frame {idx}.")

    df["polygon_perimeter"] = perimeters
    df["normalized_polygon_perimeter"] = normalized_perimeters

    # Save to CSV (only perimeter columns)
    output_file = os.path.join(analysis_data_dir, experiment_type, f"{experiment_date}_{experiment_time}_polygon_perimeter.csv")
    df[["polygon_perimeter", "normalized_polygon_perimeter"]].to_csv(output_file, index=False)
    print(f"Saved: {output_file}")


    # -------------------------------------------
    # Plot polygon perimeter over time
    # -------------------------------------------

    # Time-series plot of nearest obstacle distance
    # Convert frame index to seconds (15 fps)
    time_vals = np.arange(len(df)) / 15.0
    ts_col = "time"

    mask = df["polygon_perimeter"].notna()
    if mask.any():
        cm = 1/2.54
        fig, ax = plt.subplots(figsize=(15*cm, 10*cm), dpi=150)
        ax.plot(time_vals[mask], df.loc[mask, "polygon_perimeter"], lw=1)
        ax.set_xlabel(f"{ts_col} [s]")
        ax.set_ylabel("Perimeter [m]")
        ax.set_title(f"{experiment_date}_{experiment_time} Polygon perimeter over time")
        mean_perim = float(np.nanmean(df["polygon_perimeter"]))
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig_path = os.path.join(analysis_data_dir, f"{experiment_date}_{experiment_time}_polygon_perimeter.png")

        plt.show()
        save_fig = input(f"Do you want to save the figure {fig_path}? (y/n): ").strip().lower()
        if save_fig == 'y':
            fig.savefig(fig_path, dpi=150, transparent=True)
            print(f"Saved: {fig_path}")
        else:
            print("Figure not saved.")

    else:
        print("No valid polygon_perimeter values for time-series plot.")

else:
    print(f"Not enough robots to compute polygon perimeter: ({len(all_robot_ids)} found).")

# -------------------------------------------
# Nearest obstacle distance
# -------------------------------------------

# Compute nearest obstacle distance (handle NaNs)
if distances:
    nearest = np.nanmin(np.vstack(distances), axis=0)
    df["nearest_obstacle_distance"] = nearest
    output_file = os.path.join(analysis_data_dir, experiment_type, f"{experiment_date}_{experiment_time}_nearest_obstacle_distance.csv")
    df[["nearest_obstacle_distance"]].to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

    # Violin plot of nearest obstacle distances
    data = df["nearest_obstacle_distance"].dropna().values
    if data.size > 0:
        cm = 1/2.54  # centimeters in inches
        fig, ax = plt.subplots(figsize=(15*cm, 10*cm), dpi=150)
        vp = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False, widths=0.8)
        for body in vp["bodies"]:
            body.set_alpha(0.3)
        
        # Add only the mean as a horizontal line
        mean_val = float(np.nanmean(data))
        ax.plot([0.85, 1.15], [mean_val, mean_val], 'k-', lw=2, label='Mean')
        
        ax.set_title("Nearest obstacle distance")
        ax.set_ylabel("Distance [m]")
        ax.set_xticks([1])
        ax.set_xticklabels([str(ips)])
        ax.grid(alpha=0.3)
        ax.text(0.98, 0.95, f"Mean: {mean_val:.2f}", transform=ax.transAxes, ha="right", va="top")
        plt.tight_layout()
        fig_path = os.path.join(analysis_data_dir, experiment_type, f"{experiment_date}_{experiment_time}_{ips}_nearest_obstacle_distance_violin.png")
        plt.show()
        # Ask user if they want to save the figure
        save_fig = input(f"Do you want to save the figure {fig_path}? (y/n): ").strip().lower()
        if save_fig == 'y':
            fig.savefig(fig_path, dpi=150, transparent=True)
            print(f"Saved: {fig_path}")
        else:
            print("Figure not saved.")

    else:
        print("No nearest_obstacle_distance data to plot.")

    # Time-series plot of nearest obstacle distance
    # Convert frame index to seconds (15 fps)
    time_vals = np.arange(len(df)) / 15.0
    ts_col = "time"

    mask = df["nearest_obstacle_distance"].notna()
    if mask.any():
        cm = 1/2.54
        fig, ax = plt.subplots(figsize=(15*cm, 10*cm), dpi=150)
        ax.plot(time_vals[mask], df.loc[mask, "nearest_obstacle_distance"], lw=1)
        ax.set_xlabel(f"{ts_col} [s]")
        ax.set_ylabel("Distance [m]")
        ax.grid(alpha=0.3)
        ax.set_title(f"{experiment_date}_{experiment_time} Distance over time")
        plt.tight_layout()
        fig_path_ts = os.path.join(analysis_data_dir, experiment_type, f"{experiment_date}_{experiment_time}_{ips}_nearest_obstacle_distance.png")
        plt.show()
        save_fig = input(f"Do you want to save the figure {fig_path_ts}? (y/n): ").strip().lower()
        if save_fig == 'y':
            fig.savefig(fig_path_ts, dpi=150, transparent=True)
            print(f"Saved: {fig_path_ts}")
        else:
            print("Figure not saved.")

    else:
        print("No valid nearest_obstacle_distance values for time-series plot.")
else:
    df["nearest_obstacle_distance"] = np.nan
    print("No obstacle columns found; skipping violin plot.")

# -------------------------------------------
# Save output
# # -------------------------------------------
# output_file = "241_nearest_obstacle_distance.csv"
# df.to_csv(output_file, index=False)

# print(f"Saved: {output_file}")
