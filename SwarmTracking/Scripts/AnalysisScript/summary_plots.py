#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Created: 2025-11-18
Author: Alberto Doimo
email: alberto.doimo@uni-konstanz.de

Description:
Plots results from various runs of experiments. 
"""
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
from config_analysis import *


available_files = [f for f in os.listdir(analysis_data_dir+experiment_type) if f.endswith('.csv')]
print("Available CSV files for plotting:", available_files)

# # Define the folder path
plot_type = "polygon_perimeter" # distance_from_obstacle, polygon_perimeter
fig_path = os.path.join(output_path, f"{experiment_type}_{plot_type}.png")
os.makedirs(os.path.dirname(fig_path), exist_ok=True)

# Create figure
cm_to_inch = 1/2.54
fig, ax = plt.subplots(figsize=(fig_width*cm_to_inch, fig_height*cm_to_inch), dpi=150)
# Plot each CSV file
legend_labels = []
for csv_file in available_files:
    file_path = os.path.join(analysis_data_dir+experiment_type, csv_file)
    df = pd.read_csv(file_path)
    
    time_vals = np.arange(len(df)) / 15.0
    ts_col = "Time"

    mask = df["polygon_perimeter"].notna()
    if mask.any():
        ax.plot(time_vals[mask], df.loc[mask, "polygon_perimeter"], lw=1) # polygon_perimeter, normalized_polygon_perimeter
        # Extract date and time from csv filename for legend
        # Extract only date and time from filename (remove trailing parts)
        label = '_'.join(csv_file.replace('.csv', '').split('_')[:2])
        legend_labels.append(label)

    else:
        print(f"No valid polygon_perimeter values for {csv_file}.")

if legend_labels:
    ax.set_xlabel(f"{ts_col} [s]", fontsize = label_fontsize)
    ax.tick_params(axis='x', labelsize=tick_fontsize)
    ax.set_ylabel("Convex Polygon Perimeter [m]", fontsize = label_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    # ax.set_title(f"{experiment_type}: Nearest obstacle distance over time")
    ax.grid(alpha=0.3)
    # ax.legend(legend_labels, fontsize=6, loc ='best')
    plt.tight_layout()
    plt.show()
    # Ask user if they want to save the figure
    save_fig = input(f"Do you want to save the figure {fig_path}? (y/n): ").strip().lower()
    if save_fig == 'y':
        fig.savefig(fig_path, dpi=150, transparent=True)
        print(f"Saved: {fig_path}")
    else:
        print("Figure not saved.")
else:
    print("No valid data to plot.")
    plt.close(fig)


############################################################################


# Create violin plot
plot_type_violin = "distance_from_obstacles_violin"
fig_path_violin = os.path.join(output_path, f"{experiment_type}_{plot_type_violin}.png")

# Collect data for violin plot
violin_data = []
violin_labels = []

for csv_file in available_files:
    file_path = os.path.join(analysis_data_dir+experiment_type, csv_file)
    df = pd.read_csv(file_path)
    mask = df["nearest_obstacle_distance"].notna()
    if mask.any():
        violin_data.append(df.loc[mask, "nearest_obstacle_distance"].values)
        trial_number = len(violin_labels) + 1
        trial_duration = len(df.loc[mask]) / 15.0
        violin_labels.append(f"Trial {trial_number}\n{trial_duration:.0f} sec")

    # Sort violin data and labels by time
    sorted_pairs = sorted(zip(violin_labels, violin_data), key=lambda x: x[0])
    violin_labels = [pair[0] for pair in sorted_pairs]
    violin_data = [pair[1] for pair in sorted_pairs]

# --- Plotting ---
# Create only one subplot
fig, ax = plt.subplots(figsize=(fig_width*cm_to_inch, fig_height*cm_to_inch), dpi=150)

# ax.set_title('Customized violin plot with Percentiles')
parts = ax.violinplot(
        violin_data, showmeans=False, showmedians=False,
        showextrema=False)

for pc in parts['bodies']:
    pc.set_alpha(0.5)

# Calculate quartiles and medians manually
quartile1 = []
medians = []
quartile3 = []

for d in violin_data:
    q1, med, q3 = np.percentile(d, [10, 50, 90])
    quartile1.append(q1)
    medians.append(med)
    quartile3.append(q3)

quartile1 = np.array(quartile1)
medians = np.array(medians)
quartile3 = np.array(quartile3)

inds = np.arange(1, len(medians) + 1)
ax.scatter(inds, medians, marker='o', color='black', s=30, zorder=3)
ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=1)

# Add text labels for percentiles
for i, (q1, med, q3) in enumerate(zip(quartile1, medians, quartile3)):
    # Adjust x_pos slightly to the right of the center line
    x_pos = inds[i] + 0.2
    
    # Add text with formatting
    ax.text(x_pos, q1, f'10%:{q1:.2f}', verticalalignment='center', fontsize=tick_fontsize, color='grey')
    ax.text(x_pos+0.1, med, f'50%:{med:.2f}', verticalalignment='center', fontsize=tick_fontsize, color='black')
    ax.text(x_pos, q3, f'90%:{q3:.2f}', verticalalignment='center', fontsize=tick_fontsize, color='grey')

ax.set_ylabel("Distance [m]", fontsize = label_fontsize)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
ax.set_yticklabels([f'{tick:.1f}' for tick in ax.get_yticks()], fontsize=tick_fontsize)
ax.set_xticks(np.arange(1, len(violin_labels) + 1))
ax.set_xticklabels(violin_labels, ha='center', fontsize=label_fontsize)
# plt.tight_layout()
ax.set_xlim(0.6, len(violin_labels) + 0.9)

# Adjust margins to ensure text fits
# plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95) 
plt.show()
save_fig_violin = input(f"Do you want to save the violin plot {fig_path_violin}? (y/n): ").strip().lower()
if save_fig_violin == 'y':
    fig.savefig(fig_path_violin, dpi=150, transparent=True)
    print(f"Saved: {fig_path_violin}")
else:
    print("Violin plot not saved.")