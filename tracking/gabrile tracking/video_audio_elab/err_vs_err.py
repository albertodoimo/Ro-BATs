import os
import yaml
import numpy as np
from matplotlib import pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

file_name = '20250516_15-59-38'
# Saving directory
dir_name = './plots/'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

with open('./non_blind_analysis/' + file_name + '.yaml', "r") as file:
    try:
        data = yaml.safe_load(file)  # Use safe_load to avoid potential security issues
    except yaml.YAMLError as error:
        print(f"Error loading YAML file: {error}")

# Extracting the data
obst_distances = data['obstacle_distances']
print(len(obst_distances))
distance_errors = data['distance_errors']
obst_angles = data['obstacle_angles']
angle_errors = data['angle_errors']

# Convert lists to numpy arrays for easier manipulation
obst_distances = np.array(obst_distances)
distance_errors = np.array(distance_errors)
dst_err_mean = np.mean(distance_errors)
dst_err_std = np.std(distance_errors)
obst_angles = np.array(obst_angles)
angle_errors = np.array(angle_errors)
ang_err_mean = np.mean(angle_errors)
ang_err_std = np.std(angle_errors)
# List of 6 dark colors
colors = ['#0000FF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000', '#800080']

bins = np.array([-90, -60, -30, 0, 30, 60, 90])
indices = np.digitize(obst_angles, bins, right=False) - 1
indices = np.clip(indices, 0, len(colors) - 1)
associated_colors = np.array(colors)[indices]

# Create a scatter plot for angle errors vs distance errors
plt.figure()
plt.scatter(angle_errors, distance_errors, color=associated_colors, alpha=0.5)
plt.title('Distance Errors vs Angle Errors')
plt.xlabel('Angle Errors (degrees)')
plt.ylabel('Distance Errors (cm)')
plt.grid()
# Legend for the colors
for i, color in enumerate(colors):
    plt.scatter([], [], color=color, label=fr'{bins[i]} [deg] $\leq$ $\theta_G$$_T$ < {bins[i+1]} [deg]')
plt.xlim(-180, 180)
plt.ylim(-90, 30)
plt.legend()
plt.show()
