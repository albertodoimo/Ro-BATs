import os
import yaml
import numpy as np
from matplotlib import pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# plt.rcParams.update({
# "text.usetex": True,
# "font.family": "serif",
# "font.serif": ["Computer Modern Roman"],
# "text.latex.preamble": r"""
# \usepackage{lmodern}
# \renewcommand{\rmdefault}{cmr}
# \renewcommand{\sfdefault}{cmss}
# \renewcommand{\ttdefault}{cmtt}
# """,
#     "font.size": 16,           # Set default font size
#     "axes.labelsize": 16,      # Axis label font size
#     "xtick.labelsize": 16,     # X tick label font size
#     "ytick.labelsize": 16,     # Y tick label font size
#     "legend.fontsize": 16,     # Legend font size
#     "axes.titlesize": 16       # Title font size
# })

file_names = os.listdir('./non_blind_analysis/')
# Saving directory
dir_name = './plots/'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

with open('./non_blind_analysis/' + file_names[3], "r") as file:
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

# Create a scatter plot for distance errors
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(obst_distances, distance_errors, color='blue', alpha=0.5)
plt.title('Distance Errors vs Obstacle Distances')
plt.xlabel('Obstacle Distances (cm)')
plt.ylabel('Distance Errors (cm)')
plt.grid()

plt.subplot(2, 2, 2)
# Create a scatter plot for angle errors
plt.scatter(obst_angles, angle_errors, color='red', alpha=0.5)
plt.title('Angle Errors vs Obstacle Angles')
plt.xlabel('Obstacle Angles (degrees)')
plt.ylabel('Angle Errors (degrees)')
plt.xlim(-120, 120)
plt.xticks([-120, -90, -60, -30, 0, 30, 60, 90, 120])
plt.grid()

# plt.subplot(2, 2, 3)
# # Violin plot for distance errors, blue
# vp1 = plt.violinplot(distance_errors, np.ones((1)), showmedians=True, showextrema=False)
# for body in vp1['bodies']:
#     body.set_facecolor('blue')
#     body.set_alpha(0.5)
# # Set means line color to black
# if 'cmedians' in vp1:
#     vp1['cmedians'].set_color('black')
# plt.text(0.2, 0.2, f'Mean: {dst_err_mean:.2f}cm\nStd: {dst_err_std:.2f}cm', 
#          horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
# # Hide the x-axis label
# plt.xticks([])
# plt.title('Distance Error')
# plt.ylabel('Distance Errors (cm)')
# plt.grid()

# plt.subplot(2, 2, 4)
# # Violin plot for angle errors, red
# vp2 = plt.violinplot(angle_errors, np.ones((1)), showmedians=True, showextrema=False)
# for body in vp2['bodies']:
#     body.set_facecolor('red')
#     body.set_alpha(0.5)
# # Set means line color to black
# if 'cmedians' in vp2:
#     vp2['cmedians'].set_color('black')
# plt.text(0.2, 0.2, f'Mean: {ang_err_mean:.2f}°\nStd: {ang_err_std:.2f}°', 
#          horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
# plt.title('Angle Error')
# plt.ylabel('Angle Errors (degrees)')
# plt.xticks([])
# plt.grid()
# plt.tight_layout()
# Save the figure
# plt.savefig(dir_name + file_name + '.png', dpi=300, bbox_inches='tight')
plt.show()
