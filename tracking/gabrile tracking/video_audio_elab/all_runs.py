import os
import yaml
import numpy as np
from matplotlib import pyplot as plt


def compute_stats(x, y, bins):
    means = []
    stds = []
    for i in range(len(bins)-1):
        mask = (x >= bins[i]) & (x < bins[i+1])
        if any(mask) and y[mask].size > 5:
            means.append(np.mean(y[mask]))
            stds.append(np.std(y[mask]))
        else:
            means.append(np.nan)
            stds.append(np.nan)
    return np.array(means), np.array(stds)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams.update({
"text.usetex": True,
"font.family": "serif",
"font.serif": ["Computer Modern Roman"],
"text.latex.preamble": r"""
\usepackage{lmodern}
\renewcommand{\rmdefault}{cmr}
\renewcommand{\sfdefault}{cmss}
\renewcommand{\ttdefault}{cmtt}
""",
    "font.size": 16,           # Set default font size
    "axes.labelsize": 16,      # Axis label font size
    "xtick.labelsize": 16,     # X tick label font size
    "ytick.labelsize": 16,     # Y tick label font size
    "legend.fontsize": 16,     # Legend font size
    "axes.titlesize": 16       # Title font size
})

# results_dir = './plots/'
# if not os.path.exists(results_dir):
#     os.makedirs(results_dir)

data_dir = './non_blind_analysis/'
# data_dir = './analysis/'

data_files = os.listdir(data_dir)
# Filter out non-YAML files
data_files = [f for f in data_files if f.endswith('.yaml')]

obst_distances = []
distance_errors = []
obst_angles = []
angle_errors = []

for file_name in data_files:
    with open(data_dir + file_name, "r") as file:
        try:
            data = yaml.safe_load(file)  # Use safe_load to avoid potential security issues
        except yaml.YAMLError as error:
            print(f"Error loading YAML file: {error}")

        # Extracting the data
        obst_distances.append(data['obstacle_distances'])
        distance_errors.append(data['distance_errors'])
        obst_angles.append(data['obstacle_angles'])
        angle_errors.append(data['angle_errors'])

# Flatten the lists of lists into single lists
obst_distances = [item for sublist in obst_distances for item in sublist]
distance_errors = [item for sublist in distance_errors for item in sublist]
obst_angles = [item for sublist in obst_angles for item in sublist]
angle_errors = [item for sublist in angle_errors for item in sublist]

# Print the number of data points
print(f"Number of data points: {len(obst_distances)}")

# Convert lists to numpy arrays for easier manipulation
obst_distances = np.array(obst_distances)
distance_errors = np.array(distance_errors)
obst_angles = np.array(obst_angles)
angle_errors = np.array(angle_errors)
distance_errors *= -1
angle_errors *= -1

correct = True
if correct:
    ol_distances = distance_errors[(obst_distances > 100) | (obst_distances < 12.5)]
    ol_distances = np.vstack((ol_distances, obst_distances[(obst_distances > 100) | (obst_distances < 12.5)]))
    more_ol = distance_errors[distance_errors < -10]
    more_ol = np.vstack((more_ol, obst_distances[distance_errors < -10]))
    ol_distances = np.hstack((ol_distances, more_ol))

    ol_angles = angle_errors[(obst_distances > 100) | (obst_distances < 12.5)]
    ol_angles = np.vstack((ol_angles, obst_angles[(obst_distances > 100) | (obst_distances < 12.5)]))
    more_ol = angle_errors[distance_errors < -10]
    more_ol = np.vstack((more_ol, obst_angles[distance_errors < -10])) 
    ol_angles = np.hstack((ol_angles, more_ol))

    ol_angles, idx = np.unique(ol_angles, axis=1, return_index=True)
    ol_distances = ol_distances[:, idx]
    distance_errors[(obst_distances > 100) | (obst_distances < 12.5)] = np.nan
    obst_distances[(obst_distances > 100) | (obst_distances < 12.5)] = np.nan
    obst_distances[distance_errors < -10] = np.nan
    distance_errors[distance_errors < -10] = np.nan
    obst_angles[distance_errors == np.nan] = np.nan
    angle_errors[distance_errors == np.nan] = np.nan
    distance_errors = distance_errors[~ np.isnan(obst_distances)]
    obst_angles = obst_angles[~ np.isnan(obst_distances)]
    angle_errors = angle_errors[~ np.isnan(obst_distances)]
    obst_distances = obst_distances[~ np.isnan(obst_distances)]
    print(f"Number of data points after outliers elimination: {len(obst_angles)}")


dst_err_mean = np.mean(distance_errors)
dst_err_median = np.median(distance_errors)
dst_err_std = np.std(distance_errors)
print(f"Distance Error - Mean: {dst_err_mean:.2f} cm, Median: {dst_err_median:.2f} cm, Std Dev: {dst_err_std:.2f} cm")

ang_err_mean = np.mean(angle_errors)
ang_err_median = np.median(angle_errors)
ang_err_std = np.std(angle_errors)
print(f"Angle Error - Mean: {ang_err_mean:.2f} deg, Median: {ang_err_median:.2f} deg, Std Dev: {ang_err_std:.2f} deg")

angle_bins = np.arange(-90, 91, 5)  # Create bins every 5 deg
ang_bin_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
ang_mean, ang_std = compute_stats(obst_angles, angle_errors, angle_bins)


distance_bins = np.arange(10, 101, 3)  # Create bins every 5 deg
dist_bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
dist_mean, dist_std = compute_stats(obst_distances, distance_errors, distance_bins)

ang_mean, ang_std = compute_stats(obst_angles, angle_errors, angle_bins)
dist_mean, dist_std = compute_stats(obst_distances, distance_errors, distance_bins)

dist_vs_ang_mean, dist_vs_ang_std = compute_stats(obst_angles, distance_errors, angle_bins)
ang_vs_dist_mean, ang_vs_dist_std = compute_stats(obst_distances, angle_errors, distance_bins)

# Scatter plots
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.scatter(obst_distances, distance_errors, color='blue', alpha=0.5, label='Data Points')
# plt.scatter(ol_distances[1, :], ol_distances[0, :], color='k', alpha=0.5, marker='x', label='Outliers')
# # plt.title('Distance Error vs Obstacle Distance', fontsize=20)
# plt.xlabel('Obstacle Distance [cm]', fontsize=16)
# plt.ylabel('Distance Error [cm]', fontsize=16)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# plt.xlim(10, 100)
# plt.ylim(-75, 25)
# plt.grid()
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=14)  # Horizontal legend below x-axis
# plt.subplot(1, 2, 2)
# plt.scatter(obst_angles, distance_errors, color='blue', alpha=0.5, label='Data Points')
# plt.scatter(ol_angles[1, :], ol_distances[0, :], color='k', alpha=0.5, marker='x', label='Outliers')
# # plt.title('Distance Error vs Obstacle Angle', fontsize=20)
# plt.xlabel('Obstacle Angle [deg]', fontsize=16)
# plt.ylabel('Distance Error [cm]', fontsize=16)
# plt.grid()
# plt.xlim(90, -90)
# plt.ylim(-75, 25)
# plt.xticks(np.arange(-90, 91, 30))
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=14)
# plt.tight_layout()
# plt.savefig('dda_scatter', dpi=600, transparent=True)

# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.scatter(obst_distances, angle_errors, color='red', alpha=0.5, label='Data Points')
# plt.scatter(ol_distances[1, :], ol_angles[0, :], color='k', alpha=0.5, marker='x', label='Outliers')
# # plt.title('Angle Error vs Obstacle Distance', fontsize=20)
# plt.xlabel('Obstacle Distance [cm]', fontsize=16)
# plt.ylabel('Angle Error [deg]', fontsize=16)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# plt.xlim(10, 100)
# plt.ylim(-100, 100)
# plt.grid()
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=14)
# plt.subplot(1, 2, 2)
# plt.scatter(obst_angles, angle_errors, color='red', alpha=0.5, label='Data Points')
# plt.scatter(ol_angles[1, :], ol_angles[0, :], color='k', alpha=0.5, marker='x', label='Outliers')
# # plt.title('Angle Error vs Obstacle Angle', fontsize=20)
# plt.xlabel('Obstacle Angle [deg]', fontsize=16)
# plt.ylabel('Angle Error [deg]', fontsize=16)
# plt.xlim(90, -90)
# plt.ylim(-100, 100)
# plt.xticks(np.arange(-90, 91, 30))
# plt.grid()
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=14)
# plt.tight_layout()
# plt.savefig('ada_scatter', dpi=600, transparent=True)

# # Line plots
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 2)
# plt.plot(ang_bin_centers, ang_mean, linewidth=2, color='red')
# plt.fill_between(ang_bin_centers, ang_mean-ang_std, ang_mean+ang_std, color='red', alpha=0.2,linewidth=2)
# # plt.title('Angle Error vs Obstacle Angle', fontsize=20)
# plt.xlabel('Obstacle Angle [deg]', fontsize=16)
# plt.ylabel('Angle Error [deg]', fontsize=16)
# plt.xlim(90, -90)
# plt.ylim(-50, 50)
# plt.yticks(np.arange(-50, 51, 25))
# plt.xticks(np.arange(-90, 91, 30))
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# plt.grid()
# plt.subplot(1, 2, 1)
# plt.plot(dist_bin_centers, ang_vs_dist_mean, linewidth=2, color='red')
# plt.fill_between(dist_bin_centers, ang_vs_dist_mean-ang_vs_dist_std, ang_vs_dist_mean+ang_vs_dist_std, color='red', alpha=0.2,linewidth=2)
# # plt.title('Angle Error vs Obstacle Distance', fontsize=20)
# plt.xlabel('Obstacle Distance [cm]', fontsize=16)
# plt.ylabel('Angle Error [deg]', fontsize=16)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# plt.xlim(10, 100)
# plt.ylim(-20, 22)
# plt.grid()
# plt.tight_layout()
# plt.savefig('ada_line', dpi=600, transparent=True)

# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 2)
# plt.plot(ang_bin_centers, dist_vs_ang_mean, linewidth=2, color='blue')
# plt.fill_between(ang_bin_centers, dist_vs_ang_mean-dist_vs_ang_std, dist_vs_ang_mean+dist_vs_ang_std, color='blue', alpha=0.2,linewidth=2)
# # plt.title('Distance Error vs Obstacle Angle', fontsize=20)
# plt.xlabel('Obstacle Angle [deg]', fontsize=16)
# plt.ylabel('Distance Error [cm]', fontsize=16)
# plt.xlim(90, -90)
# plt.ylim(-4, 4)
# plt.xticks(np.arange(-90, 91, 30))
# plt.grid()
# plt.subplot(1, 2, 1)
# plt.plot(dist_bin_centers, dist_mean, linewidth=2, color='blue')
# plt.fill_between(dist_bin_centers, dist_mean-dist_std, dist_mean+dist_std, color='blue', alpha=0.2,linewidth=2)
# # plt.title('Distance Error vs Obstacle Distance', fontsize=20)
# plt.xlabel('Obstacle Distance [cm]', fontsize=16)
# plt.ylabel('Distance Error [cm]', fontsize=16)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# plt.grid()
# plt.xlim(10, 100)
# plt.ylim(-6, 4)
# plt.tight_layout()
# plt.savefig('dda_line', dpi=600, transparent=True)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
# Violin plot for distance errors, blue
vp1 = plt.violinplot(distance_errors, np.ones((1)), showmeans=True, showextrema=False)
for body in vp1['bodies']:
    body.set_facecolor('blue')
    body.set_alpha(0.5)
# Set means line color to black
if 'cmeans' in vp1:
    vp1['cmeans'].set_color('black')
# plt.text(0.2, 0.85, f'Mean: {dst_err_mean:.2f} [cm]\nMedian: {dst_err_median:.2f} [cm]\nStd. dev.: {dst_err_std:.2f} [cm]',
#          horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)
# Hide the x-axis label
plt.xticks([])
# plt.title('Distance Error Density Function', fontsize=20)
plt.ylabel('Distance Error [cm]', fontsize=16)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='black', label='Mean')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=14)
plt.ylim(-10, 10)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.grid()

plt.subplot(1, 2, 2)
# Violin plot for angle errors, red
vp2 = plt.violinplot(angle_errors, np.ones((1)), showmeans=True, showextrema=False)
for body in vp2['bodies']:
    body.set_facecolor('red')
    body.set_alpha(0.5)
# Set means line color to black
if 'cmeans' in vp2:
    vp2['cmeans'].set_color('black')
# plt.text(0.2, 0.85, f'Mean: {ang_err_mean:.2f} [deg]\nMedian: {ang_err_median:.2f} [deg]\nStd. dev.: {ang_err_std:.2f} [deg]',
        #  horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)
# plt.title('Angle Error Density Function', fontsize=20)
plt.ylabel('Angle Error [deg]', fontsize=16)
plt.xticks([])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.ylim(-50, 50)
plt.yticks(np.arange(-50, 51, 25))
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='black', label='Mean')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=14)
plt.grid()
plt.tight_layout()
plt.savefig('violin', dpi=600, transparent=True)
plt.show()
