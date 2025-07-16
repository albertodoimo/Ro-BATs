import os
import yaml
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

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


data_dir = './trajectories/'
obst_pos_dir = './obst_positions/'
file_names = os.listdir(data_dir)
obst_pos_files = os.listdir(obst_pos_dir)

group_indexes = [0, 2, 4, 6, -1]
with open(data_dir + 'conversion_factors.yaml', 'r') as file:
    try:
        data = yaml.safe_load(file)  # Use safe_load to avoid potential security issues
    except yaml.YAMLError as error:
        print(f"Error loading YAML file: {error}")
pixel_per_meter = np.array(data['pixel_to_meters'])
bottle_radius = 3.2e-2 # [m]
labels = ['A', 'B', 'C', 'D']  # Labels for the configurations
for i in range(len(group_indexes) - 1):
    trajectory = np.empty((0, 2))
    obst_positions = np.zeros((11, 2))
    for j, f in enumerate(file_names[group_indexes[i]:group_indexes[i+1]]):
        with open(data_dir + f, 'r') as file:
            try:
                data = yaml.safe_load(file)  # Use safe_load to avoid potential security issues
            except yaml.YAMLError as error:
                print(f"Error loading YAML file: {error}")
        with open(obst_pos_dir + obst_pos_files[j + group_indexes[i]], 'r') as file:
            try:
                pos_data = yaml.safe_load(file)  # Use safe_load to avoid potential security issues
            except yaml.YAMLError as error:
                print(f"Error loading YAML file: {error}")
        traj = np.array(data['trajectory'])
        norm_x = min(traj[:, 0])
        norm_y = min(traj[:, 1])
        traj[:, 0] -= norm_x  # Normalize x-coordinates
        traj[:, 1] -= norm_y  # Normalize y-coordinates
        traj /= pixel_per_meter[j + group_indexes[i]]  # Convert to meters
        trajectory = np.vstack((trajectory, traj))   
        pos = np.array(pos_data['obstacles_position'])
        pos[:, 0] -= norm_x
        pos[:, 1] -= norm_y
        pos /= pixel_per_meter[j + group_indexes[i]]
        obst_positions += pos
    obst_positions /= j + 1  # Average obstacle positions
    obst_positions[:, 1] = max(trajectory[:, 1]) - obst_positions[:, 1]
    vor = Voronoi(obst_positions)
    # fig, ax = plt.subplots(figsize=(6, 6))
    # plt.xlim(0, 2)
    # plt.ylim(0, 1.55)
    # plt.xlabel('X [m]')
    # plt.ylabel('Y [m]')
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.grid(False)
    # plt.scatter(obst_positions[:, 0], obst_positions[:, 1], 100, 'lime', 'o', label='Obstacles')
    # # plt.legend(loc='upper right', bbox_to_anchor=(0.4, -0.1))
    # plt.tight_layout()
    # plt.title(f'Voronoi Diagram for Obstacles - Configuration {labels[i]}')
    # plt.savefig(f'./voronoi_{i+1}.png', dpi=600, transparent=True)
    # obst_positions = np.flipud(obst_positions)
    # bottle_radius_pixels = bottle_radius * pixel_per_meter
    xedges = int((max(trajectory[:, 0]) - min(trajectory[:, 0]))/bottle_radius)

    yedges = int((max(trajectory[:, 1]) - min(trajectory[:, 1]))/bottle_radius)

    H, xedges, yedges = np.histogram2d(trajectory[:, 0], trajectory[:, 1], bins=[xedges, yedges])
    H = H.T
    H = np.flipud(H)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(aspect='equal')
    X, Y = np.meshgrid(xedges, yedges)
    mesh = ax.pcolormesh(X, Y, H, cmap='plasma', vmax=30)
    cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', shrink=0.5)
    cbar.set_label('Number of visits')
    ax.scatter(obst_positions[:, 0], obst_positions[:, 1], 100, 'lime', 'o', label='Obstacles')
    voronoi_plot_2d(vor, ax, show_vertices=False, show_points=False, line_colors='c', line_width=1.5, line_alpha=1, line_style='--')
    ax.set_title('Configuration' + f' {labels[i]}')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.55)
    # set legend outside the plot
    # ax.legend(loc='upper right', bbox_to_anchor=(0.4, -0.2))
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(f'./obst_{i+1}.png', dpi=600, transparent=True)

data_dir = './trajectories_control/'
file_names = os.listdir(data_dir)
with open(data_dir + 'conversion_factors.yaml', 'r') as file:
    try:
        data = yaml.safe_load(file)  # Use safe_load to avoid potential security issues
    except yaml.YAMLError as error:
        print(f"Error loading YAML file: {error}")
pixel_per_meter = np.array(data['pixel_to_meters'])
for k, f in enumerate(file_names[:-1]):
    with open(data_dir + f, 'r') as file:
        try:
            data = yaml.safe_load(file)  # Use safe_load to avoid potential security issues
        except yaml.YAMLError as error:
            print(f"Error loading YAML file: {error}")
    traj = np.array(data['trajectory'])  # Assuming the trajectory is the first key in the dictionary
    traj[:, 0] -= min(traj[:, 0])  # Normalize x-coordinates
    traj[:, 1] -= min(traj[:, 1])  # Normalize y-coordinates
    traj /= pixel_per_meter[k]  # Convert to meters
    trajectory = traj

    xedges = int((max(trajectory[:, 0]) - min(trajectory[:, 0]))/bottle_radius)

    yedges = int((max(trajectory[:, 1]) - min(trajectory[:, 1]))/bottle_radius)

    H, xedges, yedges = np.histogram2d(trajectory[:, 0], trajectory[:, 1], bins=[xedges, yedges])
    H = H.T
    H = np.flipud(H)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(aspect='equal')
    X, Y = np.meshgrid(xedges, yedges)
    mesh = ax.pcolormesh(X, Y, H, cmap='plasma', vmax=30)
    cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', shrink=0.5)
    cbar.set_label('Number of visits')
    ax.set_title(f'Control {k+1}')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.55)
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(f'./control_{k+1}.png', dpi=600, transparent=True)

plt.tight_layout()
plt.show()
