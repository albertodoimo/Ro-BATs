import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Set font to Times New Roman
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['text.usetex'] = False  # Use if LaTeX is not required

# Paths for the data and figure save locations
global_path = '/Users/alberto/Documents/UNIVERSITA/MAGISTRALE/tesi/github/Ro-BATs/tracking/csv/'
save_path = '/Users/alberto/Documents/UNIVERSITA/MAGISTRALE/tesi/github/Ro-BATs/tracking/figures/'

# File names for each run
filenames = [
    ('robat_data_linux_gcc_20241023_165510 cut_CC.csv',
     'robat_data_linux_srp_20241023_172103 cut_SRP.csv',
     'robat_data_linux_music_20241023_175337 cut_MUSIC.csv'),
    ('robat_data_linux_gcc_20241024_162305_CC.csv',
     'robat_data_linux_spr_20241024_164412_SRP.csv',
     'robat_data_linux_music_20241024_170519_MUSIC.csv'),
    ('robat_data_linux_gcc_20241025_100745_CC.csv',
     'robat_data_linux_srp_20241025_102503_SRP.csv',
     'robat_data_linux_music_20241025_104942_MUSIC.csv'),
    ('robat_data_linux_gcc 2024-10-25__13-06-12_CC.csv',
     'robat_data_linux_srp 2024-10-25__12-49-27_SRP.csv',
     'robat_data_linux_music 2024-10-25__11-46-11_MUSIC.csv')
]

# Plot parameters
labelsize = 10
titlesize = 15
pad = 15
s = 5
alpha = 0.3

# Create the 4x3 subplot layout
fig, axes = plt.subplots(4, 3, figsize=(11, 9))
plt.subplots_adjust(left=0.1, right=0.2, top=0.7, bottom=0.2, wspace=0.8, hspace=0.5)

# Loop through each run and plot in the respective row
for i, (filename1, filename2, filename3) in enumerate(filenames):
    # Load data for each algorithm in the current run
    df1 = pd.read_csv(global_path + filename1)
    df2 = pd.read_csv(global_path + filename2)
    df3 = pd.read_csv(global_path + filename3)
    
    # Extract data from each DataFrame
    error1, gt_angle1 = df1.iloc[:, -1], df1.iloc[:, 2]
    error2, gt_angle2 = df2.iloc[:, -1], df2.iloc[:, 2]
    error3, gt_angle3 = df3.iloc[:, -1], df3.iloc[:, 2]
    
    # Plot each algorithm in the corresponding column of the current row
    axes[i, 0].scatter(gt_angle1, error1, color='red', marker='o', alpha=alpha, s=s)
    #axes[i, 0].set_title(f'Run {i+1} - GCC-PHAT', fontsize=titlesize, pad=pad)
    
    axes[i, 1].scatter(gt_angle2, error2, color='green', marker='o', alpha=alpha, s=s)
    #axes[i, 1].set_title(f'Run {i+1} - SRP-PHAT', fontsize=titlesize, pad=pad)
    
    axes[i, 2].scatter(gt_angle3, error3, color='blue', marker='o', alpha=alpha, s=s)
    #axes[i, 2].set_title(f'Run {i+1} - MU.SI.C', fontsize=titlesize, pad=pad)
    
    # Set axis labels for the last row only
    if i == 3:
        axes[i, 0].set_xlabel('Ground truth angle [degrees]', fontsize=labelsize)
        axes[i, 1].set_xlabel('Ground truth angle [degrees]', fontsize=labelsize)
        axes[i, 2].set_xlabel('Ground truth angle [degrees]', fontsize=labelsize)

    # Set y-axis label for the first column only
    axes[i, 0].set_ylabel('\n\nError [degrees]', fontsize=labelsize)

    # Add "Run X" label on the y-axis beside each row
    fig.text(0.02, 0.83 - i*0.23, f'RUN {i+1}', va='center', ha='center', fontsize=titlesize, rotation='vertical')


    if i==0:
        axes[i, 0].set_title(f'GCC-PHAT', fontsize=titlesize, pad=pad)
        axes[i, 1].set_title(f'SRP-PHAT', fontsize=titlesize, pad=pad)
        axes[i, 2].set_title(f'MU.SI.C', fontsize=titlesize, pad=pad)

    # Set limits and grid for all subplots
    for j in range(3):
        axes[i, j].set_ylim([0, 180])
        axes[i, j].set_xticks(np.arange(0, 361, 30))
        axes[i, j].set_yticks(np.arange(0, 181, 30))
        axes[i, j].grid(True)

# Show the plot
plt.suptitle('COMPARISON BETWEEN DIFFERENT RUNS', fontsize=18, y=1.01)
plt.tight_layout()


# Save figure if needed
plt.savefig(save_path + 'comparison_summary.png', dpi=300)
plt.show()


#MEAN and STD

save_name = 'SUMMARY'
name1 = 'robat_gcc_sum_linux'
filename1 = name1 + '.csv'
name2 = 'robat_srp_sum_linux'
filename2 = name2 + '.csv'
name3 = 'robat_music_sum_linux'
filename3 = name3 + '.csv'

df1 = pd.read_csv(global_path + filename1)
df2 = pd.read_csv(global_path + filename2)
df3 = pd.read_csv(global_path + filename3)

# Create figure
plt.figure(figsize=(10, 6))
labelsize = 13
titlesize = 15
pad = 15

# Extract data
error1 = df1.iloc[:, -1]
gt_angle1 = df1.iloc[:, 2]
error2 = df2.iloc[:, -1]
gt_angle2 = df2.iloc[:, 2]
error3 = df3.iloc[:, -1]
gt_angle3 = df3.iloc[:, 2]

# Calculate mean and std for each method across angle ranges
angle_bins = np.arange(0, 361, 10)  # Create bins every 10 degrees
bin_centers = (angle_bins[:-1] + angle_bins[1:]) / 2

def calculate_stats(angles, errors, bins):
    means = []
    stds = []
    for i in range(len(bins)-1):
        mask = (angles >= bins[i]) & (angles < bins[i+1])
        means.append(np.mean(errors[mask]))
        stds.append(np.std(errors[mask]))
    return np.array(means), np.array(stds)

mean1, std1 = calculate_stats(gt_angle1, error1, angle_bins)
mean2, std2 = calculate_stats(gt_angle2, error2, angle_bins)
mean3, std3 = calculate_stats(gt_angle3, error3, angle_bins)

# Plot mean lines and standard deviation bands
plt.plot(bin_centers, mean1, 'r-', label=f'GCC-PHAT (Mean: {np.mean(error1):.2f}°)', linewidth=2)
plt.fill_between(bin_centers, mean1-std1, mean1+std1, color='red', alpha=0.2)

plt.plot(bin_centers, mean2, 'g-', label=f'SRP-PHAT (Mean: {np.mean(error2):.2f}°)', linewidth=2)
plt.fill_between(bin_centers, mean2-std2, mean2+std2, color='green', alpha=0.2)

plt.plot(bin_centers, mean3, 'b-', label=f'MU.SI.C (Mean: {np.mean(error3):.2f}°)', linewidth=2)
plt.fill_between(bin_centers, mean3-std3, mean3+std3, color='blue', alpha=0.2)

plt.xlabel('Ground truth angle [degrees]', fontsize=labelsize)
plt.ylabel('Error [degrees]', fontsize=labelsize)
plt.ylim([0, 180])
plt.xticks(np.arange(0, 361, 30))
plt.yticks(np.arange(0, 181, 30))
plt.grid(True)
plt.title(f'{save_name}: Mean Error and Standard Deviation', fontsize=titlesize, pad=pad)
plt.legend(loc='upper right')

# Save and show plot
plt.savefig(save_path + save_name + '_std_only', dpi=300, bbox_inches='tight')
plt.show()