import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import seaborn as sns

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
labelsize = 13
titlesize = 18
pad = 15
s = 5
alpha = 0.3

#Create the 4x3 subplot layout
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
    #axes[i, 2].set_title(f'Run {i+1} - MUSIC', fontsize=titlesize, pad=pad)
    
    # Set axis labels for the last row only
    if i == 3:
        axes[i, 0].set_xlabel('\nGround truth angle [degrees]', fontsize=labelsize)
        axes[i, 1].set_xlabel('\nGround truth angle [degrees]', fontsize=labelsize)
        axes[i, 2].set_xlabel('\nGround truth angle [degrees]', fontsize=labelsize)

    # Set y-axis label for the first column only
    axes[i, 0].set_ylabel('\n\nError [degrees]', fontsize=labelsize)

    # Add "Run X" label on the y-axis beside each row
    fig.text(0.02, 0.83 - i*0.23, f'RUN {i+1}', va='center', ha='center', fontsize=titlesize, rotation='vertical')


    if i==0:
        axes[i, 0].set_title(f'GCC-PHAT', fontsize=titlesize, pad=pad)
        axes[i, 1].set_title(f'SRP-PHAT', fontsize=titlesize, pad=pad)
        axes[i, 2].set_title(f'MUSIC', fontsize=titlesize, pad=pad)

    # Set limits and grid for all subplots
    for j in range(3):
        axes[i, j].set_ylim([0, 180])
        axes[i, j].set_xticks(np.arange(0, 361, 30))
        axes[i, j].set_yticks(np.arange(0, 181, 30))
        axes[i, j].grid(True)

# Show the plot
plt.suptitle('COMPARISON BETWEEN DIFFERENT RUNS\n', fontsize=18)
plt.tight_layout()


# Save figure if needed
plt.savefig(save_path + 'comparison_summary.png', dpi=300)
plt.show(block=False)


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
plt.figure(figsize=(10, 7))
labelsize = 20
titlesize = 22
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
#plt.plot(bin_centers, mean1, 'r-', label=f'GCC-PHAT (Mean: {np.mean(error1):.1f}°, STD: {np.std(error1):.1f}°)', linewidth=2)
#plt.fill_between(bin_centers, mean1-std1, mean1+std1, color='red', alpha=0.2,linewidth=2)

#plt.plot(bin_centers, mean2, 'g-', label=f'SRP-PHAT (Mean: {np.mean(error2):.1f}° STD: {np.std(error2):.1f}°)', linewidth=2)
#plt.fill_between(bin_centers, mean2-std2, mean2+std2, color='green', alpha=0.2,linewidth=2)
#
plt.plot(bin_centers, mean3, 'b-', label=f'MUSIC (Mean: {np.mean(error3):.1f}° STD: {np.std(error3):.1f}°)', linewidth=2)
plt.fill_between(bin_centers, mean3-std3, mean3+std3, color='blue', alpha=0.2,linewidth=2)

plt.xlabel('Ground truth angle [degrees]', fontsize=labelsize)
plt.ylabel('Error [degrees]', fontsize=labelsize)
plt.ylim([0, 180])
plt.xticks(np.arange(0, 361, 30))
plt.yticks(np.arange(0, 181, 30))
plt.grid(True)
plt.title(f'{save_name}: Mean Error and Standard Deviation', fontsize=titlesize, pad=pad)
plt.legend(loc='upper right', fontsize=labelsize)

# Save and show plot
plt.savefig(save_path + save_name + '_std_music', dpi=300, bbox_inches='tight')
#plt.show()

# Create figure
plt.figure(figsize=(10, 7))

# Prepare data for violin plot
data_dict = {
    'GCC-PHAT': error1,
    'SRP-PHAT': error2,
    'MUSIC': error3
}

# Calculate statistics
stats = {
    'GCC-PHAT': {'mean': np.mean(error1), 'std': np.std(error1)},
    'SRP-PHAT': {'mean': np.mean(error2), 'std': np.std(error2)},
    'MUSIC': {'mean': np.mean(error3,), 'std': np.std(error3)}
}

# Create violin plot
parts = plt.violinplot([error1, error2, error3], 
                      showmeans=True, showextrema=False)

# Customize violin plot colors
colors = ['red', 'green','blue' ]
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.6)
    pc.set_edgecolor(colors[i])
    pc.set_linewidth(2)

# Customize other elements
parts['cmeans'].set_color('black')
parts['cmeans'].set_linewidth(1)

# Customize the plot
plt.xticks([1, 2, 3], [f'\nGCC-PHAT\nMEAN={stats["GCC-PHAT"]["mean"]:.1f}°\nSTD={stats["GCC-PHAT"]["std"]:.1f}°',
                       f'\nSRP-PHAT\nMEAN={stats["SRP-PHAT"]["mean"]:.1f}°\nSTD={stats["SRP-PHAT"]["std"]:.1f}°',
                       f'\nMUSIC\nMEAN={stats["MUSIC"]["mean"]:.1f}°\nSTD={stats["MUSIC"]["std"]:.1f}°'], fontsize=labelsize)

plt.ylabel('Error [degrees]', fontsize=labelsize)
plt.title('Error Distribution Comparison', fontsize=titlesize, pad=15)
plt.grid(True, alpha=0.3)

# Add legend for mean and median
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='black', linewidth=2, label='Mean')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=labelsize)

# Set y-axis limits
plt.ylim(0, 180)
plt.yticks(np.arange(0, 181, 30))

# Adjust layout
plt.tight_layout()

# Save and show plot
plt.savefig(save_path + save_name + '_error_violin', dpi=300, bbox_inches='tight')
plt.show()

# Load collision probability data
collision_prob = [0, 0, (15/33)*100]

# Define labels and colors for each method
labels = ['GCC-PHAT', 'SRP-PHAT', 'MUSIC']
colors = ['red', 'green', 'blue']

# Create figure
plt.figure(figsize=(5,5))

# Plot histogram
plt.bar(labels, collision_prob, color=colors, alpha=0.6, edgecolor=colors,linewidth=2, width=0.5)

# Customize the plot
for i, val in enumerate(collision_prob):
    plt.text(i, val + 0.02, f'{val:.1f}', ha='center', va='bottom', fontsize=labelsize)

plt.ylabel('Probability (%)', fontsize=labelsize)
plt.title('Collision Probability Comparison', fontsize=titlesize-2, pad=15)
plt.grid(axis='y', alpha=0.3)


# Add legend for mean
from matplotlib.lines import Line2D

# Set y-axis limits
plt.ylim(-5, 100)
plt.yticks(np.arange(0, 100, 10))


# Adjust layout
plt.tight_layout()

# Save and show plot
plt.savefig(save_path + 'collision_probability_histogram', dpi=300, bbox_inches='tight')
plt.show()
