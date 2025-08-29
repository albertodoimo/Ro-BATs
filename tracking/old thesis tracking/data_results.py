#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import soundfile as wav

# Function to compute RMS of audio signal
# Set font to Times New Roman
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['text.usetex'] = False  # Use if LaTeX is not required

# Load the data from the CSV file
global_path = '/Users/alberto/Documents/UNIVERSITA/MAGISTRALE/tesi/github/Ro-BATs/tracking/csv/'
save_path = '/Users/alberto/Documents/UNIVERSITA/MAGISTRALE/tesi/github/Ro-BATs/tracking/figures/'

#save_name = 'RUN 1'
#name1 = 'robat_data_gcc_20241023_165510 cut_3_CC'
#filename1 = name1 +'.csv'
#name2 = 'robat_data_srp_20241023_172103 cut_SRP'
#filename2 = name2 +'.csv'
#name3 = 'robat_data_music_20241023_175337 cut_MUSIC'
#filename3 = name3 +'.csv'

#zsave_name = 'RUN 2'
#name1 = 'robat_data_gcc_20241024_162305_CC'
#filename1 = name1 +'.csv'
#name2 = 'robat_data_spr_20241024_164412_SRP'
#filename2 = name2 +'.csv'
#name3 = 'robat_data_music_20241024_170519_MUSIC'
#filename3 = name3 +'.csv'

#save_name = 'RUN 3'
#name1 = 'robat_data_gcc_20241025_100745_CC'
#filename1 = name1 +'.csv'
#name2 = 'robat_data_srp_20241025_102503_SRP'
#filename2 = name2 +'.csv'
#name3 = 'robat_data_music_20241025_104942_MUSIC'
#filename3 = name3 +'.csv'

#save_name = 'RUN 4'
#name1 = 'robat_data_gcc 2024-10-25__13-06-12_CC'
#filename1 = name1 +'.csv'
#name2 = 'robat_data_srp 2024-10-25__12-49-27_SRP'
#filename2 = name2 +'.csv'
#name3 = 'robat_data_music 2024-10-25__11-46-11_MUSIC'
#filename3 = name3 +'.csv'

# CORRECTED ERROR

save_name = 'RUN 1'
name1 = 'robat_data_linux_gcc_20241023_165510 cut_CC'
filename1 = name1 +'.csv'
name2 = 'robat_data_linux_srp_20241023_172103 cut_SRP'
filename2 = name2 +'.csv'
name3 = 'robat_data_linux_music_20241023_175337 cut_MUSIC'
filename3 = name3 +'.csv'

# save_name = 'RUN 2'
# name1 = 'robat_data_linux_gcc_20241024_162305_CC'
# filename1 = name1 +'.csv'
# name2 = 'robat_data_linux_spr_20241024_164412_SRP'
# filename2 = name2 +'.csv'
# name3 = 'robat_data_linux_music_20241024_170519_MUSIC'
# filename3 = name3 +'.csv'

# save_name = 'RUN 3'
# name1 = 'robat_data_linux_gcc_20241025_100745_CC'
# filename1 = name1 +'.csv'
# name2 = 'robat_data_linux_srp_20241025_102503_SRP'
# filename2 = name2 +'.csv'
# name3 = 'robat_data_linux_music_20241025_104942_MUSIC'
# filename3 = name3 +'.csv'

# save_name = 'RUN 4'
# name1 = 'robat_data_linux_gcc 2024-10-25__13-06-12_CC'
# filename1 = name1 +'.csv'
# name2 = 'robat_data_linux_srp 2024-10-25__12-49-27_SRP'
# filename2 = name2 +'.csv'
# name3 = 'robat_data_linux_music 2024-10-25__11-46-11_MUSIC'
# filename3 = name3 +'.csv'

save_name = 'SUMMARY'
name1 = 'robat_gcc_sum_linux'
filename1 = name1 +'.csv'
name2 = 'robat_srp_sum_linux'
filename2 = name2 +'.csv'
name3 = 'robat_music_sum_linux'
filename3 = name3 +'.csv'

df1 = pd.read_csv(global_path + filename1)
df2 = pd.read_csv(global_path + filename2)
df3 = pd.read_csv(global_path + filename3)

labelsize = 13
legendsize = 8
titlesize = 12
pad = 15
s = 5
alpha = 0.3
line_styles = ['-', '--', '-.', ':']

#%%
angle_deg = np.linspace(0, 360, len(df1))  # Adjust x-axis based on the number of rows in df
angle_deg2 = np.linspace(0, 360, len(df2)) 
angle_deg3 = np.linspace(0, 360, len(df3)) 

# Extract the error and detected angle columns
error1 = df1.iloc[:, -1]  # Last column contains error values
gt_angle1 = df1.iloc[:, 2]  # gt angle from 3rd column
error2 = df2.iloc[:, -1]  # Last column contains error values
gt_angle2 = df2.iloc[:, 2]  # gt angle from 3rd column
error3 = df3.iloc[:, -1]  # Last column contains error values
gt_angle3 = df3.iloc[:, 2]  # gt angle from 3rd column

# Create the scatter plot with all detected values
plt.figure(figsize=(6, 10))
plt.suptitle(f'{save_name} COMPARISON', fontsize= 16)


plt.subplot(311)
plt.scatter(gt_angle1, error1, label='Error', color='red', marker='o', alpha=alpha, s=s)
plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)

# Add labels and title
#plt.xlabel('\nGround truth angle [degrees]\n',fontsize=labelsize)
plt.ylabel('Error [degrees]',fontsize=labelsize)
plt.ylim([0,180])
plt.xticks(np.arange(0, 361, 30))
plt.yticks(np.arange(0, 181, 30))
plt.grid(True)
plt.title(f'GCC-PHAT', fontsize=titlesize, pad=pad)
#plt.legend(loc='upper right', fontsize=legendsize, title_fontsize= legendsize)

# Compute and plot the overall mean line
#overall_mean_error = np.mean(error)  # Compute the overall mean of the error
#plt.axhline(y=overall_mean_error, color='green', linestyle='--', label=f'Mean Error = {overall_mean_error:.2f}')

plt.subplot(312)
plt.scatter(gt_angle2, error2, label='Error', color='green', marker='o', alpha=alpha, s=s)

# Add labels and title
#plt.xlabel('\nGround truth angle [degrees]\n',fontsize=labelsize)
plt.ylabel('Error [degrees]',fontsize=labelsize)
plt.ylim([0,180])
plt.xticks(np.arange(0, 361, 30))
plt.yticks(np.arange(0, 181, 30))
plt.grid(True)
plt.title(f'SRP-PHAT', fontsize=titlesize, pad=pad)
#plt.legend(loc='upper right', fontsize=legendsize, title_fontsize= legendsize)

# Compute and plot the overall mean line
#overall_mean_error = np.mean(error)  # Compute the overall mean of the error
#plt.axhline(y=overall_mean_error, color='green', linestyle='--', label=f'Mean Error = {overall_mean_error:.2f}')

plt.subplot(313)
plt.scatter(gt_angle3, error3, label='Error', color='blue', marker='o', alpha=alpha, s=s)

# Add labels and title
plt.xlabel('\nGround truth angle [degrees]\n',fontsize=labelsize)
plt.ylabel('Error [degrees]',fontsize=labelsize)
plt.ylim([0,180])
plt.xticks(np.arange(0, 361, 30))
plt.yticks(np.arange(0, 181, 30))
plt.grid(True)
plt.title(f'MUSIC', fontsize=titlesize, pad=pad)
#plt.legend(loc='upper right', fontsize=legendsize, title_fontsize= legendsize)

# Compute and plot the overall mean line
#overall_mean_error = np.mean(error)  # Compute the overall mean of the error
#plt.axhline(y=overall_mean_error, color='green', linestyle='--', label=f'Mean Error = {overall_mean_error:.2f}')


# Show the legend and the plot

plt.savefig(save_path+save_name, dpi=300, bbox_inches='tight')
plt.show()

# %%
