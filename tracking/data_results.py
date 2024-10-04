#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file
global_path = '/Users/alberto/Documents/UNIVERSITA/MAGISTRALE/tesi/github/Ro-BATs/tracking/'
filename_CC = 'robat_data_2024-08-29__18-55-34 cut3_CC.csv'
filename_PRA = 'robat_data_2024-08-29__18-55-34 cut3_PRA.csv'

df_CC = pd.read_csv(global_path + filename_CC)
df_PRA = pd.read_csv(global_path + filename_PRA)
# Generate x-values (angles) from 0 to 360
angle_deg = np.arange(0, 360, 1)  # Integer values for angles from 0 to 360

# Extract the last column for y-values (error values)
error_values_CC = df_CC.iloc[:, -1]  # Assuming last column contains error values
error_values_PRA = df_PRA.iloc[:, -1]  # Assuming last column contains error values

error = np.size(360)
#%%
for i,ii in range(len(error_values_CC)):
    for ii in error:
        if int(error_values_CC[i]):
            print('error_values_PRA[i] ', error_values_CC[i])
            error_values_PRA[i] = 0
# Ensure the detected angle values match the length of the x-axis (truncate or extend)
detected_values_CC = np.resize(error_values_CC, len(angle_deg))
detected_values_PRA = np.resize(error_values_PRA, len(angle_deg))
# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# First subplot: Error values
axs[0].scatter(angle_deg, detected_values_CC, color='red')
axs[0].set_title('error Values CC')
axs[0].set_xlabel('Angle (degrees)')
axs[0].set_ylabel('Error (degrees)')
axs[0].grid(True)

# Second subplot: Detected angle values
axs[1].scatter(angle_deg, detected_values_PRA, color='blue')
axs[1].set_title('error Values MUSIC')
axs[1].set_xlabel('Angle (degrees)')
axs[0].set_ylabel('Error (degrees)')
axs[1].grid(True)

# Adjust layout and display the plots
plt.tight_layout()
plt.show()

# %%
