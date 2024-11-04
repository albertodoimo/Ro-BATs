#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file
global_path = '/Users/alberto/Documents/UNIVERSITA/MAGISTRALE/tesi/github/Ro-BATs/tracking/'
save_path = '/Users/alberto/Documents/UNIVERSITA/MAGISTRALE/tesi/github/Ro-BATs/tracking/figures/'
name = 'robat_data_gcc 2024-10-25__13-06-12_CC'
filename = name +'.csv'

df = pd.read_csv(global_path + filename)

#%%

angle_deg = np.linspace(0, 360, len(df))  # Adjust x-axis based on the number of rows in df

# Extract the error and detected angle columns
error = df.iloc[:, -1]  # Last column contains error values
gt_angle = df.iloc[:, 2]  # gt angle from 3rd column

# Create the scatter plot with all detected values
plt.figure(figsize=(10, 6))
plt.scatter(gt_angle, error, label='Error', color='red', marker='.', alpha=0.7)


# Add labels and title
plt.xlabel('gt (degrees)')
plt.ylabel('error (degrees)')
plt.title(f'Scatter Plot of Error in: \n {filename}' )

# Optional: Add a grid for better readability
plt.grid(True)

# Compute and plot the overall mean line
overall_mean_error = np.mean(error)  # Compute the overall mean of the error
plt.axhline(y=overall_mean_error, color='green', linestyle='--', label=f'Mean Error = {overall_mean_error:.2f}')
plt.ylim([0,250])


# Show the legend and the plot
plt.legend()

plt.savefig(save_path+name)

plt.show()
# %%
