#%%

import numpy as np
import os
data = np.load('/home/alberto/Documents/ActiveSensingCollectives_lab/' \
'Ro-BATs/tracking/pypylon/data/2025-08-22_15-00-27:0876_basler_tracking_markers.npy', allow_pickle=True)
# %%
print(data)
# %%

data_robot = np.load('/home/alberto/Desktop/TIMESTAMPS_134.34.226.241_2025-08-21__17-35-44.npy', allow_pickle=True)
# %%
print(data_robot)

# %%
data_robot = np.load('/home/alberto/Desktop/experimental_data/TIMESTAMPS_134.34.226.241_2025-08-22__10-51-55.npy', allow_pickle=True)
print(data_robot)
# %%
np.set_printoptions(threshold=np.inf)
print("Extended data:")
print(data)

# %%

# timestamps_npy_file = os.path.join(dir, 'data', '134.34.226.241', 'TIMESTAMPS_134.34.226.241_2025-08-22__17-07-23.npy')
timestamps = np.load('/home/alberto/Documents/ActiveSensingCollectives_lab/Ro-BATs/tracking/audio_camera_sync/data/134.34.226.241/TIMESTAMPS_134.34.226.241_2025-08-22__17-07-23.npy', allow_pickle=True)
