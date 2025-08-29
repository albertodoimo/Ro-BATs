# %%

import numpy as np 
import soundfile as sf
import matplotlib.pyplot as plt 
from utilities import *
import scipy.signal as sig
import csv
import os


# %%
# import signature noise signal
dir = os.path.dirname(os.path.abspath(__file__))
video_out_signal, fs = sf.read(os.path.join(dir, 'data/video', 'alternating_white_noise.wav'))
video_out_signal = video_out_signal[fs*5:fs*6]  # take only one channel

#%% 
# Plot the time-domain signal and spectrogram
plt.figure(figsize=(15, 8))

# Time-domain plot
plt.subplot(2, 1, 1)
plt.plot(np.linspace(0, len(video_out_signal) / fs, len(video_out_signal)), video_out_signal)
plt.title('Output Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# Spectrogram plot
plt.subplot(2, 1, 2, sharex=plt.gca())
plt.specgram(video_out_signal, Fs=fs, NFFT=512, noverlap=256, cmap='viridis')
plt.title('Spectrogram of the Output Signal')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.ylim(0, 25e3)
plt.tight_layout()
# plt.show()

# %%
# import robot audio 

robat_238_raw_audio, fs = sf.read(os.path.join(dir, 'data','134.34.226.241', 'MULTIWAV_134.34.226.241_2025-08-21__16-04-26.wav'))
robat_238_raw_audio, fs = sf.read(os.path.join(dir, 'data','134.34.226.238', 'MULTIWAV_134.34.226.238_2025-08-22__19-17-06.wav'))

# cut the signal
start = 0
end  = 23
robat_238_raw_audio = robat_238_raw_audio[fs*start:fs*end,:]

#%% # Plot the time-domain signal and spectrogram

channel_to_show = 2
robat_238_audio = robat_238_raw_audio[:,channel_to_show]
plt.figure(figsize=(15, 8))

# Time-domain plot
plt.subplot(2, 1, 1)
plt.plot(np.linspace(0, len(robat_238_audio) / fs, len(robat_238_audio)), robat_238_audio, label=f'Mic {channel_to_show+1}')
plt.title('Robot Audio Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()

# Spectrogram plot
plt.subplot(2, 1, 2, sharex=plt.gca())
plt.specgram(robat_238_audio, Fs=fs, NFFT=512, noverlap=256, cmap='viridis')
plt.title('Spectrogram of the Robot Audio Signal')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.ylim(0, 25e3)
plt.tight_layout()
# plt.show()

# %%
# Apply the highpass filter
cutoff = 50 # cutoff frequency in Hz
# Plot the filter frequency response
sos = signal.butter(2, cutoff, 'hp', fs=fs, output='sos')
robat_238_audio_hp = sig.sosfilt(sos, robat_238_audio)

#%% 
#correlate
filtered_output = np.roll(signal.correlate(robat_238_audio_hp, video_out_signal, mode='same', method='direct'), -len(video_out_signal)//2)
filtered_envelope = np.abs(signal.hilbert(filtered_output))

# plot the envelope
plt.figure(figsize=(15, 5))
plt.plot(np.linspace(0, len(filtered_envelope) / fs, len(filtered_envelope)), filtered_envelope)
plt.title('Correlation result')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.tight_layout()
# plt.show()

peaks, properties = signal.find_peaks(filtered_envelope, prominence=4)

print(f'{np.shape(peaks)} peaks are found at:', peaks / fs, '[s]')

#%%
# plot the peaks
plt.figure(figsize=(15, 5))
plt.plot(np.linspace(0, len(robat_238_audio_hp) / fs, len(robat_238_audio_hp)), robat_238_audio_hp)
plt.plot(peaks/fs, robat_238_audio_hp[peaks], 'ro')
plt.title('Detected Peaks')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.tight_layout()
# plt.xlim([9.48,9.50])
# plt.show()
# %%
# Load timestamps from CSV file
# timestamps_file = os.path.join(dir, 'data', '134.34.226.241', 'TIMESTAMPS_134.34.226.241_2025-08-21__16-04-26.csv')
# timestamps = []
# with open(timestamps_file, 'r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         try:
#             # Assuming timestamps are in seconds in the first column
#             # Convert UNIX timestamp to seconds relative to the first timestamp
#             unix_time = float(row[0])
#             if not timestamps:
#                 first_unix_time = unix_time
#             timestamps.append(unix_time - first_unix_time + 0.120)
#         except ValueError:
#             continue

#%% 

# Load timestamps from .npy file
timestamps_npy_file = os.path.join(dir, 'data', '134.34.226.241', 'TIMESTAMPS_134.34.226.241_2025-08-21__16-04-26.npy')
timestamps_npy_file = os.path.join(dir, 'data', '134.34.226.238', 'TIMESTAMPS_134.34.226.238_2025-08-22__19-17-06.npy')
timestamps = np.load(timestamps_npy_file, allow_pickle=True)
# timestamps = timestamps - timestamps[0]
# Convert timestamps to seconds
ts_start_sync = timestamps[0][0] + peaks[0] / fs
print(f'Timestamps for sync: {ts_start_sync}')
#%%
# Extract timestamps where 'at = true' from the specified .npy file
camera_file = os.path.join(dir, 'data/camera', '2025-08-21_16-04-35:1623_basler_tracking_markers.npy')
camera_file = os.path.join(dir, 'data/camera', '2025-08-22_19-16-53:0606_basler_tracking_markers.npy')

camera_data = np.load(camera_file, allow_pickle=True)
# Assume each row is a marker, and the last column is a boolean
camera_ts_true = [entry["timestamp"] for entry in camera_data if entry.get("noise_on")]
camera_ts_start = camera_ts_true[0] if camera_ts_true else None
camera_ts_start = float(camera_ts_start)
print(f'Camera timestamps for sync: {camera_ts_start}')

# %%
Delta_sync = ts_start_sync - camera_ts_start 
print(f'Delta sync: {Delta_sync}')
# %%
# Overlay timestamps on the audio plot
plt.figure(figsize=(15, 5))
plt.plot(np.linspace(0, len(robat_238_audio_hp) / fs, len(robat_238_audio_hp)), robat_238_audio_hp, label='Highpass Audio')

# Overlay all timestamps as vertical lines

# %%
def extract_timestamps_from_npy(file_path):
    """
    Extracts timestamps from a .npy file.
    Returns a numpy array of timestamps in seconds.
    """
    data = np.load(file_path, allow_pickle=True)
    timestamps = [row[0] for row in data]
    return np.array(timestamps)

# %%
timestamps_extr = extract_timestamps_from_npy(timestamps_npy_file)
timestamps_extr = timestamps_extr - timestamps_extr[0]

plt.close()

plt.figure(figsize=(15, 5))
plt.plot(np.linspace(0, len(robat_238_audio_hp) / fs, len(robat_238_audio_hp)), robat_238_audio_hp, label='Highpass Audio')
plt.vlines(timestamps_extr, ymin=np.min(robat_238_audio_hp), ymax=np.max(robat_238_audio_hp), color='red', label='Timestamps')
plt.legend()
plt.title('Highpass Audio with Timestamps Overlay')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()


# %%
