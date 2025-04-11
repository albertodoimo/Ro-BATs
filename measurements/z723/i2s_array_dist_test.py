# %%
# 
import sounddevice as sd
import numpy as np
import scipy.signal as signal
from matplotlib import pyplot as plt
import os
import soundfile as sf

def pow_two_pad_and_window(vec, show = True):
    window = signal.windows.tukey(len(vec), alpha=0.2)
    windowed_vec = vec * window
    padded_windowed_vec = np.pad(windowed_vec, (0, 2**int(np.ceil(np.log2(len(windowed_vec)))) - len(windowed_vec)))
    if show:
        dur = len(padded_windowed_vec) / fs
        t = np.linspace(0, dur, len(windowed_vec))
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(t, windowed_vec)
        plt.subplot(2, 1, 2)
        plt.specgram(windowed_vec, NFFT=64, noverlap=32, Fs=fs)
        # Removed redundant plt.show() call
    return padded_windowed_vec/max(padded_windowed_vec)

def pow_two(vec):
    return np.pad(vec, (0, 2**int(np.ceil(np.log2(len(vec)))) - len(vec)))

if __name__ == "__main__":

    fs = 96e3
    dur = 20e-3
    hi_freq =  1e3
    low_freq = 40e3
    n_sweeps = 5
    
    t_tone = np.linspace(0, dur, int(fs*dur))
    chirp = signal.chirp(t_tone, hi_freq, t_tone[-1], low_freq)
    sig = pow_two_pad_and_window(chirp, show=True)

    silence_dur = 100 # [ms]
    silence_samples = int(silence_dur * fs/1000)
    silence_vec = np.zeros((silence_samples, ))
    full_sig = pow_two(np.concatenate((sig, silence_vec)))
    print('len = ', len(full_sig))
    stereo_sig = np.hstack([full_sig.reshape(-1, 1), full_sig.reshape(-1, 1)])

    output_sig = np.float32(stereo_sig)

    current_frame = 0
    def callback(outdata, frames, time, status):
        global current_frame
        if status:
            print(status)
        chunksize = min(len(output_sig) - current_frame, frames)
        outdata[:chunksize] = output_sig[current_frame:current_frame + chunksize]
        if chunksize < frames:
            outdata[chunksize:] = 0
            raise sd.CallbackAbort()
        current_frame += chunksize

# %% Libraries and files


# Load audio files, then plot a 6x6 grid
DIR = "./array_calibration/226_238/dist_test/2025-04-09/original/"  # Directory containing the audio files
audio_files = os.listdir(DIR)  # List all files in the sweeps directory
audio_files.sort()  # Sort the files in ascending order

# Directory to save the extracted channels
output_dir = "./array_calibration/226_238/dist_test/2025-04-09/extracted_channels/"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Path to the multi-channel WAV file
for file in audio_files:
    file_path = os.path.join(DIR, file)

    dist_name = file.split('.')[0]
    print(f"Processing file: {dist_name}")

    # Read the multi-channel WAV file
    audio_data, sample_rate = sf.read(DIR + file)

    # Check the shape of the audio data
    print(f"Audio data shape: {audio_data.shape}")  # (samples, channels)

    # Extract individual channels
    num_channels = audio_data.shape[1]  # Number of channels
    channels = [audio_data[:, i] for i in range(num_channels)]

    # Save each channel as a separate WAV file
    for i, channel_data in enumerate(channels):
        output_file = os.path.join(output_dir, dist_name+f"_{i + 1}.wav")  # Path to the output file
        sf.write(output_file, channel_data, sample_rate)
        print(f"Saved channel {i + 1} to {output_file}")

#%%

# List all extracted channel files separated by channel number
from natsort import natsorted
import cmath
import os

# Directory containing the extracted channels
extracted_channels_dir = "./array_calibration/226_238/dist_test/2025-04-09/extracted_channels"
os.makedirs(extracted_channels_dir, exist_ok=True)  # Create the directory if it doesn't exist

# List all extracted channel files
channel_files = os.listdir(extracted_channels_dir)

# Filter out directories, keep only files
channel_files = [f for f in channel_files if os.path.isfile(os.path.join(extracted_channels_dir, f))]

# Sort the files naturally by the last part of their names (e.g., channel number)
sorted_channel_files = natsorted(channel_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

# Group files by the last part of their name (channel number)
grouped_files = {}

for file in sorted_channel_files:
    # Extract the channel number from the file name (e.g., "350_1.wav" -> "1")
    channel_number = int(file.split('_')[-1].split('.')[0])

    # Group files by channel number
    if channel_number not in grouped_files:
        grouped_files[channel_number] = []
    grouped_files[channel_number].append(file)

for i in range(len(grouped_files)):
    grouped_files[i+1].sort()

# Print grouped files
for channel_number, files in grouped_files.items():
    print(f"Channel {channel_number}:")
    for f in files:
        print(f"  {f}")


# %%

# Define the matched filter function
def matched_filter(recording, chirp_template):
    chirp_template = chirp_template[::-1]  # Time-reversed chirp
    filtered_output = signal.fftconvolve(recording, chirp_template, mode='same')
    return filtered_output

# Detect peaks in the matched filter output
def detect_peaks(filtered_output, threshold=0.5):
    peaks, _ = signal.find_peaks(filtered_output, height=threshold * np.max(filtered_output))
    return peaks

# Process each channel
DIR_first_sweep = extracted_channels_dir + "/first_sweep/"  # Directory to save the first sweeps
os.makedirs(DIR_first_sweep, exist_ok=True)  # Create the directory if it doesn't exist

channel_number = 1
for i in range(len(grouped_files)):
    files = grouped_files[i+1]
    print(f"Processing Channel {channel_number}:")
    
    # Create a new figure for each channel
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_title(f"Channel {channel_number}")
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

    for file in files[0:(len(files))]:
        file_path = os.path.join(extracted_channels_dir, file)
        recording, sample_rate = sf.read(file_path)

        # Apply matched filtering
        filtered_output = matched_filter(recording, chirp)

        # Detect peaks
        peaks = detect_peaks(filtered_output)

        if len(peaks) > 0:
            # Extract the first sweep
            first_sweep_start = peaks[0]
            first_sweep_end = first_sweep_start + len(chirp)
            first_sweep = recording[first_sweep_start:first_sweep_end]

            sf.write(DIR_first_sweep + file, first_sweep, int(fs))
            # Plot the first sweep
            dist_name = file.split('_')[0]
            ax.plot(np.linspace(0,len(first_sweep),len(first_sweep))/fs, first_sweep, label=f"{dist_name}")
            
        else:
            print(f"No sweeps detected in {file} - Channel {channel_number}")
        

    # Plot all angles, skipping '360'
    fig1, axs = plt.subplots(2, 6, figsize=(18, 8), sharey=True)
    dist = [file.split('_')[0] for file in files]  

    idx_to_plot = 0
    for idx, file in enumerate(files):

        file_path = os.path.join(DIR_first_sweep, file)
        audio, fs = sf.read(file_path)

        rms = np.sqrt(np.mean(audio**2))
        rms_db = 20 * np.log10(rms)

        row = idx_to_plot // 6
        col = idx_to_plot % 6

        ax = axs[row, col]
        ax.plot(np.linspace(0, len(audio) / fs, len(audio)), audio)
        ax.set_title(f"dist: {dist[idx]} ")  
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        ax.legend([f'RMS: {rms:.5f}\nRMS: {rms_db:.5f} dB'], loc='upper left')

        idx_to_plot += 1

    plt.suptitle(f"Channel {channel_number}: First Sweep for Each dist", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
    plt.show(block = False)

    ax.legend()
    channel_number += 1

plt.show(block = False)
    
 #%%
# RMS values of the first sweep for each channel

num_channels = len(grouped_files)
fig, axs = plt.subplots(1, num_channels, figsize=(18, 10))
fig.suptitle("RMS Values of First Sweeps for Each Channel", fontsize=16)

for i in range(num_channels):
    channel_number = i + 1
    files = grouped_files[channel_number]
    
    rms_values = []
    rms_values_norm_db = []
    dist = []
    
    rms_values = []
    rms_values_norm_db = []
    dist = []
    
    # Extract distance from filename and store with the filename
    dist_files = []
    for file in files:
        if 'C' in file:
            continue
        file_path = os.path.join(DIR_first_sweep, file)
        audio, fs = sf.read(file_path)
        
        rms = np.sqrt(np.mean(audio**2))

        rms_values.append(rms)
        
        dist_name = file.split('c')[0]
        dist_val = int(dist_name.split('_')[0])
        dist_files.append((dist_val, file))
    
    # Sort files based on distance
    dist_files.sort(key=lambda x: x[0])
    
    # Extract sorted filenames and distances
    sorted_files = [file for _, file in dist_files]
    dist = [str(dist_val) for dist_val, _ in dist_files]
    rms_values = []

    for file in sorted_files:
        file_path = os.path.join(DIR_first_sweep, file)
        audio, fs = sf.read(file_path)
        rms = np.sqrt(np.mean(audio**2))
        rms_values.append(rms)
    
    rms_values_norm = rms_values / rms_values[-1]
    rms_values_norm_db = 20 * np.log10(rms_values_norm)
    # Plot RMS values for the channels over distances
    ax = axs[i]
    ax.plot(dist, rms_values_norm_db, marker='o')
    ax.set_title(f"Channel {channel_number}")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Normalized RMS (dB)")
    ax.set_ylim(0, 10)
    ax.set_yticks(np.arange(0, 15, 1))
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#%%
# Plot RMS values across channels for each distance
distances = []
for i in range(num_channels):
    channel_number = i + 1
    files = grouped_files[channel_number]
    
    # Extract distance from filename
    for file in files:
        if 'C' in file:
            continue
        dist_name = file.split('c')[0]
        dist_val = int(dist_name.split('_')[0])
        if dist_val not in distances:
            distances.append(dist_val)

distances.sort()

rms_values_across_channels = {}
for dist in distances:
    rms_values_across_channels[dist] = []

for i in range(num_channels):
    channel_number = i + 1
    files = grouped_files[channel_number]
    
    rms_values = []
    dist_files = []
    
    # Extract distance from filename and store with the filename
    for file in files:
        if 'C' in file:
            continue
        file_path = os.path.join(DIR_first_sweep, file)
        audio, fs = sf.read(file_path)
        rms = np.sqrt(np.mean(audio**2))
        
        dist_name = file.split('c')[0]
        dist_val = int(dist_name.split('_')[0])
        
        rms_values_across_channels[dist_val].append(rms)

# Normalize RMS values for each distance
for dist in distances:
    max_rms = max(rms_values_across_channels[dist])
    rms_values_across_channels[dist] = [20 * np.log10(rms / max_rms) for rms in rms_values_across_channels[dist]]

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
channel_numbers = list(range(1, num_channels + 1))

for dist in distances:
    ax.plot(channel_numbers, rms_values_across_channels[dist], marker='o', label=f'Distance: {dist}')

ax.set_xlabel("Channel Number")
ax.set_ylabel("Normalized RMS (dB)")
ax.set_title("RMS Values Across Channels for Each Distance")
ax.set_xticks(channel_numbers)
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()


# %%
# Plot the RMS values for each channel
