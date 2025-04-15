#%%
# 
# import sounddevice as sd
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
    return padded_windowed_vec/max(padded_windowed_vec)

def pow_two(vec):
    return np.pad(vec, (0, 2**int(np.ceil(np.log2(len(vec)))) - len(vec)))

if __name__ == "__main__":

    fs = 96e3
    dur = 20e-3
    hi_freq =  1e3
    low_freq = 40e3
    
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
# Load audio files

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
    grouped_files[i+1].sort(key=lambda x: int(x.split('c')[0]))

# Print grouped files
for channel_number, files in grouped_files.items():
    print(f"Channel {channel_number}:")
    for f in files:
        print(f"  {f}")


# %%

# Define the matched filter function
def matched_filter(recording, chirp_template):
    chirp_template = chirp_template[::-1]  # Time-reversed chirp
    filtered_output = signal.fftconvolve(recording, chirp_template, mode='valid')
    return filtered_output

# Detect peaks in the matched filter output
def detect_peaks(filtered_output, threshold=0.8):
    peaks, _ = signal.find_peaks(filtered_output, height=threshold * np.max(filtered_output))
    return peaks

# Process each channel
DIR_first_sweep = extracted_channels_dir + "/first_sweep/"  # Directory to save the first sweeps
os.makedirs(DIR_first_sweep, exist_ok=True)  # Create the directory if it doesn't exist

# Dictionary to store RMS values for all files
rms_values_dict = {}

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
        print(f"Peaks detected in {file}: {len(peaks)}")
        
        if len(peaks) > 0:
            # Extract the first sweep
            first_sweep_start = peaks[0]
            first_sweep_end = first_sweep_start + len(chirp)
            first_sweep = recording[first_sweep_start:first_sweep_end]

            # Calculate RMS values for all detected peaks
            rms_values = []
            for peak in peaks:
                sweep_start = peak
                sweep_end = sweep_start + len(chirp)
                sweep = recording[sweep_start:sweep_end]
                rms = np.sqrt(np.mean(sweep**2))
                rms_values.append(rms)
                print(f"RMS value of sweep at peak {peak} in {file}: {rms:.5f}")
            
            # Calculate the average RMS value of all peaks
            average_rms = np.mean(rms_values)
            # Store mean RMS value in the dictionary
            rms_values_dict[file] = average_rms

            print(f"Average RMS value of all sweeps in {file}: {average_rms:.5f}")

            sf.write(DIR_first_sweep + file, first_sweep, int(fs))
            # Plot the first sweep
            dist_name = file.split('_')[0]

            ax.plot(np.linspace(0,len(first_sweep),len(first_sweep))/fs, first_sweep, label=f"{dist_name}")
            ax.legend(loc = 'upper right', ncol = 2)
            print(f" {len(peaks)} sweeps detected in {file} - Channel {channel_number}")

        else:
            print(f"No sweeps detected in {file} - Channel {channel_number}")
        

    # Plot all dist
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

# Print the dictionary of RMS values
print("\nRMS Values for All Files:")
for file, rms_value in rms_values_dict.items():
    print(f"{file}: {rms_value:.5f}")

 # %%
# RMS values of the overall recording for each channel and each angle

num_channels = len(grouped_files)

# Linear plot of all channels
fig_linear, ax_linear = plt.subplots(figsize=(6, 7))
fig_linear.suptitle("RMS Values of Overall Recording for All Channels", fontsize=12)

# Subplots for each channel
fig_channels, axs_channels = plt.subplots(1, num_channels, figsize=(5*num_channels,10))
fig_channels.suptitle("RMS Values of Overall Recording for Each Channel", fontsize=16)

# Plotting RMS values for each channel
angles = np.linspace(0, 360, 18, endpoint=False)  # Angles for each measurement
rms_values_channel = {}

# Collect RMS values for each channel

channel_data = {}
norm_channel_data = {}

for file, rms_value in rms_values_dict.items():
    if 'C' in file:
        continue

    dist_name = file.split('c')[0]
    channel_number = int(file.split('_')[-1].split('.')[0])

    if channel_number not in channel_data:
        channel_data[channel_number] = {'distances': [], 'rms_values': []}

    if channel_number not in norm_channel_data:
        norm_channel_data[channel_number] = {'distances': [], 'rms_values': []}

    channel_data[channel_number]['distances'].append(dist_name)
    norm_channel_data[channel_number]['distances'].append(dist_name)

    if len(channel_data[channel_number]['distances']) == 0:
        pass#channel_data[channel_number]['rms_values'].append(20 * np.log10(rms_value/max_rms))
    else:
        first_distance = norm_channel_data[channel_number]['distances'][0]
        first_rms = rms_values_dict[first_distance+'cm_'+ str(channel_number)+'.wav']
        print(f"first rms: {first_rms}")
        norm_channel_data[channel_number]['rms_values'].append(20 * np.log10(rms_value/first_rms))

        channel_data[channel_number]['rms_values'].append(20 * np.log10(rms_value))


for channel_number, data in norm_channel_data.items():
    ax_linear.plot(data['distances'], data['rms_values'], marker='o', linestyle='-', label=f'Channel {channel_number}')
    print(f"Channel {channel_number}: {data['rms_values']}")
    ax_linear.tick_params(axis='x', rotation=45)
    ax_linear.grid(True)
    ax_linear.legend()
    
    # Plotting each channel in a separate subplot
    color = ax_linear.get_lines()[-1].get_color()  # Get the color of the last line in ax_linear
    ax_channel = axs_channels[channel_number-1]
    ax_channel.plot(data['distances'], data['rms_values'], marker='o', linestyle='-', label=f'Channel {channel_number}', color=color)
    ax_channel.set_title(f'Channel {channel_number}')
    ax_channel.set_xlabel('Distance')
    if channel_number == 1:
        ax_channel.set_ylabel('Normalized RMS (dB)')
        ax_channel.yaxis.set_label_position("left")
    ax_channel.grid(True)
    ax_channel.sharey(axs_channels[0])

# Plot RMS values across channels for each distance
fig_distances, ax_distances = plt.subplots(figsize=(10, 6))
fig_distances.suptitle("RMS Values Across Channels for Each Distance", fontsize=16)

distance_data = {}

for channel_number, data in channel_data.items():
    for dist, rms in zip(data['distances'], data['rms_values']):
        if dist not in distance_data:
            distance_data[dist] = []
        distance_data[dist].append(rms)

for dist, rms_values in distance_data.items():
    ax_distances.plot(range(1, len(rms_values) + 1), rms_values, marker='o', linestyle='-', label=f'Distance {dist}')
    print(f"Distance {dist}: {rms_values}")

ax_distances.set_xlabel('Channel Number')
ax_distances.set_xticks(range(1, len(rms_values) + 1))
ax_distances.set_ylabel('Normalized RMS (dB)')
ax_distances.grid(True)
ax_distances.legend(title="Distances", loc='upper right')

plt.grid(True)
plt.show(block=False)

fig_channels.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show(block=False)

# %%
