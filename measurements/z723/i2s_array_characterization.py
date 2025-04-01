#%%
import sounddevice as sd
import numpy as np
import scipy.signal as signal
import time
from matplotlib import pyplot as plt

def get_soundcard_outstream(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return i

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
    dur = 5e-3
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

    device = get_soundcard_outstream(sd.query_devices())
# %%
    try:
        for i in range(n_sweeps): 
            stream = sd.OutputStream(samplerate=fs,
                        blocksize=0,
                        device=device,
                        channels=2,
                        callback=callback,
                        latency='low')
                
            with stream:
                while stream.active:
                    pass

            current_frame = 0
            print('Chirped %d' % (i+1))
            time.sleep(1)

    except KeyboardInterrupt:
        print('Interrupted by user')

# %% Libraries and files

import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft

# Load audio files, then plot them in a 6x6 grid
DIR = "./array_calibration/226_238/2025-03-27/original/"  # Directory containing the audio files
audio_files = os.listdir(DIR)  # List all files in the sweeps directory
audio_files.sort()  # Sort the files in ascending order

# Path to the multi-channel WAV file
angle_name = '350'
filename = angle_name +'.wav'

# Read the multi-channel WAV file
audio_data, sample_rate = sf.read(DIR + filename)

# Check the shape of the audio data
print(f"Audio data shape: {audio_data.shape}")  # (samples, channels)

# Extract individual channels
num_channels = audio_data.shape[1]  # Number of channels
channels = [audio_data[:, i] for i in range(num_channels)]

#%%

#  List all extracted channel files separated by channel number
from natsort import natsorted
import cmath
import os

# Directory containing the extracted channels
extracted_channels_dir = "./array_calibration/226_238/2025-03-27/extracted_channels/"

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
    filtered_output = signal.fftconvolve(recording, chirp_template, mode='valid')
    return filtered_output

# Detect peaks in the matched filter output
def detect_peaks(filtered_output, threshold=0.5):
    peaks, _ = signal.find_peaks(filtered_output, height=threshold * np.max(filtered_output))
    return peaks

# Process each channel
DIR_first_sweep = "./array_calibration/226_238/2025-03-27/extracted_channels/first_sweep/"  # Directory to save the first sweeps
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

    for file in files:
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
            angle_name = file.split('_')[0]
            if int(angle_name):
                ax.plot(np.linspace(0,len(first_sweep),len(first_sweep))/fs, first_sweep, label=f"{angle_name}")
            
        else:
            print(f"No sweeps detected in {file} - Channel {channel_number}")
        

    # Plot all angles 
    fig1, axs = plt.subplots(4, 5, figsize=(20, 15), sharey=True)
    angles = [file.split('_')[0] for file in files]  # Extract angle names from filenames

    for idx, file in enumerate(files):
        file_path = os.path.join(DIR_first_sweep, file)
        audio, fs = sf.read(file_path)
        
        row = idx // 5
        col = idx % 5
        
        ax = axs[row, col]
        ax.plot(np.linspace(0, len(audio) / fs, len(audio)), audio)
        ax.set_title(f"Angle: {angles[idx]} degrees")  # Use extracted angle name with units
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)

    plt.suptitle(f"Channel {channel_number}: First Sweep for Each Angle", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
    plt.show()

    ax.legend()
    channel_number += 1

plt.show()
    
 #%%
# RMS values of the first sweep for each channel

num_channels = len(grouped_files)
fig_polar, axs_polar = plt.subplots(1, num_channels, figsize=(18, 5), subplot_kw={'projection': 'polar'})
fig_polar.suptitle("RMS Values of First Sweeps for Each Channel", fontsize=16)

for i in range(num_channels):
    channel_number = i + 1
    files = grouped_files[channel_number]
    
    rms_values = []
    angles = []
    
    for file in files:
        file_path = os.path.join(DIR_first_sweep, file)
        audio, fs = sf.read(file_path)
        
        rms = np.sqrt(np.mean(audio**2))
        rms_values.append(rms)
        
        angle_name = file.split('_')[0]
        angles.append(int(angle_name))
    
    # Convert angles to radians
    angles_rad = np.radians(angles)
    
    # Plot RMS values in polar plot
    max_rms = max(max(rms_values) for channel_number in grouped_files for file in grouped_files[channel_number]
                  for rms_values in [[np.sqrt(np.mean(sf.read(os.path.join(DIR_first_sweep, file))[0]**2)) for file in grouped_files[channel_number]]])

    # Plot RMS values in polar plot
    ax_polar = axs_polar[i] if num_channels > 1 else axs_polar
    ax_polar.plot(angles_rad, rms_values, linestyle='-', label=f"Channel {channel_number}")
    ax_polar.set_title(f"Channel {channel_number}")
    ax_polar.set_theta_zero_location("N")  # Set 0 degrees to North
    ax_polar.set_theta_direction(-1)  # Set clockwise direction
    ax_polar.set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))  # Set angle ticks
    ax_polar.set_xlabel("Angle (degrees)")
    ax_polar.set_ylabel("RMS Value", position=(0, 1), ha='left')
    ax_polar.set_ylim(0, max_rms * 1.1)  # Set common y-axis limits
    ax_polar.set_rlabel_position(0)


fig_polar.tight_layout()
plt.show()
# %%
# RMS values of the overall recording for each channel and each angle

num_channels = len(grouped_files)
fig_polar, axs_polar = plt.subplots(1, num_channels, figsize=(18, 5), subplot_kw={'projection': 'polar'})
fig_polar.suptitle("RMS Values of Overall Recording for Each Channel", fontsize=16)

for i in range(num_channels):
    channel_number = i + 1
    files = grouped_files[channel_number]
    
    rms_values = []
    angles = []
    
    for file in files:
        file_path = os.path.join(extracted_channels_dir, file)
        audio, fs = sf.read(file_path)
        
        rms = np.sqrt(np.mean(audio**2))
        rms_values.append(rms)
        
        angle_name = file.split('_')[0]
        angles.append(int(angle_name))
    
    # Convert angles to radians
    angles_rad = np.radians(angles)
    
    # Plot RMS values in polar plot
    max_rms = max(max(rms_values) for channel_number in grouped_files for file in grouped_files[channel_number]
                  for rms_values in [[np.sqrt(np.mean(sf.read(os.path.join(extracted_channels_dir, file))[0]**2)) for file in grouped_files[channel_number]]])

    # Plot RMS values in polar plot
    ax_polar = axs_polar[i] if num_channels > 1 else axs_polar
    ax_polar.plot(angles_rad, rms_values, linestyle='-', label=f"Channel {channel_number}")
    ax_polar.set_title(f"Channel {channel_number}")
    ax_polar.set_theta_zero_location("N")  # Set 0 degrees to North
    ax_polar.set_theta_direction(-1)  # Set clockwise direction
    ax_polar.set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))  # Set angle ticks
    ax_polar.set_xlabel("Angle (degrees)")
    ax_polar.set_ylabel("RMS Value", position=(0, 1), ha='left')
    ax_polar.set_ylim(0, max_rms * 1.1)  # Set common y-axis limits
    ax_polar.set_rlabel_position(0)


fig_polar.tight_layout()
plt.show()


# %%
