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
        plt.show()
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

# Directory to save the extracted channels
output_dir = DIR + "/extracted_channels/"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

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

# Save each channel as a separate WAV file
for i, channel_data in enumerate(channels):
    output_file = os.path.join(output_dir, angle_name+f"_{i + 1}.wav")  # Path to the output file
    sf.write(output_file, channel_data, sample_rate)
    print(f"Saved channel {i + 1} to {output_file}")
#%%

#  List all extracted channel files separated by channel number
import os
from natsort import natsorted
import soundfile as sf

# Directory containing the extracted channels
extracted_channels_dir = "./array_calibration/226_238/2025-03-27/extracted_channels/"

# List all extracted channel file
channel_files = os.listdir(extracted_channels_dir)

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
channel_number = 1
for i in range(len(grouped_files)):
    files = grouped_files[i+1]
    print(f"Processing Channel {channel_number}:")
    
    # Create a new figure for each channel
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_title(f"Channel {channel_number}")
    ax.set_xlabel("Samples")
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

            # Plot the first sweep
            angle_name = file.split('_')[0]
            ax.plot(first_sweep, label=f"{angle_name}")
            
        else:
            print(f"No sweeps detected in {file} - Channel {channel_number}")
    channel_number += 1
    ax.legend()

    plt.tight_layout()
    plt.show()
# %%

# Load audio files, then plot them in a 6x6 grid
DIR = "./cut_sweeps/"  # Directory containing the audio files
audio_files = os.listdir(DIR)  # List all files in the sweeps directory
audio_files.sort()  # Sort the files in ascending order

# Load audio files, then plot them in a 6x6 grid
DIR_noise = "./noise_floor_5ms/"  # Directory containing the audio files
noise_files = os.listdir(DIR_noise)  # List all files in the sweeps directory
noise_files.sort()  # Sort the files in ascending order

# %% Plot of the collected data

fig, axs = plt.subplots(6, 6, figsize=(20, 20))

for i in range(6):
    for j in range(6):
        # Load audio file
        audio, fs = soundfile.read(DIR + audio_files[i * 6 + j])
        # Plot audio file
        axs[i, j].plot(np.linspace(0, len(audio) / fs, len(audio)), audio)
        axs[i, j].set_title(audio_files[i * 6 + j])
        axs[i, j].set_xlabel("Time (s)")
        axs[i, j].set_ylabel("Amplitude")
        # Shared x and y axes
        axs[i, j].sharex(axs[0, 0])
        axs[i, j].sharey(axs[0, 0])

plt.suptitle("Recorded 5ms sweeps - CE32A-4 1/4\" Mini Speaker", fontsize=30)
plt.tight_layout()
plt.show(block=False)
