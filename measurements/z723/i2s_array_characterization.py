#%%
import sounddevice as sd
import numpy as np
import scipy.signal as signal
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
# # %% 
#     try:
#         for i in range(n_sweeps): 
#             stream = sd.OutputStream(samplerate=fs,
#                         blocksize=0,
#                         device=device,
#                         channels=2,
#                         callback=callback,
#                         latency='low')
#                 
#             with stream:
#                 while stream.active:
#                     pass
# 
#             current_frame = 0
#             print('Chirped %d' % (i+1))
#             time.sleep(1)
# 
#     except KeyboardInterrupt:
#         print('Interrupted by user')

# %% Libraries and files
import os
import soundfile as sf

# Load audio files, then plot a 6x6 grid
DIR = "./array_calibration/226_238/2025-03-27/original/"  # Directory containing the audio files
audio_files = os.listdir(DIR)  # List all files in the sweeps directory
audio_files.sort()  # Sort the files in ascending order

# Directory to save the extracted channels
output_dir = "./array_calibration/226_238/2025-03-27/extracted_channels/"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Path to the multi-channel WAV file
for file in audio_files:
    file_path = os.path.join(DIR, file)

    angle_name = file.split('.')[0]
    print(f"Processing file: {angle_name}")

    # Read the multi-channel WAV file
    audio_data, sample_rate = sf.read(DIR + file)

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
# List all extracted channel files separated by channel number
from natsort import natsorted

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
def detect_peaks(filtered_output, threshold=0.8):
    peaks, _ = signal.find_peaks(filtered_output, height=threshold * np.max(filtered_output), distance=(silence_dur/1000+dur)*fs)
    return peaks

# Process each channel
DIR_first_sweep = "./array_calibration/226_238/2025-03-27/extracted_channels/first_sweep/"  # Directory to save the first sweeps

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

            # Calculate RMS value of the first sweep
            rms_first_sweep = np.sqrt(np.mean(first_sweep**2))
            print(f"RMS value of the first sweep in {file}: {rms_first_sweep:.5f}")

            # Store RMS value in the dictionary
            rms_values_dict[file] = rms_first_sweep

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

            print(f"Average RMS value of all sweeps in {file}: {average_rms:.5f}")

            sf.write(DIR_first_sweep + file, first_sweep, int(fs))
            # Plot the first sweep
            angle_name = file.split('_')[0]
            if int(angle_name):
                ax.plot(np.linspace(0,len(first_sweep),len(first_sweep))/fs, first_sweep, label=f"{angle_name}")
                ax.legend(loc = 'upper right', ncol = 2)
            if len(peaks) < n_sweeps:
                print(f"Only {len(peaks)} sweeps detected in {file} - Channel {channel_number}; expected {n_sweeps}.\n Try adjusting the threshold in detect_peaks.")
                # Plot the filtered output
                # plt.figure(figsize=(15, 5))
                # plt.title(f"Filtered Output - {file}")
                # plt.plot(np.linspace(0, len(filtered_output), len(filtered_output)) / fs, filtered_output, label=f"{file}")
                # plt.plot(peaks / fs, filtered_output[peaks], "x", label="Detected Peaks")
                # plt.xlabel("Seconds")
                # plt.ylabel("Amplitude")
                # plt.grid(True)
                # plt.legend()
        else:
            print(f"No sweeps detected in {file} - Channel {channel_number}")
        

    # Plot all angles, skipping '360'
    fig1, axs = plt.subplots(9, 4, figsize=(15, 30), sharey=True)
    angles = [file.split('_')[0] for file in files]  # Extract angle names from filenames

    idx_to_plot = 0
    for idx, file in enumerate(files):
        if angles[idx] == '360':
            continue  # Skip the 360 angle

        file_path = os.path.join(DIR_first_sweep, file)
        audio, fs = sf.read(file_path)

        rms = np.sqrt(np.mean(audio**2))
        rms_db = 20 * np.log10(rms)
        
        row = idx_to_plot // 4
        col = idx_to_plot % 4
        
        ax = axs[row, col]
        ax.plot(np.linspace(0, len(audio) / fs, len(audio)), audio)
        ax.set_title(f"Angle: {angles[idx]} degrees ")  # Use extracted angle name with units
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        ax.legend([f'RMS: {rms:.5f}\nRMS: {rms_db:.5f} dB'], loc='upper left')

        idx_to_plot += 1

    plt.suptitle(f"Channel {channel_number}: First Sweep for Each Angle", fontsize=20)
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
fig_polar, axs_polar = plt.subplots(1, num_channels, figsize=(18, 5), subplot_kw={'projection': 'polar'})
fig_polar.suptitle("RMS Values of Overall Recording for Each Channel", fontsize=16)

for i in range(num_channels):
    channel_number = i + 1
    files = grouped_files[channel_number]
    
    rms_values = []
    rms_values_norm_db = []
    angles = []
    
    for file in files:
        file_path = os.path.join(extracted_channels_dir, file)
        rms = rms_values_dict[file]
        rms_values.append(rms)

        rms_values_norm = rms_values / rms_values[0]
        rms_values_norm_db = 20 * np.log10(rms_values_norm)

        angle_name = file.split('_')[0]
        angles.append(int(angle_name))
    
    # Convert angles to radians
    angles_rad = np.radians(angles)
    
    # Plot RMS values in polar plot
    ax_polar = axs_polar[i] if num_channels > 1 else axs_polar
    ax_polar.plot(angles_rad, rms_values_norm_db, linestyle='-', label=f"Channel {channel_number}")
    ax_polar.set_title(f"Channel {channel_number}")
    ax_polar.set_theta_zero_location("N")  # Set 0 degrees to North
    ax_polar.set_theta_direction(-1)  # Set clockwise direction
    ax_polar.set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))  # Set angle ticks
    ax_polar.set_xlabel("Angle (degrees)")
    ax_polar.set_ylabel("RMS Value dB", position=(0, 1), ha='left')
    ax_polar.set_rlabel_position(0)

# Linear plot of all channels
fig_linear, ax_linear = plt.subplots(figsize=(10, 6))
fig_linear.suptitle("RMS Values of Overall Recording for All Channels", fontsize=16)

for i in range(num_channels):
    channel_number = i + 1
    files = grouped_files[channel_number]
    
    rms_values = []
    angles = []
    
    for file in files:
        file_path = os.path.join(extracted_channels_dir, file)
        rms = rms_values_dict[file]
        rms_values.append(rms)

        angle_name = file.split('_')[0]
        angles.append(int(angle_name))
    
    # Plot RMS values in linear plot
    ax_linear.plot(angles, rms_values, marker='.', linestyle='-', label=f"Channel {channel_number}")

ax_linear.set_xlabel("Angle (degrees)")
ax_linear.set_xticks(np.linspace(0, 380, 19, endpoint=False))  # Set angle ticks
ax_linear.set_ylabel("RMS Value")
ax_linear.legend()
ax_linear.grid(True)

fig_polar.tight_layout()
plt.show(block = False)

# %%

import soundfile as sf
from scipy import fft

# Directory containing the extracted channels

# Central frequencies of the bands
central_freq = np.array([4e3, 6e3, 8e3, 10e3, 12e3, 14e3, 16e3, 18e3, 20e3, 22e3, 24e3, 26e3, 28e3, 30e3, 32e3, 34e3, 36e3, 38e3])
BW = 1e3  # Bandwidth of the bands
linestyles = ["-", "--", "-.", ":", "-", "--"]  # Line styles for the plot

# Group central frequencies into 3 sets of 6 bands each
num_bands_per_plot = 6
central_freq_sets = [central_freq[i * num_bands_per_plot:(i + 1) * num_bands_per_plot] for i in range(3)]

# Number of microphones
num_mics = num_channels

# Plot for each microphone
for mic in range(1, num_mics + 1):
    files = grouped_files[mic]
    angles = [int(file.split('_')[0]) for file in files]  # Extract angles from filenames

    # Create a figure with 3 polar subplots
    fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "polar"}, figsize=(15, 5))
    plt.suptitle(f"Polar Frequency Response - Microphone {mic}", fontsize=20)

    for ax_idx, ax in enumerate(axes):
        ii = 0
        for fc in central_freq_sets[ax_idx]:
            
            audio_patt = []

            for file in files:
                file_path = os.path.join(DIR_first_sweep, file)
                audio, fs = sf.read(file_path)

                # Compute FFT
                audio_freq = fft.fft(audio, n=2048)
                audio_freq = audio_freq[:1024]
                freqs = fft.fftfreq(2048, 1 / fs)[:1024]

                # Compute mean radiance in the band
                band_mean = np.mean(np.abs(audio_freq[(freqs > fc - BW) & (freqs < fc + BW)]))
                audio_patt.append(band_mean)

            # Normalize and plot
            audio_patt_norm = audio_patt / audio_patt[0] # Normalize the radiance
            audio_patt_norm_dB = 20 * np.log10(audio_patt_norm) # Convert the radiance to dB
            
            if fc >= 10e3:
                label = f"{fc / 1e3:.0f} kHz"
            else:
                label = f"{fc / 1e3:.0f} kHz"

            ax.plot(np.deg2rad(angles), audio_patt_norm_dB, label=label, linestyle=linestyles[ii])
            ii +=1
        # Configure polar plot
        ax.legend(loc="upper right")
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_zero_location("N")  # Set 0 degrees to North
        ax.set_theta_direction(-1)  # Set clockwise direction
        ax.set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))  # Set angle ticks
        ax.set_yticks(np.linspace(-35, 0, 6))
        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel("RMS Value dB", position=(0, 1), ha='left')
        ax.set_rlabel_position(0)


    plt.tight_layout()

    plt.show()

# %%
