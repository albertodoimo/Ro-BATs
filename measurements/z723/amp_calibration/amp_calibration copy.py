# %%
import librosa
import numpy as np
import os
import soundfile as sf
import librosa.display
import matplotlib.pyplot as plt
import scipy.signal as signal

# %%
# Load audio files, then plot a 6x6 grid
DIR = "./2025-04-08/original/"  # Directory containing the audio files
audio_files = os.listdir(DIR)  # List all files in the sweeps directory
audio_files.sort()  # Sort the files in ascending order

#%% 

# Define the matched filter function
def matched_filter(recording, chirp_template):
    chirp_template = chirp_template[::-1]  # Time-reversed chirp
    filtered_output = signal.fftconvolve(recording, chirp_template, mode='valid')
    return filtered_output

# Detect peaks in the matched filter output
def detect_peaks(filtered_output, threshold=0.5):
    peaks, _ = signal.find_peaks(filtered_output, height=threshold * np.max(filtered_output))
    return peaks

def compute_rms(audio_signal):
    """Compute the RMS of the audio signal."""
    return np.sqrt(np.mean(audio_signal**2))

def rms_to_db(rms_value):
    """Convert RMS value to decibels."""
    return 20 * np.log10(rms_value) if rms_value > 0 else -np.inf

DIR_first_sweep = "./2025-04-08/second_sweep/"  # Directory to save the first sweeps
os.makedirs(DIR_first_sweep, exist_ok=True)  # Create the directory if it doesn't exist

durns = np.array([3, 5, 2000] )*1e-3
fs = 192000 # Hz
t = np.linspace(0, durns[1], int(fs*durns[1]))
start_f, end_f = 1e3, 95e3
sweep = signal.chirp(t, start_f, t[-1], end_f)
sweep *= signal.windows.tukey(sweep.size, 0.95)
sweep *= 0.8
sweep_padded = np.pad(sweep, pad_width=[int(fs*0.1)]*2, constant_values=[0,0])


dur = len(sweep) / fs
t = np.linspace(0, dur, len(sweep))
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, sweep)
plt.subplot(2, 1, 2)
plt.specgram(sweep, NFFT=64, noverlap=32, Fs=fs)

rms_values = []
db_values = []
for file in audio_files:
    file_path = os.path.join(DIR, file)

    date = file.split('_')[1]
    date = date.split('g')[1]

    hour = file.split('_')[2]
    hour = hour.split('.')[0]
    hour = hour.split('-')[0]
    minute = file.split('_')[2]
    minute = minute.split('.')[0]
    minute = minute.split('-')[1]
    print(f"Extracted Hour: {hour}, Min: {minute}")

    file_time = file.split('.')[0].split('_')[2]

    # Create a new figure for each channel
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_title(file.split('_')[1] + f'\n Hour: {hour}, Minute: {minute}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('RMS Value')
    ax.grid(True)

    recording, sample_rate = sf.read(DIR +file)

    # Apply matched filtering
    filtered_output = matched_filter(recording, sweep)

    # Detect peaks
    peaks = detect_peaks(filtered_output)

    if len(peaks) > 0:
        # Extract the first sweep
        first_sweep_start = peaks[0]
        first_sweep_end = first_sweep_start + len(sweep)
        first_sweep = recording[first_sweep_start:first_sweep_end]

        rms_value = compute_rms(first_sweep)

        rms_values.append([rms_value, hour, minute])
        #print(rms_values)
        db_value = rms_to_db(rms_value)
        db_values.append([db_value, hour, minute])

        sf.write(DIR_first_sweep + file_time + '.wav', first_sweep, int(fs))
        # Plot the first sweep
        ax.plot(np.linspace(0,len(first_sweep),len(first_sweep))/fs, first_sweep, label=file_time)

print(f"RMS Values: {rms_values}")
print(f"dB Values: {db_values}")

plt.show(block = False)
    

#%%

# plot rms values and db values
fig, axs = plt.subplots(2, 1, figsize=(15, 15))
ax = axs[0]
ax.set_title('RMS Values')
ax.set_xlabel('Time (s)')
ax.set_ylabel('RMS Value')
ax.grid(True)
time = [f"{val[1]}:{val[2]}" for val in rms_values]
time_in_minutes = [int(val[1]) + int(val[2])/60 for val in rms_values]
ax.plot(time_in_minutes, [val[0] for val in rms_values], label='RMS Values', marker='o')
ax.legend()
ax.set_xticks(time_in_minutes)
ax.set_xticklabels(time)
# plot db values
ax = axs[1]
ax.set_title('DB Values')
ax.set_xlabel('Time (s)')
ax.set_ylabel('DB Value')
ax.grid(True)
ax.plot(time_in_minutes, [val[0] for val in db_values], label='DB Values', marker='o')
ax.legend()
ax.set_xticks(time_in_minutes)
ax.set_xticklabels(time)
ax.set_yticks(np.arange(np.floor(min([val[0] for val in db_values])*10)/10, np.ceil(max([val[0] for val in db_values])*10)/10, 0.2))
plt.suptitle(date)

plt.show(block = False)
# %%
