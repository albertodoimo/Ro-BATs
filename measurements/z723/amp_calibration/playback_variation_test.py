# %%
import numpy as np
import os
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.signal as signal

# %%
# Load audio files
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

DIR_first_sweep = "./2025-04-08/first_sweep/"  # Directory to save the first sweeps
DIR_second_sweep = "./2025-04-08/second_sweep/"  # Directory to save the first sweeps
os.makedirs(DIR_first_sweep, exist_ok=True)  # Create the directory if it doesn't exist
os.makedirs(DIR_second_sweep, exist_ok=True)  # Create the directory if it doesn't exist

durns = np.array([3, 5, 2000] )*1e-3

fs = 192000 # Hz
t1 = np.linspace(0, durns[0], int(fs*durns[0]))
start_f1, end_f1 = 1e3, 95e3
sweep1 = signal.chirp(t1, start_f1, t1[-1], end_f1)
sweep1 *= signal.windows.tukey(sweep1.size, 0.95)
sweep1 *= 0.8
sweep1_padded = np.pad(sweep1, pad_width=[int(fs*0.1)]*2, constant_values=[0,0])

t2 = np.linspace(0, durns[1], int(fs*durns[1]))
start_f2, end_f2 = 1e3, 95e3
sweep2 = signal.chirp(t2, start_f2, t2[-1], end_f2)
sweep2 *= signal.windows.tukey(sweep2.size, 0.95)
sweep2 *= 0.8
sweep2_padded = np.pad(sweep2, pad_width=[int(fs*0.1)]*2, constant_values=[0,0])

dur1 = len(sweep1) / fs
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t1, sweep1)
plt.subplot(2, 1, 2)
plt.specgram(sweep1, NFFT=64, noverlap=32, Fs=fs)
plt.suptitle('Sweep 1')

dur2 = len(sweep2) / fs
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t2, sweep2)
plt.subplot(2, 1, 2)
plt.specgram(sweep2, NFFT=64, noverlap=32, Fs=fs)
plt.suptitle('Sweep 2')
plt.figure()

rms_values1 = []
db_values1 = []
rms_values2 = []
db_values2 = []
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
    fig1, ax1 = plt.subplots(figsize=(15, 5))
    ax1.set_title(file.split('_')[1] + f'\n Hour: {hour}, Minute: {minute}')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('RMS Value')
    ax1.grid(True)
    ax1.legend()

    fig2, ax2 = plt.subplots(figsize=(15, 5))
    ax2.set_title(file.split('_')[1] + f'\n Hour: {hour}, Minute: {minute}')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('RMS Value')
    ax2.grid(True)
    ax2.legend()

    recording, sample_rate = sf.read(DIR +file)

    # Apply matched filtering
    filtered_output1 = matched_filter(recording, sweep1)
    filtered_output2 = matched_filter(recording, sweep2)

    # Detect peaks
    peaks1 = detect_peaks(filtered_output1)
    peaks2 = detect_peaks(filtered_output2)

    if len(peaks1) > 0:
        # Extract the first sweep
        first_sweep_start = peaks1[0]
        first_sweep_end = first_sweep_start + len(sweep1)
        first_sweep = recording[first_sweep_start:first_sweep_end]

        rms_value = compute_rms(first_sweep)

        rms_values1.append([rms_value, hour, minute])
        #print(rms_values)
        db_value = rms_to_db(rms_value)
        db_values1.append([db_value, hour, minute])

        sf.write(DIR_first_sweep + file_time + '.wav', first_sweep, int(fs))
        # Plot the first sweep
        ax1.plot(np.linspace(0,len(first_sweep),len(first_sweep))/fs, first_sweep, label=file_time)

    if len(peaks2) > 0:
        # Extract the first sweep
        second_sweep_start = peaks2[0]
        second_sweep_end = second_sweep_start + len(sweep2)
        second_sweep = recording[second_sweep_start:second_sweep_end]

        rms_value = compute_rms(second_sweep)

        rms_values2.append([rms_value, hour, minute])
        #print(rms_values)
        db_value = rms_to_db(rms_value)
        db_values2.append([db_value, hour, minute])

        sf.write(DIR_second_sweep + file_time + '.wav', second_sweep, int(fs))
        # Plot the first sweep
        ax2.plot(np.linspace(0,len(second_sweep),len(second_sweep))/fs, second_sweep, label=file_time, color='orange')

print(f"RMS Values: {rms_values1}")
print(f"dB Values: {db_values1}")
print(f"RMS Values: {rms_values2}")
print(f"dB Values: {db_values2}")

plt.show(block = True)
    

#%% 
# first sweep analysis
# plot rms values and db values
fig, axs = plt.subplots(2, 1, figsize=(15, 15))
ax = axs[0]
ax.set_title('RMS Values')
ax.set_xlabel('Time (s)')
ax.set_ylabel('RMS Value')
ax.grid(True)
time = [f"{val[1]}:{val[2]}" for val in rms_values1]
time_in_minutes = [int(val[1]) + int(val[2])/60 for val in rms_values1]
ax.plot(time_in_minutes, [val[0] for val in rms_values1], label='RMS Values', marker='o')
ax.legend()
ax.set_xticks(time_in_minutes)
ax.set_xticklabels(time)
# plot db values
ax = axs[1]
ax.set_title('DB Values')
ax.set_xlabel('Time (s)')
ax.set_ylabel('DB Value')
ax.grid(True)
ax.plot(time_in_minutes, [val[0] for val in db_values1], label='DB Values', marker='o')
ax.legend()
ax.set_xticks(time_in_minutes)
ax.set_xticklabels(time)
ax.set_yticks(np.arange(np.floor(min([val[0] for val in db_values1])*10)/10, np.ceil(max([val[0] for val in db_values1])*10)/10, 0.2))
plt.suptitle(date + ' first sweep')

plt.show(block = False)
#%%
# second sweep analysis

# plot rms values and db values
fig, axs = plt.subplots(2, 1, figsize=(15, 15))
ax = axs[0]
ax.set_title('RMS Values')
ax.set_xlabel('Time (s)')
ax.set_ylabel('RMS Value')
ax.grid(True)
time = [f"{val[1]}:{val[2]}" for val in rms_values2]
time_in_minutes = [int(val[1]) + int(val[2])/60 for val in rms_values2]
ax.plot(time_in_minutes, [val[0] for val in rms_values2], label='RMS Values', marker='o')
ax.legend()
ax.set_xticks(time_in_minutes)
ax.set_xticklabels(time)
# plot db values
ax = axs[1]
ax.set_title('DB Values')
ax.set_xlabel('Time (s)')
ax.set_ylabel('DB Value')
ax.grid(True)
ax.plot(time_in_minutes, [val[0] for val in db_values2], label='DB Values', marker='o')
ax.legend()
ax.set_xticks(time_in_minutes)
ax.set_xticklabels(time)
ax.set_yticks(np.arange(np.floor(min([val[0] for val in db_values2])*10)/10, np.ceil(max([val[0] for val in db_values2])*10)/10, 0.2))
plt.suptitle(date + ' second sweep')

plt.show(block = False)
# %%
