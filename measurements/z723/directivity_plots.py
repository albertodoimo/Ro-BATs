# %% Libraries and files

import os
import soundfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Load audio files, then plot them in a 6x6 grid
DIR = "./cut_sweeps/"  # Directory containing the audio files
audio_files = os.listdir(DIR)  # List all files in the sweeps directory
audio_files.sort()  # Sort the files in ascending order

# Load audio files, then plot them in a 6x6 grid
DIR_noise = "./noise_floor/"  # Directory containing the audio files
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
# save figure
fig.savefig("./figures/sweeps.png")

# %% Radiance computation

channels = []
for i in np.arange(len(audio_files)):
    audio, fs = soundfile.read(DIR + audio_files[i])
    # if audio.shape[0] > 1919:
    #     audio = audio[0:1919]
    channels.append(audio)
channels = np.array(channels)

Channels = fft.fft(channels, n=2048, axis=1)
#Channels_uni = Channels[:,0:1024] # Select the first half of the FFT
Channels_uni = Channels
freqs = fft.fftfreq(2048, 1 / fs) # Compute the frequency vector
#freqs = freqs[0:1024]# Select the first half of the frequency vector


R = 1 # Distance from the source in meters

radiance = 4 * np.pi * R * np.abs(Channels_uni) # Radiance computation
theta = np.linspace(0, 350, 36) # Angle vector
theta = np.append(theta, theta[0])

# %% SNR computation between the sweeps and the noise floor measurements for each freq band

def calculate_snr_bands(audio, noise):

    # Compute signal power (RMS value squared)
    P_signal = np.mean(audio**2)

    # extract noise power
    P_noise = np.mean(noise**2)

    # Compute SNR in dB
    SNR = 10 * np.log10(P_signal / P_noise)

    # Noise floor in dBFS (full scale)
    noise_floor = 10 * np.log10(P_noise)

    return SNR, noise_floor

noises = [] # Load noise files
for i in np.arange(len(noise_files)): 
    noise, fs = soundfile.read(DIR_noise + noise_files[i]) 
    noises.append(noise)
noises = np.array(noises)

NOISES = fft.fft(noises, n=2048, axis=1) # Compute the FFT of the noise files
# NOISES_uni = NOISES[:,0:1024] # Select the first half of the FFT
NOISES_uni = NOISES
# %% Radiance display at multiple frequencies

central_freq = np.array([4e3, 6e3, 8e3, 10e3, 12e3, 14e3, 16e3, 18e3]) # Central frequencies of the bands
BW = 1e3 # Bandwidth of the bands

linestyles = ["-", "--", "-.", ":"] # Line styles for the plot
# Create a figure and a set of subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, subplot_kw={"projection": "polar"},figsize=(13, 13))
plt.suptitle("Radiance Pattern - CE32A-4 1/4\" Mini Speaker")
i = 0

for fc in central_freq[0:4]:
    rad_patt = np.mean(
        radiance[:, (freqs < fc + BW) & (freqs > fc - BW)], axis=1
    ) # Compute the mean radiance in the band
    rad_patt_norm = rad_patt / np.max(rad_patt) # Normalize the radiance
    rad_patt_norm_dB = 20 * np.log10(rad_patt_norm) # Convert the radiance to dB
    rad_patt_norm_dB = np.append(rad_patt_norm_dB, rad_patt_norm_dB[0]) # Append the first value to the end of the vector

    snrs = []
    for ii in np.arange(len(Channels_uni)):
        snr_value, noise_floor_value = calculate_snr_bands(
                                Channels_uni[ii, (freqs < fc + BW) & (freqs > fc - BW)],    
                                NOISES_uni[ii, (freqs < fc + BW) & (freqs > fc - BW)]) # Compute the SNR in the band
        snrs.append(snr_value)
    snrs = np.array(snrs)
    snrs = np.append(snrs, snrs[0])

    # Plot the radiance pattern
    if str(fc)[0:2] == '10': # Display the frequency in kHz
        ax1.plot(
        np.deg2rad(theta),
        rad_patt_norm_dB,
        label=str(fc)[0:2] + " [kHz]",
        linestyle=linestyles[i],
        )
        ax3.plot(
        np.deg2rad(theta),
        snrs,
        label=str(fc)[0:2] + " [kHz]",
        linestyle=linestyles[i],
        )
    else:
        ax1.plot(
        np.deg2rad(theta),
        rad_patt_norm_dB,
        label=str(fc)[0:1] + " [kHz]",
        linestyle=linestyles[i],
        )
        ax3.plot(
        np.deg2rad(theta),
        snrs,
        label=str(fc)[0:1] + " [kHz]",
        linestyle=linestyles[i],
        )   
    i += 1

# Display the legend
ax1.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
# offset polar axes by -90 degrees
ax1.set_theta_offset(np.pi / 2)
# set theta direction to clockwise
ax1.set_theta_direction(-1)
# more theta ticks
ax1.set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))
# less radial ticks
ax1.set_yticks(np.linspace(-40, 0, 5))
# Display the radial labels
ax1.set_rlabel_position(0)

ax3.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
# offset polar axes by -90 degrees
ax3.set_theta_offset(np.pi / 2)
# set theta direction to clockwise
ax3.set_theta_direction(-1)
# more theta ticks
ax3.set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))
# less radial ticks
ax3.set_yticks(np.linspace(0, 60, 7))
ax3.set_rlabel_position(0)

i = 0
for fc in central_freq[4:8]:
    rad_patt = np.mean(
        radiance[:, (freqs < fc + BW) & (freqs > fc - BW)], axis=1
    )
    noise_patt = np.mean(
        NOISES_uni[:, (freqs < fc + BW) & (freqs > fc - BW)], axis=1
    )
    rad_patt_norm = rad_patt / np.max(rad_patt)
    rad_patt_norm_dB = 20 * np.log10(rad_patt_norm)
    rad_patt_norm_dB = np.append(rad_patt_norm_dB, rad_patt_norm_dB[0])
    
    snrs = []
    for ii in np.arange(len(Channels_uni)):
        #Channels_uni = np.abs(Channels_uni) #moved outside the loop
        snr_value, noise_floor_value = calculate_snr_bands(Channels_uni[ii, (freqs < fc + BW) & (freqs > fc - BW)]    
                                 , NOISES_uni[ii, (freqs < fc + BW) & (freqs > fc - BW)])
        snrs.append(snr_value)
    snrs = np.array(snrs)
    snrs = np.append(snrs, snrs[0])

    ax2.plot(
        np.deg2rad(theta),
        rad_patt_norm_dB,
        label=str(fc)[0:2] + " [kHz]",
        linestyle=linestyles[i],
    )
    ax4.plot(
        np.deg2rad(theta),
        snrs,
        label=str(fc)[0:2] + " [kHz]",
        linestyle=linestyles[i],
        )
    i += 1
ax2.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
# offset polar axes by -90 degrees
ax2.set_theta_offset(np.pi / 2)
# set theta direction to clockwise
ax2.set_theta_direction(-1)
# more theta ticks
ax2.set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))
# less radial ticks
ax2.set_yticks(np.linspace(-40, 0, 5))
ax2.set_rlabel_position(0)

ax4.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
# offset polar axes by -90 degrees
ax4.set_theta_offset(np.pi / 2)
# set theta direction to clockwise
ax4.set_theta_direction(-1)
# more theta ticks
ax4.set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))
# less radial ticks
ax4.set_yticks(np.linspace(0, 60, 7))
ax4.set_rlabel_position(0)
ax4.set_title("SNR Pattern - CE32A-4 1/4\" Mini Speaker")
ax3.set_title("SNR Pattern - CE32A-4 1/4\" Mini Speaker")

plt.tight_layout()
plt.show()
#save figure
fig.savefig("./figures/radiance_SNR.png")

# %% Mean radiance pattern display

rad_patt = np.mean(radiance, axis=1)
rad_patt_norm = rad_patt / np.max(rad_patt)
rad_patt_norm_dB = 20 * np.log10(rad_patt_norm)
rad_patt_norm_dB = np.append(rad_patt_norm_dB, rad_patt_norm_dB[0])

fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))
ax.plot(np.deg2rad(theta), rad_patt_norm_dB)
# offset polar axes by -90 degrees
ax.set_theta_offset(np.pi / 2)
# set theta direction to clockwise
ax.set_theta_direction(-1)
# more theta ticks
ax.set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))
ax.set_ylabel("dB")
# less radial ticks
ax.set_yticks(np.linspace(-40, 0, 5))
ax.set_rlabel_position(0)
ax.set_title(
    "CE32A-4 1/4\" Mini Speaker - Overall Radiance Pattern 1[kHz] - 20[kHz]"
)

plt.show()
#save figure
fig.savefig("./figures/radiance_overall.png")

# %% Plot of the noise data

fig, axs = plt.subplots(6, 6, figsize=(20, 20))

for i in range(6):
    for j in range(6):
        # Load audio file
        audio, fs = soundfile.read(DIR_noise + audio_files[i * 6 + j])
        # Plot audio file
        axs[i, j].plot(np.linspace(0, len(audio) / fs, len(audio)), audio)
        axs[i, j].set_title(audio_files[i * 6 + j])
        axs[i, j].set_xlabel("Time (s)")
        axs[i, j].set_ylabel("Amplitude")
        # ylim for better visualization
        axs[i, j].set_ylim([-0.02, 0.02])
        # Shared x and y axes
        axs[i, j].sharex(axs[0, 0])
        axs[i, j].sharey(axs[0, 0])

plt.suptitle("Recorded 50ms noise floor - CE32A-4 1/4\" Mini Speaker", fontsize=30)
plt.tight_layout()
plt.show(block=False)
# save figure
fig.savefig("./figures/noises.png")


#%% overall SNR computation

def calculate_snr(audio, noise):

    # Compute signal power (RMS value squared)
    P_signal = np.mean(audio**2)

    # If noise segment is provided, extract noise power
    P_noise = np.mean(noise**2)

    # Compute SNR in dB
    SNR = 10 * np.log10(P_signal / P_noise)

    # Noise floor in dBFS (full scale)
    noise_floor = 10 * np.log10(P_noise)

    return SNR, noise_floor

snrs = []
for i in np.arange(len(channels)):
    snr_value, noise_floor_value = calculate_snr(channels[i], noises[i])
    snrs.append(snr_value)
snrs = np.array(snrs)
snrs = np.append(snrs, snrs[0])

fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))
ax.plot(np.deg2rad(theta), snrs)
# offset polar axes by -90 degrees
ax.set_theta_offset(np.pi / 2)
# set theta direction to clockwise
ax.set_theta_direction(-1)
# more theta ticks
ax.set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))
ax.set_ylabel("dB")
# less radial ticks
ax.set_yticks(np.linspace(-10, 30, 5))
ax.set_rlabel_position(0)
ax.set_title("CE32A-4 1/4inch Mini Speaker - overall signal-to-noise ratio 1[kHz] - 20[kHz]")

plt.show()
#save figure
fig.savefig("./figures/snr_overall.png")

# %%
