# %%
import numpy as np 
import soundfile as sf
import matplotlib.pyplot as plt 
from utilities import *
import scipy.signal as sig

#%%

DIR = "./2025-06-03/"  # Directory containing the audio files

gras_40_or, fs = sf.read('./2025-06-03/-40db_02_24k_5sweeps_channel9_192k.wav')
gras_35_or, fs = sf.read('./2025-06-03/-35db_02_24k_5sweeps_channel9_192k.wav')
gras_30_or, fs = sf.read('./2025-06-03/-30db_02_24k_5sweeps_channel9_192k.wav')
gras_25_or, fs = sf.read('./2025-06-03/-25db_02_24k_5sweeps_channel9_192k.wav')
gras_20_or, fs = sf.read('./2025-06-03/-20db_02_24k_5sweeps_channel9_192k.wav')

# Load the 1 Pa reference tone 
gras_1Pa_tone, fs = sf.read('./2025-06-03/ref_tone_gras_1Pa_ch9_30dB_chA_20dB.wav', start=int(fs*0.5),
                        stop=int(fs*1.5))

# %% Plot the audio signals
# durns = np.array([3, 4, 5, 8, 10] )*1e-3
durns = np.array([3, 4, 5, 8] )*1e-3

chirp = []
all_sweeps = []
for durn in durns:
    t = np.linspace(0, durn, int(fs*durn))
    #start_f, end_f = 1e3, 20e3
    start_f, end_f = 2e2, 24e3
    sweep = signal.chirp(t, start_f, t[-1], end_f)
    #sweep *= signal.windows.tukey(sweep.size, 0.95)
    sweep *= signal.windows.tukey(sweep.size, 0.2)
    sweep *= 0.8
    sweep_padded = np.pad(sweep, pad_width=[int(fs*0.1)]*2, constant_values=[0,0])
    all_sweeps.append(sweep_padded)
    chirp.append(sweep)

#%%
# Define the matched filter function
def matched_filter(recording, chirp_template):
    filtered_output = np.roll(signal.correlate(recording, chirp_template, 'same', method='direct'), -len(chirp_template)//2)
    filtered_output *= signal.windows.tukey(filtered_output.size, 0.1)
    filtered_envelope = np.abs(signal.hilbert(filtered_output))
    return filtered_envelope

# Detect peaks in the matched filter output
def detect_peaks(filtered_output, sample_rate):
    peaks, properties = signal.find_peaks(filtered_output, prominence=5, distance=0.2 * sample_rate)
    return peaks


# load the GRAS and SPH0645 audio files
# gras_pbk_audio_or, fs_gras = sf.read('./2025-06-03/02_24k_5sweeps_channel9_192k.wav')

chirp_to_use = 0

gras_40_matched = matched_filter(gras_40_or, chirp[chirp_to_use])
gras_35_matched = matched_filter(gras_35_or, chirp[chirp_to_use])
gras_30_matched = matched_filter(gras_30_or, chirp[chirp_to_use])
gras_25_matched = matched_filter(gras_25_or, chirp[chirp_to_use])
gras_20_matched = matched_filter(gras_20_or, chirp[chirp_to_use])

# Detect peaks
peaks_gras_40 = detect_peaks(matched_filter(gras_40_matched, chirp[chirp_to_use]), fs)
peaks_gras_35 = detect_peaks(matched_filter(gras_35_matched, chirp[chirp_to_use]), fs)
peaks_gras_30 = detect_peaks(matched_filter(gras_30_matched, chirp[chirp_to_use]), fs)
peaks_gras_25 = detect_peaks(matched_filter(gras_25_matched, chirp[chirp_to_use]), fs)
peaks_gras_20 = detect_peaks(matched_filter(gras_20_matched, chirp[chirp_to_use]), fs)

print(f"Detected peaks: gras = {len(peaks_gras_40)}")
print(f"Detected peaks: gras = {len(peaks_gras_35)}")
print(f"Detected peaks: gras = {len(peaks_gras_30)}")
print(f"Detected peaks: gras = {len(peaks_gras_25)}")
print(f"Detected peaks: gras = {len(peaks_gras_20)}")

# Plot the matched filter outputs and detected peaks for all audio files
plt.figure(figsize=(15, 10))

audio_labels = ['-40 dB', '-35 dB', '-30 dB', '-25 dB', '-20 dB']
matched_outputs = [gras_40_matched, gras_35_matched, gras_30_matched, gras_25_matched, gras_20_matched]
peaks_list = [peaks_gras_40, peaks_gras_35, peaks_gras_30, peaks_gras_25, peaks_gras_20]

for i, (matched, peaks, label) in enumerate(zip(matched_outputs, peaks_list, audio_labels), 1):
    plt.subplot(5, 1, i)
    t = np.linspace(0, len(matched) / fs, len(matched))
    plt.plot(t, matched, label=f'Matched Output {label}')
    plt.plot(peaks / fs, matched[peaks], 'ro', label='Detected Peaks')
    plt.title(f'Matched Filter Output - GRAS {label}')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()

plt.show()

# %%
gras_40 = gras_40_or[int(peaks_gras_40[chirp_to_use]):int(peaks_gras_40[chirp_to_use]) + int(fs*durns[chirp_to_use])]
gras_35 = gras_35_or[int(peaks_gras_35[chirp_to_use]):int(peaks_gras_35[chirp_to_use]) + int(fs*durns[chirp_to_use])]
gras_30 = gras_30_or[int(peaks_gras_30[chirp_to_use]):int(peaks_gras_30[chirp_to_use]) + int(fs*durns[chirp_to_use])]
gras_25 = gras_25_or[int(peaks_gras_25[chirp_to_use]):int(peaks_gras_25[chirp_to_use]) + int(fs*durns[chirp_to_use])]
gras_20 = gras_20_or[int(peaks_gras_20[chirp_to_use]):int(peaks_gras_20[chirp_to_use]) + int(fs*durns[chirp_to_use])]

# Plot the extracted chirp segments for each amplifier level
plt.figure(figsize=(12, 15))
audio_segments = [gras_40, gras_35, gras_30, gras_25, gras_20]
labels = ['-40 dB', '-35 dB', '-30 dB', '-25 dB', '-20 dB']

# Find global min and max for y-axis
ymin = min([segment.min() for segment in audio_segments])
ymax = max([segment.max() for segment in audio_segments])

for i, (segment, label) in enumerate(zip(audio_segments, labels), 1):
    t = np.linspace(0, len(segment) / fs, len(segment))
    plt.subplot(5, 1, i)
    plt.plot(t, segment)
    plt.title(f'Extracted Chirp Segment - GRAS {label}')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.ylim(ymin, ymax)
    plt.tight_layout()
    plt.grid()
plt.show()

#%%
rms_1Pa_tone = rms(gras_1Pa_tone)
print(f'The calibration mic has a sensitivity of {np.round(rms_1Pa_tone,3)}rms/Pa. RMS relevant only for this ADC!')

gras_overallaudio_Parms_40 = rms(gras_40)/rms_1Pa_tone
gras_overallaudio_Parms_35 = rms(gras_35)/rms_1Pa_tone
gras_overallaudio_Parms_30 = rms(gras_30)/rms_1Pa_tone
gras_overallaudio_Parms_25 = rms(gras_25)/rms_1Pa_tone
gras_overallaudio_Parms_20 = rms(gras_20)/rms_1Pa_tone

# %%
print(f'GRAS dBrms SPL measures:{pascal_to_dbspl(gras_overallaudio_Parms_40)} dB SPL for -40 amplifier level')
print(f'GRAS dBrms SPL measures:{pascal_to_dbspl(gras_overallaudio_Parms_35)} dB SPL for -35 amplifier level')
print(f'GRAS dBrms SPL measures:{pascal_to_dbspl(gras_overallaudio_Parms_30)} dB SPL for -30 amplifier level')
print(f'GRAS dBrms SPL measures:{pascal_to_dbspl(gras_overallaudio_Parms_25)} dB SPL for -25 amplifier level')
print(f'GRAS dBrms SPL measures:{pascal_to_dbspl(gras_overallaudio_Parms_20)} dB SPL for -20 amplifier level')

# %%
# plot the dbrms values with grid 
db_levels = [-40, -35, -30, -25, -20]
dbspl_values = [
    pascal_to_dbspl(gras_overallaudio_Parms_40),
    pascal_to_dbspl(gras_overallaudio_Parms_35),
    pascal_to_dbspl(gras_overallaudio_Parms_30),
    pascal_to_dbspl(gras_overallaudio_Parms_25),
    pascal_to_dbspl(gras_overallaudio_Parms_20)
]
min_val = min(dbspl_values)
normalized = [v - min_val for v in dbspl_values]

plt.figure(figsize=(10, 6))
plt.plot(db_levels, dbspl_values, marker='o')
for x, y, norm in zip(db_levels, dbspl_values, normalized):
    plt.annotate(f'{norm[0]:.2f} dB', (x-1, y+1), textcoords="offset points", xytext=(10,5), ha='left', fontsize=9)
plt.title('Amplifier linearity check  SPL Measures')
plt.xlabel('Amplifier output Level [dB]')
plt.ylabel('dBrms SPL')
plt.xticks(db_levels)
plt.yticks(np.array(dbspl_values).reshape(1, 5)[0])
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
