# -*- coding: utf-8 -*-
"""
Example run-through
===================
This IS NOT a complete run-through. The follow things need to implemented by the user
    * playback signal segmentation 
    * bandpass filtering / other cleaning
    * deconvolution to remove reflections

Other things to remember
------------------------
The GRAS microphone and target microphone may or may not be recorded with the same
analog-digital-converter!! Be Very Aware of this while comparing the sensitivities 
between the two microphones!!

Experimental notes
------------------
* Remember to write down all the GAIN values in dB for every recording you make.

WARNING
-------
As of APRIL 2025 -  this is a prototype module only. 
See below for the steps you need to take to ensure everything makes sense.


Always double-check with a 'validation' signal
----------------------------------------------
Test your calibration outcomes with a second signal that wasn't used anywhere before
and also recorded by both the GRAs and target mic. 

Created on Sat Apr 12 07:35:50 2025

@author: theja
"""

# %%
import numpy as np 
import soundfile as sf
import matplotlib.pyplot as plt 
from utilities import *
import scipy.signal as sig
from scipy import fft

#%%
durns = np.array([3, 4, 5, 8, 10] )*1e-3
fs = 48000 # Hz

chirp = []
all_sweeps = []
for durn in durns:
    t = np.linspace(0, durn, int(fs*durn))
    start_f, end_f = 1e3, 20e3
    sweep = signal.chirp(t, start_f, t[-1], end_f)
    sweep *= signal.windows.tukey(sweep.size, 0.95)
    sweep *= 0.8
    sweep_padded = np.pad(sweep, pad_width=[int(fs*0.1)]*2, constant_values=[0,0])
    all_sweeps.append(sweep_padded)
    chirp.append(sweep)

# Read the saved WAV file
out_signal, _ = sf.read('1_20k_5sweeps.wav')

# Plot the time-domain signal and spectrogram
plt.figure(figsize=(15, 8))

# Time-domain plot
plt.subplot(2, 1, 1)
plt.plot(np.linspace(0, len(out_signal) / fs, len(out_signal)), out_signal)
plt.title('Output Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# Spectrogram plot
plt.subplot(2, 1, 2, sharex=plt.gca())
plt.specgram(out_signal, Fs=fs, NFFT=512, noverlap=256, cmap='viridis')
plt.title('Spectrogram of the Output Signal')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.ylim(0, 25e3)
plt.tight_layout()

plt.figure(figsize=(15, 10))
for i, sweep in enumerate(chirp):
    plt.subplot(len(chirp), 2, 2 * i + 1)
    plt.plot(np.linspace(0, len(sweep) / fs, len(sweep)), sweep)
    plt.title(f'Sweep {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    # Spectrogram plot

    plt.subplot(len(chirp), 2, 2 * i + 2)
    plt.specgram(sweep, Fs=fs, NFFT=32, noverlap=16, cmap='viridis')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.ylim(0, 25e3)
    plt.suptitle('Output sweeps', fontsize=20)

plt.tight_layout()
plt.show()

#%%
# Define the matched filter function
def matched_filter(recording, chirp_template):
    filtered_output = np.roll(signal.correlate(recording, chirp_template, 'same', method='direct'), -len(chirp_template)//2)
    filtered_output *= signal.windows.tukey(filtered_output.size, 0.1)
    filtered_envelope = np.abs(signal.hilbert(filtered_output))
    return filtered_envelope

# Detect peaks in the matched filter output
def detect_peaks(filtered_output, sample_rate):
    peaks, properties = signal.find_peaks(filtered_output, prominence=0.5, distance=0.2 * sample_rate)
    return peaks

gras_pbk_audio_or, fs_gras = sf.read('./2025-05-09/ref_tone_gras2025-05-09_11-59-29.wav')
SPH0645_pbk_audio_or, fs_SPH0645 = sf.read('./2025-05-09/extracted_channels/channel_separation/000_3.wav')

chirp_to_use = 0
# resampling 
gras_pbk_audio_res = sig.resample(gras_pbk_audio_or, int(fs_SPH0645*len(gras_pbk_audio_or)/fs_gras))

gras_pbk_audio_matched = matched_filter(gras_pbk_audio_res, chirp[chirp_to_use])
SPH0645_pbk_audio_matched = matched_filter(SPH0645_pbk_audio_or, chirp[chirp_to_use])

# Detect peaks
peaks_gras = detect_peaks(gras_pbk_audio_matched, fs_SPH0645)
peaks_SPH0645 = detect_peaks(SPH0645_pbk_audio_matched, fs_SPH0645)
print(f"Detected peaks: gras = {len(peaks_gras)}, SPH0645 = {len(peaks_SPH0645)}")

# plot the peaks
plt.figure(figsize=(15, 5))
plt.subplot(2, 1, 1)
plt.plot(np.linspace(0, len(gras_pbk_audio_matched) / fs_SPH0645, len(gras_pbk_audio_matched)), gras_pbk_audio_matched)
plt.plot(peaks_gras/fs_SPH0645, gras_pbk_audio_matched[peaks_gras], 'ro')
plt.title('Matched Filter Output - GRAS')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
#plt.xlim(3.5, 3.54)
plt.subplot(2, 1, 2)
plt.plot(np.linspace(0, len(SPH0645_pbk_audio_matched) / fs_SPH0645, len(SPH0645_pbk_audio_matched)), SPH0645_pbk_audio_matched)
plt.plot(peaks_SPH0645/fs_SPH0645, SPH0645_pbk_audio_matched[peaks_SPH0645], 'ro')
plt.title('Matched Filter Output - SPH0645')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
#plt.xlim(4.325, 4.34)
plt.tight_layout()
plt.show()

#%% 
gras_pbk_audio = gras_pbk_audio_res[int(peaks_gras[chirp_to_use]):int(peaks_gras[chirp_to_use]) + int(fs_SPH0645*durns[chirp_to_use])]
SPH0645_pbk_audio = SPH0645_pbk_audio_or[int(peaks_SPH0645[chirp_to_use]):int(peaks_SPH0645[chirp_to_use]) + int(fs_SPH0645*durns[chirp_to_use])]

# Plot the playback signals
plt.figure(figsize=(10,5))
plt.plot(np.linspace(0,len(gras_pbk_audio)/fs_SPH0645, len(gras_pbk_audio)) ,gras_pbk_audio)
plt.title('Playback signal from GRAS mic')
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Amplitude', fontsize=12)
plt.figure(figsize=(10,5))
plt.plot(np.linspace(0,len(SPH0645_pbk_audio)/fs_SPH0645, len(SPH0645_pbk_audio)), SPH0645_pbk_audio)
plt.title('Playback signal from SPH0645 mic')
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Amplitude', fontsize=12)

# Load the 1 Pa reference tone 
gras_1Pa_tone, fs_gras = sf.read('./2025-05-09/gras_1Pa_ch9_30dB_chA_20dB.wav', start=int(fs_gras*0.5),
                        stop=int(fs_gras*1.5))

gras_1Pa_tone = sig.resample(gras_1Pa_tone, fs_SPH0645*1)

# Plot the reference tone
plt.figure(figsize=(15,5))
plt.plot(np.linspace(0,len(gras_1Pa_tone)/fs_SPH0645, len(gras_1Pa_tone)), gras_1Pa_tone)
plt.title('GRAS 1 KHz 1 Pa calibrator reference tone')
plt.xlim(0, 0.01)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude', fontsize=12)
plt.grid()


#%%
# Check the SNR at the spectral level - use a silent audio clip from the above recordings
# to be sure that your measurements mean something useful. Remember garbage in, garbage out.
 
# raise NotImplementedError('Check your SNR from the ambient sound!')
# snr_target = dB(bandwise_tgtmic/tgt_silence_bandwise)
# snr_gras = dB(bandwise_grasmic/gras_silence_bandwise)

# %% SNR computation between the sweeps and the noise floor measurements for each freq band

central_freq = np.arange(1000, fs_SPH0645//2, 250) 
BW = 0.25e3 # Bandwidth of the bands

def calculate_snr(audio, noise):

    # Compute signal power (RMS value squared)
    P_signal = np.mean(audio**2)

    # extract noise power
    P_noise = np.mean(noise**2)

    # Compute SNR in dB
    SNR = 10 * np.log10(P_signal / P_noise)

    return SNR

# Use the initial part opf the GRAS recording as noise
noise_SPH0645 = SPH0645_pbk_audio_or[int(0.520*fs_SPH0645):int(0.523*fs_SPH0645)] 

plt.figure()
plt.plot(np.linspace(0, len(noise_SPH0645)/fs_SPH0645, len(noise_SPH0645)), noise_SPH0645)
plt.show()

Noise_fft = np.fft.rfft(noise_SPH0645) # Compute the FFT of the noise files
Noise_fftfreqs = np.fft.rfftfreq(noise_SPH0645.size, 1/fs_SPH0645)

plt.figure()
plt.plot(Noise_fftfreqs,np.abs(Noise_fft))
plt.xlabel('Frequency, Hz', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.xticks(np.arange(0, fs_SPH0645//2, 1000), rotation=45)
plt.title('FFT of the noise floor')
plt.grid()
plt.show()

# Signal fft of SPH0645 playback audio

SPH0645_fft = np.fft.rfft(SPH0645_pbk_audio) # Compute the FFT of the noise files
SPH0645_fftfreqs = np.fft.rfftfreq(SPH0645_pbk_audio.size, 1/fs_SPH0645)

plt.figure()
plt.plot(SPH0645_fftfreqs,np.abs(SPH0645_fft))
plt.xlabel('Frequency, Hz', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.title('FFT of the SPH0645 playback audio')
plt.xticks(np.arange(0, fs_SPH0645//2, 1000), rotation=45)
plt.grid()
plt.show()

snrs = []
for fc in central_freq:
    snr_value = calculate_snr(
                            SPH0645_fft[(SPH0645_fftfreqs < fc + BW) & (SPH0645_fftfreqs > fc - BW)],    
                            Noise_fft[(Noise_fftfreqs < fc + BW) & (Noise_fftfreqs > fc - BW)]) # Compute the SNR in the band
    snrs.append(snr_value)
snrs = np.array(snrs)
snrs = snrs - np.min(snrs)

plt.figure()
plt.plot(central_freq, snrs, label='SNR')
plt.xlabel('Frequencies, Hz', fontsize=12);
plt.ylabel('dB', fontsize=12)
plt.grid()
plt.xticks(np.arange(0, fs_SPH0645//2, 1000), rotation=45)
plt.title('SNR of the SPH0645 playback audio')
plt.tight_layout()
plt.show()

#%%
# Calibration mic: Calculate the rms_Pascal of the 1 Pa calibration tone
rms_1Pa_tone = rms(gras_1Pa_tone)
print(f'The calibration mic has a sensitivity of {np.round(rms_1Pa_tone,3)}rms/Pa. RMS relevant only for this ADC!')

# Now measure mic RMS over all frequency bands
gras_centrefreqs, gras_freqrms = calc_native_freqwise_rms(gras_pbk_audio, fs_SPH0645)

# Filter out the frequencies that are not in the original signal 
mask = (gras_centrefreqs >= 1000) & (gras_centrefreqs <= 20e3)

idx_1k = np.where(gras_centrefreqs >= 1e3)[0][0]
idx_20k = np.where(gras_centrefreqs <= 20e3)[0][-1]
print(f"Index for 1 kHz: {idx_1k}, frequency: {gras_centrefreqs[idx_1k]}")
print(f"Index for 20 kHz: {idx_20k}, frequency: {gras_centrefreqs[idx_20k]}")

gras_centrefreqs = gras_centrefreqs[mask]

gras_freqrms = gras_freqrms[idx_1k:idx_20k+1]

# Convert from RMS to Pascals (rms equivalent) since we know the GRAS sensitivity
gras_freqParms = gras_freqrms/rms_1Pa_tone # now the levels of each freq band in Pa_rms

plt.figure()
a0 = plt.subplot(211)
plt.plot(gras_centrefreqs, gras_freqParms)
plt.ylabel('Pressure_rmseqv., Pa', fontsize=12)
plt.title('GRAS mic recording of playback')
plt.grid()
plt.xticks(np.linspace(1000, 20000, 20), rotation=45)
plt.subplot(212, sharex=a0)
plt.plot(gras_centrefreqs, pascal_to_dbspl(gras_freqParms))
plt.xlabel('Frequencies, Hz', fontsize=12);
plt.ylabel('Sound pressure level,\n dBrms SPL re 20$\mu$Pa', fontsize=12)
plt.grid()
plt.xticks(np.linspace(1000, 20000, 20), rotation=45)
plt.tight_layout()


#%%
# Target microphone. Here we'll cover the case where we only get an RMS/Pa
# sensitivity. The other option is to calculate a mV/Pa sensitivity - which allows
# you to use the mic aross different ADCs - but also needs more info on the ADC specs

SPH0645_centrefreqs, SPH0645_freqrms = calc_native_freqwise_rms(SPH0645_pbk_audio, fs_SPH0645)

# Filter out the frequencies that are not in the original signal 
mask = (SPH0645_centrefreqs >= 1000) & (SPH0645_centrefreqs <= 20e3)

idx_1k = np.where(SPH0645_centrefreqs >= 1e3)[0][0]
idx_20k = np.where(SPH0645_centrefreqs <= 20e3)[0][-1]
print(f"Index for 1 kHz: {idx_1k}, frequency: {SPH0645_centrefreqs[idx_1k]}")
print(f"Index for 20 kHz: {idx_20k}, frequency: {SPH0645_centrefreqs[idx_20k]}")

SPH0645_centrefreqs = SPH0645_centrefreqs[mask]

SPH0645_freqrms = SPH0645_freqrms[idx_1k:idx_20k+1]

plt.figure()
a0 = plt.subplot(211)
plt.plot(SPH0645_centrefreqs, SPH0645_freqrms)
plt.ylabel('a.u. rmseqv.', fontsize=15)
plt.title('SPH0645 mic recording of playback')
plt.grid()
plt.xticks(np.linspace(1000, 20000, 20), rotation=45)
plt.subplot(212, sharex=a0)
plt.plot(SPH0645_centrefreqs, dB(SPH0645_freqrms))
plt.xlabel('Frequencies, Hz', fontsize=12);
plt.ylabel('dBrms a.u.', fontsize=12)
plt.grid()
plt.xticks(np.linspace(1000, 20000, 20), rotation=45)
plt.tight_layout()


#%%
# Now let's calculate the RMS/Pa sensitivity using the knowledge from the 
# calibration mic
SPH0645_sensitivity = np.array(SPH0645_freqrms)/np.array(gras_freqParms)

print(f'The target mic has a sensitivity at {SPH0645_centrefreqs[0]} Hz is: {dB(SPH0645_sensitivity[0])} dB a.u. rms/Pa')
plt.figure()
a0 = plt.subplot(211)
plt.plot(SPH0645_centrefreqs, SPH0645_sensitivity)
plt.ylabel('a.u. RMS/Pa', fontsize=12)
plt.title('Target mic sensitivity')
plt.grid()
plt.xticks(np.linspace(1000, 20000, 20), rotation=45)
plt.subplot(212, sharex=a0)
plt.plot(SPH0645_centrefreqs, dB(SPH0645_sensitivity))
plt.xlabel('Frequencies, Hz', fontsize=12);
plt.ylabel('dB a.u. rms/Pa', fontsize=12)
plt.grid()
plt.xticks(np.linspace(1000, 20000, 20), rotation=45)
plt.tight_layout()
# plt.ylim(-60,0)

#%% 
# We now have the target mic sensitivity - how do we use it to calculate the
# actual dB SPL? 

# Here we load a separate 'recorded sound' - a 'validation' audio clip let's call it 
chirp_to_use = 1

gras_pbk_audio_matched = matched_filter(gras_pbk_audio_res, chirp[chirp_to_use])
SPH0645_pbk_audio_matched = matched_filter(SPH0645_pbk_audio_or, chirp[chirp_to_use])

# Detect peaks
peaks_gras = detect_peaks(gras_pbk_audio_matched, fs_SPH0645)
peaks_SPH0645 = detect_peaks(SPH0645_pbk_audio_matched, fs_SPH0645)
print(f"Detected peaks: gras = {len(peaks_gras)}, SPH0645 = {len(peaks_SPH0645)}")

# plot the peaks
plt.figure(figsize=(15, 5))
plt.subplot(2, 1, 1)
plt.plot(np.linspace(0, len(gras_pbk_audio_matched) / fs_SPH0645, len(gras_pbk_audio_matched)), gras_pbk_audio_matched)
plt.plot(peaks_gras/fs_SPH0645, gras_pbk_audio_matched[peaks_gras], 'ro')
plt.title('Matched Filter Output - GRAS')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
#plt.xlim(3.5, 3.54)
plt.subplot(2, 1, 2)
plt.plot(np.linspace(0, len(SPH0645_pbk_audio_matched) / fs_SPH0645, len(SPH0645_pbk_audio_matched)), SPH0645_pbk_audio_matched)
plt.plot(peaks_SPH0645/fs_SPH0645, SPH0645_pbk_audio_matched[peaks_SPH0645], 'ro')
plt.title('Matched Filter Output - SPH0645')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
#plt.xlim(4.325, 4.34)
plt.tight_layout()
plt.show()


gras_rec = gras_pbk_audio_res[int(peaks_gras[chirp_to_use]):int(peaks_gras[chirp_to_use]) + int(fs_SPH0645*durns[chirp_to_use])]
recorded_sound = SPH0645_pbk_audio_or[int(peaks_SPH0645[chirp_to_use]):int(peaks_SPH0645[chirp_to_use]) + int(fs_SPH0645*durns[chirp_to_use])]

# Plot the playback signals
plt.figure(figsize=(10,5))
plt.plot(np.linspace(0,len(gras_rec)/fs_SPH0645, len(gras_rec)) ,gras_rec)
plt.title('Validation signal from GRAS mic')
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Amplitude', fontsize=12)
plt.figure(figsize=(10,5))
plt.plot(np.linspace(0,len(recorded_sound)/fs_SPH0645, len(recorded_sound)), recorded_sound)
plt.title('Validation signal from SPH0645 mic')
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Amplitude', fontsize=12)

#%%
# And finally let's check that the calibration makes sense
# using a sound that we didn't use to calculate the sensitivity
# If the length of the recorded target mic audio here is not the same as the calibration audio. 
#  then you'll need to interpolate the microphone sensitivity using interpolate_freq_response in the
# utilities.py module
recsound_centrefreqs, freqwise_rms = calc_native_freqwise_rms(recorded_sound, fs_SPH0645)

mask = (recsound_centrefreqs >= 1000) & (recsound_centrefreqs <= 20e3)
idx_1k = np.where(recsound_centrefreqs >= 1e3)[0][0]
idx_20k = np.where(recsound_centrefreqs <= 20e3)[0][-1]
print(f"Index for 1 kHz: {idx_1k}, frequency: {recsound_centrefreqs[idx_1k]}")
print(f"Index for 20 kHz: {idx_20k}, frequency: {recsound_centrefreqs[idx_20k]}")
recsound_centrefreqs = recsound_centrefreqs[mask]
freqwise_rms = freqwise_rms[idx_1k:idx_20k+1]

interp_sensitivity = interpolate_freq_response([SPH0645_centrefreqs, SPH0645_sensitivity],
                          recsound_centrefreqs)
freqwise_Parms = freqwise_rms/interp_sensitivity # go from rms to Pa(rmseq.)
freqwiese_dbspl = pascal_to_dbspl(freqwise_Parms)

gras_centrefreqs, gras_freqrms = calc_native_freqwise_rms(gras_rec, fs_SPH0645)

idx_1k = np.where(gras_centrefreqs >= 1e3)[0][0]
idx_20k = np.where(gras_centrefreqs <= 20e3)[0][-1]
print(f"Index for 1 kHz: {idx_1k}, frequency: {gras_centrefreqs[idx_1k]}")
print(f"Index for 20 kHz: {idx_20k}, frequency: {gras_centrefreqs[idx_20k]}")
gras_centrefreqs = gras_centrefreqs[mask]
gras_freqrms = gras_freqrms[idx_1k:idx_20k+1]

gras_Pa = gras_freqrms/rms_1Pa_tone
gras_dbspl = pascal_to_dbspl(gras_Pa)

plt.figure()
plt.plot(gras_centrefreqs,gras_dbspl, label='gras')
plt.plot(recsound_centrefreqs,freqwiese_dbspl, label='SPH0645')
plt.plot(central_freq, snrs, label='SNR')
plt.ylabel('dBrms SPL, re 20$\mu$Pa', fontsize=12)
plt.xlabel('Frequency, Hz', fontsize=12)
plt.title('Validation by comparing of GRAS and a secondary SPH0645 recording')
plt.legend()
plt.grid()
plt.xticks(np.linspace(1000, 20000, 20), rotation=45)
plt.show()

#%%
# Now we know the sensitivity of the target mic - let's finally calculate
# the dB SPL of the recorded sound!
# We rely on combining the Pa rms of all relevant frequencies 
# e.g. see https://electronics.stackexchange.com/questions/642109/summing-rms-values-proof

frequency_band = [0.2e3, 15e3] # min, max frequency to do the compensation Hz

# Choose to calculate the dB SPL only for the frequency range of interest.
# Target mic
tgtmic_relevant_freqs = np.logical_and(recsound_centrefreqs>=frequency_band[0],
                                recsound_centrefreqs<=frequency_band[1])
total_rms_freqwise_Parms = np.sqrt(np.sum(freqwise_Parms[tgtmic_relevant_freqs]**2))

# Ground truth GRAS mic audio of the same sound. Here we use only the relevant 
# frequency band of the recorded sweep
gras_relevant_freqs = np.logical_and(gras_centrefreqs>=frequency_band[0],
                                gras_centrefreqs<=frequency_band[1])

#%% This is one way to do it - use the RAW audio, and calculate its rms
# since we the overall flat sensitivity of the GRAS
gras_overallaudio_Parms = rms(gras_rec)/rms_1Pa_tone
# This is the other way to do it by combining RMS values - like we did for the Sennheiser
gras_totalrms_Parms = np.sqrt(np.sum(gras_Pa[gras_relevant_freqs]**2))

print(f'GRAS dBrms SPL measures:{pascal_to_dbspl(gras_overallaudio_Parms)}, {pascal_to_dbspl(gras_totalrms_Parms)}')
print(f'SPH0645 dBrms SPL measure: {pascal_to_dbspl(total_rms_freqwise_Parms)}')

# %%
