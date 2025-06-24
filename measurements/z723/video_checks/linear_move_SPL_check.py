# Video file analysis of robat 240 moving linearly towards the 238, while 238 outputs a cosntant amplitude chirp. 
# Goal: validation of SPL value obtained by the recordings, knowing the sensitivity of the mems array and dayton loudspeaker output level. 
# 
# Author: Alberto Doimo
# Date: 2025-06-18
# Z723

#%%
import numpy as np
import soundfile as sf
import librosa
from scipy import signal
import io
import cv2
import matplotlib.pyplot as plt
from moviepy import VideoFileClip
from scipy.interpolate import interp1d
import pandas as pd
import sys
from utilities import *

#%%
# cross-correlation of the audio from the gorpo and the robat 240 to match them in time

video_DIR = '/home/alberto/Videos/GOPRO/2025-06-18/'
audio_DIR = './2025-06-18/'

amplitude = 0.5

if amplitude == 1:
    video_name = np.array(['GX010558.MP4','GX010559.MP4',]) # run1, amplitude 1
    audio_name = np.array(['MULTIWAV_2025-06-18__11-24-12.wav', 'MULTIWAV_2025-06-18__11-26-09.wav']) # run1, amplitude 1
    time_1m_mark = 10.4 # sec
    time_05m_mark = 20.5 # sec
elif amplitude == 0.5:
    audio_name = np.array(['MULTIWAV_2025-06-18__14-02-27.wav']) # run2, amplitude 0.5
    video_name = np.array(['GX010561.MP4']) # run2, amplitude 0.5
    time_1m_mark = 7.23 # sec
    time_05m_mark = 18.3 # sec
else: 
    raise ValueError('Amplitude not recognized, choose a suitable value')

fs = int(48e3)
total_array_spl = {}
for mic_num in range(5):
    # Load the video file
    video = VideoFileClip(video_DIR+video_name[0])

    # Extract audio
    gopro_fps = video.fps
    camera = video.audio
    camera_audio, sr = sf.read(audio_DIR + 'GX010561.wav')
    robot_audio, sr = sf.read(audio_DIR + audio_name[0])
    robot_audio = robot_audio[:, mic_num]

    # Design the highpass filter
    cutoff = 50 # cutoff frequency in Hz

    # Apply the filter
    sos = signal.butter(1, cutoff, 'hp', fs=fs, output='sos')
    robot_audio = signal.sosfilt(sos, robot_audio)

    xcorr = np.roll(signal.correlate(camera_audio, robot_audio, mode='same'), -len(robot_audio) // 2)
    index = np.argmax(np.abs(xcorr))
    start_frame = int(index / sr * gopro_fps)
    time_offset = index / sr
    print('Start frame: %d' % start_frame, 'at time %.2f seconds' % (index / sr))

    delay = np.zeros(int(time_offset*sr))
    robot_audio_delayed = np.concatenate((delay, robot_audio))

    time_camera = np.arange(len(camera_audio)) / sr
    time_robot = np.arange(len(robot_audio_delayed)) / sr
    time_xcorr = np.arange(len(xcorr)) / sr

    # result plots
    plt.figure(figsize=(15, 15),)
    plt.yticks([])

    plt.subplot(3, 1, 1, sharex=plt.gca())
    plt.plot(time_camera, camera_audio)
    plt.title('Camera Audio')
    plt.ylabel('Amplitude')
    plt.grid()

    plt.subplot(3, 1, 2, sharex=plt.gca())
    plt.plot(time_robot, robot_audio_delayed)
    plt.axvline(time_1m_mark, color = 'c', linestyle='-', label='1m Mark')
    plt.axvline(time_05m_mark, color='orange', linestyle='-', label='0.5m Mark')
    plt.title('Robot Audio')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3, sharex=plt.gca())
    plt.plot(time_xcorr, xcorr)
    plt.axvline(index / sr, color='r', linestyle='--', label='Start Frame')

    plt.title('Cross-correlation')
    plt.xlabel('Time [s]')
    plt.ylabel('Correlation')
    plt.legend()
    plt.grid()

    plt.show(block=False)

    # %%
    robot_audio_trim = robot_audio_delayed[int((time_1m_mark-0.1)*fs):int((time_05m_mark+0.1)*fs)]

    plt.figure()
    plt.plot(np.arange(0,len(robot_audio_trim))/fs,robot_audio_trim)
    plt.title('Trimmed Robot Audio')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid()

    # %%
    # match filtering and peak finding to get the peak time positions

    # Define the matched filter function
    def matched_filter(recording, chirp_template):
        filtered_output = np.roll(signal.correlate(recording, chirp_template, 'same', method='direct'), -len(chirp_template)//2)
        filtered_envelope = np.abs(signal.hilbert(filtered_output))
        return filtered_envelope

    # Detect peaks in the matched filter output
    def detect_peaks(filtered_output, sample_rate):
        peaks, properties = signal.find_peaks(filtered_output, prominence=0.2, distance=0.7 * sample_rate)
        return peaks

    output_file = audio_DIR + '01_24k_1sweeps_4ms_amp08_48k.wav'
    chirp_len = 0.004  # 4 ms
    chirp_template, fs_chirp = sf.read(output_file)
    chirp_template = chirp_template[int(0.1*fs_chirp): int((0.1+chirp_len)*fs_chirp)] 

    plt.plot(np.arange(0, len(chirp_template)) / fs_chirp, chirp_template)
    plt.grid()
    plt.title('Chirp Template')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    robot_audio_matched = matched_filter(robot_audio_trim, chirp_template)

    peaks = detect_peaks(robot_audio_matched, fs)

    print(f"Detected peaks: gras = {len(peaks)}")

    # plot the peaks
    plt.figure(figsize=(15, 5))
    plt.subplot(2, 1, 1)
    plt.plot(np.linspace(0, len(robot_audio_matched) / fs, len(robot_audio_matched)), robot_audio_matched)
    plt.plot(peaks/fs, robot_audio_matched[peaks], 'ro')
    plt.title('Matched Filter Output')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.grid()
    plt.show(block=False)

    # %%
    # cut the audio around the peaks to analyze the SPL

    # peak at 1 m mark
    robot_audio_trim_1m = robot_audio_trim[(peaks[0]): int(peaks[0] + chirp_len*fs_chirp)]
    rms_1m = rms(robot_audio_trim_1m)

    # Plot the trimmed audio at 1m mark
    plt.figure()
    plt.plot(np.arange(len(robot_audio_trim_1m)) / fs, robot_audio_trim_1m)
    plt.title('Trimmed Audio at 1m Mark')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show(block=False)

        
    # peak at 0.5 m mark
    plt.figure()
    robot_audio_trim_05m = robot_audio_trim[(peaks[-1]): int(peaks[-1] + chirp_len*fs_chirp)]
    rms_05m = rms(robot_audio_trim_05m)

    # Plot the trimmed audio at 0.5m mark
    plt.plot(np.arange(len(robot_audio_trim_05m)) / fs, robot_audio_trim_05m)
    plt.xlabel('Time [ms]')
    plt.title('Trimmed Audio at 0.5m Mark')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show(block=False)

    # Analyze all detected peaks in the audio
    window_length = int(chirp_len * fs_chirp)  # 4 ms window
    audio_segments = []
    rms_values = []

    plt.figure(figsize=(12, 6))
    for i, peak in enumerate(peaks):
        start = peak
        end = peak + window_length
        segment = robot_audio_trim[start:end]
        audio_segments.append(segment)
        rms_val = rms(segment)
        rms_values.append(rms_val)

    # print("RMS values for each detected peak:")
    # for i, val in enumerate(rms_values):
    #     print(f"Peak {i+1}: RMS = {val}")

    # For compatibility with later code, keep the first and last segments as 1m and 0.5m marks
    robot_audio_trim_1m = audio_segments[0]
    robot_audio_trim_05m = audio_segments[-1]
    rms_1m = rms_values[0]
    rms_05m = rms_values[-1]
    
    # %%
    # upload array sensitivity
    sensitivity_path = '/home/alberto/Documents/ActiveSensingCollectives_lab/Ro-BATs/measurements/z723/array_calibration/226_238/Knowles_SPH0645LM4H-B_sensitivity.csv'
    sensitivity = pd.read_csv(sensitivity_path)
    freqs = np.array(sensitivity.iloc[:, 0])  # first column contains frequencies
    sens_freqwise_rms = np.array(sensitivity.iloc[:, 1])  # Last column contains sensitivity values 

    # Plot the uploaded sensitivity
    plt.figure(figsize=(15, 8))
    a0 = plt.subplot(211)
    plt.plot(freqs, sens_freqwise_rms)
    plt.xlabel('Frequencies, Hz', fontsize=12);
    plt.ylabel('dB a.u. rms/Pa', fontsize=12)
    plt.title(f'Knowles SPH0645LM4H-B Sensitivity', fontsize=20)
    plt.grid()
    plt.xticks(np.linspace(1000, 20000, 20), rotation=45)

    plt.subplot(212, sharex=a0)
    plt.plot(freqs, dB(sens_freqwise_rms))
    plt.xlabel('Frequencies, Hz', fontsize=12);
    plt.ylabel('dB a.u. rms/Pa', fontsize=12)
    plt.grid()
    plt.xticks(np.linspace(1000, 20000, 20), rotation=45)
    plt.tight_layout()
    plt.show(block=False)

    recsound_centrefreqs_1m, freqwise_rms_1m = calc_native_freqwise_rms(robot_audio_trim_1m, fs)
    recsound_centrefreqs_05m, freqwise_rms_05m = calc_native_freqwise_rms(robot_audio_trim_05m, fs)
    
    # Loop over all audio segments to compute SPL for each detected peak
    interp_sensitivity_1m = interpolate_freq_response([freqs, sens_freqwise_rms],
                            recsound_centrefreqs_1m)
    freqwise_Parms_1m = freqwise_rms_1m/interp_sensitivity_1m # go from rms to Pa(rmseq.)
    freqwiese_dbspl_1m = pascal_to_dbspl(freqwise_Parms_1m)

    interp_sensitivity_05m = interpolate_freq_response([freqs, sens_freqwise_rms],
                            recsound_centrefreqs_05m)
    freqwise_Parms_05m = freqwise_rms_05m/interp_sensitivity_05m # go from rms to Pa(rmseq.)
    freqwiese_dbspl_05m = pascal_to_dbspl(freqwise_Parms_05m)

    frequency_band = [0.2e3, 24e3] # min, max frequency to do the compensation Hz

    # Choose to calculate the dB SPL only for the frequency range of interest.
    # Target mic
    tgtmic_relevant_freqs = np.logical_and(recsound_centrefreqs_1m>=frequency_band[0],
                                    recsound_centrefreqs_1m<=frequency_band[1])
    total_rms_freqwise_Parms_1m = np.sqrt(np.sum(freqwise_Parms_1m[tgtmic_relevant_freqs]**2))

    tgtmic_relevant_freqs = np.logical_and(recsound_centrefreqs_05m>=frequency_band[0],
                                    recsound_centrefreqs_05m<=frequency_band[1])
    total_rms_freqwise_Parms_05m = np.sqrt(np.sum(freqwise_Parms_05m[tgtmic_relevant_freqs]**2))
    print(f'MICROPHONE: {mic_num+1}')
    print(f"print(f'SPH0645 dBrms SPL measure at 1m : {pascal_to_dbspl(total_rms_freqwise_Parms_1m)}")
    print(f"print(f'SPH0645 dBrms SPL measure at 0.5m: {pascal_to_dbspl(total_rms_freqwise_Parms_05m)}")
    print(f"SPH0645 dBrms SPL diff {-pascal_to_dbspl(total_rms_freqwise_Parms_1m)+pascal_to_dbspl(total_rms_freqwise_Parms_05m)} dB")
    # %%

    spl_values = []
    for i, segment in enumerate(audio_segments):
        # Calculate freqwise RMS for the segment
        centrefreqs, freqwise_rms = calc_native_freqwise_rms(segment, fs)
        # Interpolate sensitivity
        interp_sensitivity = interpolate_freq_response([freqs, sens_freqwise_rms], centrefreqs)
        # Convert to Pascals
        freqwise_Parms = freqwise_rms / interp_sensitivity
        # Convert to dB SPL
        freqwise_dbspl = pascal_to_dbspl(freqwise_Parms)
        # Limit to frequency band of interest
        relevant_freqs = np.logical_and(centrefreqs >= frequency_band[0], centrefreqs <= frequency_band[1])
        total_rms_freqwise_Parms = np.sqrt(np.sum(freqwise_Parms[relevant_freqs] ** 2))
        spl = pascal_to_dbspl(total_rms_freqwise_Parms)
        spl_values.append(spl)
    total_array_spl[mic_num] = spl_values


# %%
robot_speed = 1/(np.shape(robot_audio_trim)[0]/fs)  # v=s/t
# print(f"Robot speed: {robot_speed} m/s")

time_peaks = peaks / fs
# print(f'time_peaks: {time_peaks}')

# Distribute distances linearly between 1m and 0.5m for the detected peaks
distances = np.linspace(1.0, 0.5, len(peaks))
plt.close('all')

plt.figure(figsize=(10, 5))
for mic_num in range(5):
    plt.plot(distances, total_array_spl[mic_num], 'o-', label='Mic ' + str(mic_num + 1))
    plt.xlabel('Distance [m]')
    plt.xticks(distances, labels=[f"{d:.3f}" for d in distances])
    plt.ylabel('dB SPL')
    flattened_spl_values = [value for sublist in total_array_spl.values() for value in sublist]
    plt.yticks(np.arange(np.min(flattened_spl_values), np.max(flattened_spl_values), 1))
    plt.title(f'SPL vs Distance for Detected Peaks, amplitude {amplitude}')
    plt.grid()
    plt.legend()

plt.savefig(f'./2025-06-18/figures/SPL_vs_Distance_amplitude_{amplitude}_robat_240.png', dpi=600, bbox_inches='tight')
plt.show()
# %%
