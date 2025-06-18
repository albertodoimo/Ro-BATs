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



#%%
# cross-correlation of the audio from the gorpo and the robat 240 to match them in time

video_DIR = '/home/alberto/Videos/GOPRO/2025-06-18/'
audio_DIR = './2025-06-18/'

amplitude = 0.5

time_1m_mark = 8.23 # sec
time_05m_mark = 18.3 # sec


if amplitude == 1:
    video_name = np.array(['GX010558.MP4','GX010559.MP4',]) # run1, amplitude 1
    audio_name = np.array(['MULTIWAV_2025-06-18__11-24-12.wav', 'MULTIWAV_2025-06-18__11-26-09.wav']) # run1, amplitude 1
elif amplitude == 0.5:
    audio_name = np.array(['MULTIWAV_2025-06-18__14-02-27.wav']) # run2, amplitude 0.5
    video_name = np.array(['GX010561.MP4']) # run2, amplitude 0.5
else: 
    raise ValueError('Amplitude not recognized, choose a suitable value')

fs = int(48e3)

# Load the video file
video = VideoFileClip(video_DIR+video_name[0])

# Extract audio
gopro_fps = video.fps
camera = video.audio
camera_audio, sr = sf.read(audio_DIR + 'GX010561.wav')
robot_audio, sr = sf.read(audio_DIR + audio_name[0])
robot_audio = robot_audio[:, 2]

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
robot_audio = np.concatenate((delay, robot_audio))

time_camera = np.arange(len(camera_audio)) / sr
time_robot = np.arange(len(robot_audio)) / sr
time_xcorr = np.arange(len(xcorr)) / sr

# result plots
plt.figure(figsize=(15, 15),)

plt.subplot(3, 1, 1, sharex=plt.gca())
plt.plot(time_camera, camera_audio)
plt.title('Camera Audio')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(3, 1, 2, sharex=plt.gca())
plt.plot(time_robot, robot_audio)
plt.axvline(time_1m_mark, color = 'c', linestyle='-', label='1m Mark')
plt.axvline(time_05m_mark, color='b', linestyle='-', label='0.5m Mark')
plt.title('Robot Audio')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(3, 1, 3, sharex=plt.gca())
plt.plot(time_xcorr, xcorr)
plt.axvline(index / sr, color='r', linestyle='--', label='Start Frame')

plt.title('Cross-correlation')
plt.xlabel('Time [s]')
plt.ylabel('Correlation')
plt.legend()
plt.grid()

plt.tight_layout()

plt.show()

# %%
robot_audio_trim = robot_audio[int((time_1m_mark-0.1)*fs):int((time_05m_mark+0.1)*fs)]

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
chirp_template, fs_chirp = sf.read(output_file)
chirp_template = chirp_template[int(0.1*fs_chirp): int(0.104*fs_chirp)] 

plt.plot(np.arange(0, len(chirp_template)) / fs_chirp, chirp_template)
robot_audio_matched = matched_filter(robot_audio_trim, chirp_template)

peaks = detect_peaks(robot_audio_matched, fs)

print(f"Detected peaks: gras = {len(peaks)}")

# plot the peaks
plt.figure(figsize=(15, 5))
plt.subplot(2, 1, 1)
plt.plot(np.linspace(0, len(robot_audio_matched) / fs, len(robot_audio_matched)), robot_audio_matched)
plt.plot(peaks/fs, robot_audio_matched[peaks], 'ro')
plt.title('Matched Filter Output - GRAS')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# %%
# cut the audio around the peaks to analyze the SPL

def rms(X):
    return np.sqrt(np.mean(X**2))

dB = lambda X: 20*np.log10(abs(np.array(X).flatten()))

db_to_linear = lambda X: 10**(X/20)

# peak at 1 m mark
robot_audio_trim_1m = robot_audio_trim[(peaks[0]): int(peaks[0] + 0.004*fs_chirp)]
rms_1m = rms(robot_audio_trim_1m)

plt.plot(robot_audio_trim_1m)
plt.show()
    
# peak at 0.5 m mark
plt.figure()
robot_audio_trim_05m = robot_audio_trim[(peaks[-1]): int(peaks[-1] + 0.004*fs_chirp)]
rms_05m = rms(robot_audio_trim_05m)

plt.plot(robot_audio_trim_05m)
plt.show()

# %%
# calculate the sound intensity in dB of the audio recordings

dB_difference = -20*np.log10(rms_1m/rms_05m)
print(f"RMS at 1m: {dB(rms_1m)}, RMS at 0.5m: {dB(rms_05m)}")
print(f"Difference in dB: {dB_difference:.2f} dB")

# %%
# upload array sensitivity 

def calc_native_freqwise_rms(X, fs):
    '''
    Converts the FFT spectrum into a band-wise rms output. 
    The frequency-resolution of the spectrum/audio size decides
    the frequency resolution in general. 
    
    Parameters
    ----------
    X : np.array
        Audio
    fs : int
        Sampling rate in Hz
    
    Returns 
    -------
    fftfreqs, freqwise_rms : np.array
        fftfreqs holds the frequency bins from the RFFT
        freqwise_rms is the RMS value of each frequency bin. 
    '''
    rfft = np.fft.rfft(X)
    fftfreqs = np.fft.rfftfreq(X.size, 1/fs)
    # now calculate the rms per frequency-band
    freqwise_rms = []
    for each in rfft:
        mean_sq_freq = np.sum(abs(each)**2)/rfft.size
        rms_freq = np.sqrt(mean_sq_freq/(2*rfft.size-1))
        freqwise_rms.append(rms_freq)
    return fftfreqs, freqwise_rms


# Make an interpolation function 
def interpolate_freq_response(mic_freq_response, new_freqs):
    ''' 
    Parameters
    ----------
    mic_freq_response : tuple/list
        A tuple/list with two entries: (centrefreqs, centrefreq_RMS).
        
    new_freqs : list/array-like
        A set of new centre frequencies that need to be interpolated to. 

    Returns 
    -------
    tgtmicsens_interp : 
        
    Attention
    ---------
    Any frequencies outside of the calibration range will automatically be 
    assigned to the lowest sensitivity values measured in the input centrefreqs
    
    '''
    centrefreqs, mic_sensitivity = mic_freq_response 
    tgtmic_sens_interpfn = interp1d(centrefreqs, mic_sensitivity,
                                    kind='cubic', bounds_error=False,
                                    fill_value=np.min(mic_sensitivity))
    # interpolate the sensitivity of the mic to intermediate freqs
    tgtmicsens_interp = tgtmic_sens_interpfn(new_freqs)
    return tgtmicsens_interp


sensitivity_path = '/home/alberto/Documents/ActiveSensingCollectives_lab/Ro-BATs/measurements/z723/array_calibration/226_238/Knowles_SPH0645LM4H-B_sensitivity.csv'

recsound_centrefreqs, freqwise_rms = calc_native_freqwise_rms(robot_audio_trim_1m, fs)

# mask = (recsound_centrefreqs >= start_f) & (recsound_centrefreqs <= end_f)
# idx_start = np.where(recsound_centrefreqs >= start_f)[0][0]
# idx_end = np.where(recsound_centrefreqs <= end_f)[0][-1]
# print("\nInfos about the frequencies considered in the analysis:")
# print(f"Index for starting freq = {start_f} Hz: {idx_start}, correspondent frequency band: {recsound_centrefreqs[idx_start]} Hz")
# print(f"Index for ending freq = {end_f} Hz: {idx_end}, correspondent frequency band: {recsound_centrefreqs[idx_end]} Hz")

# recsound_centrefreqs = recsound_centrefreqs[mask]
# freqwise_rms = freqwise_rms[idx_start:idx_end+1]

interp_sensitivity = interpolate_freq_response([SPH0645_centrefreqs, SPH0645_sensitivity],
                          recsound_centrefreqs)
freqwise_Parms = freqwise_rms/interp_sensitivity # go from rms to Pa(rmseq.)
