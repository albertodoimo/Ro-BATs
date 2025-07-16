import os
import sys
import io
import ffmpeg
import numpy as np
import soundfile as sf
import librosa
from sonar import sonar
from scipy import signal
from das_v2 import das_filter
from capon import capon_method
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

def pow_two_pad_and_window(vec, show=False):
    window = signal.windows.tukey(len(vec), alpha=0.3)
    windowed_vec = vec * window
    padded_windowed_vec = np.pad(windowed_vec, (0, 2**int(np.ceil(np.log2(len(windowed_vec)))) - len(windowed_vec)))
    if show:
        dur = len(padded_windowed_vec) / fs
        t = np.linspace(0, dur, len(padded_windowed_vec))
        plt.figure()
        plt.plot(t, padded_windowed_vec)
        plt.show()
    return padded_windowed_vec/max(padded_windowed_vec)*0.8

os.chdir(os.path.dirname(os.path.abspath(__file__)))

offsets = np.load('./offsets/20250514_17-20-16_offsets.npy')

camera_path = './videos/GX010521.mp4'
robot_path = './audio/20250514_17-20-16.wav'
video_fps = 60
try:
    robot_audio, fs = sf.read(robot_path)
    robot_audio = robot_audio[:, 0]
    print( 'Robot audio duration: %.1f [s]' % (len(robot_audio)/fs))
    # Run ffmpeg to extract audio and pipe as WAV
    out, _ = (
        ffmpeg
        .input(camera_path)
        .output('pipe:', format='wav', acodec='pcm_s16le')
        .run(capture_stdout=True, capture_stderr=True)
    )

    # Load audio from bytes using soundfile
    camera_audio, sr = librosa.load(io.BytesIO(out), sr=fs, mono=True)
    print( 'Camera audio duration: %.1f [s]' % (len(camera_audio)/fs))

    xcorr = np.roll(signal.correlate(camera_audio, robot_audio, mode='same'), -len(robot_audio) // 2)
    index = np.argmax(np.abs(xcorr))
    start_frame = int(index / sr * video_fps)
    print('Detected frame: %d' % start_frame)
    fs = 176400
    dur = 3e-3
    hi_freq = 60e3
    low_freq = 20e3
    output_threshold = -50 # [dB]
    distance_threshold = 20 # [cm]

    METHOD = 'das' # 'das', 'capon'
    if METHOD == 'das':
        spatial_filter = das_filter
    elif METHOD == 'capon':
        spatial_filter = capon_method

    t_tone = np.linspace(0, dur, int(fs*dur))
    chirp = signal.chirp(t_tone, hi_freq, t_tone[-1], low_freq)    
    sig = pow_two_pad_and_window(chirp)

    C_AIR = 343
    min_distance = 10e-2
    discarded_samples = int(np.floor(((min_distance + 2.5e-2)*2)/C_AIR*fs))
    max_distance = 1
    max_index = int(np.floor(((max_distance + 2.5e-2)*2)/C_AIR*fs))
    def update(frame):
        # print(curr_end/fs)
        audio_data, _ = sf.read(robot_path, start=offsets[frame, 0], frames=offsets[frame, 1])
        video_frame = int(offsets[frame, 0] / fs * video_fps) + start_frame
        print('Video frame: %d' % video_frame)  
        dB_rms = 20*np.log10(np.mean(np.std(audio_data, axis=0)))    
        if dB_rms > output_threshold:
            filtered_signals = signal.correlate(audio_data, np.reshape(sig, (-1, 1)), 'same', method='fft')
            roll_filt_sigs = np.roll(filtered_signals, -len(sig)//2, axis=0)
            
            try:
                distance, direct_path, obst_echo = sonar(roll_filt_sigs, discarded_samples, max_index, fs)
                distance = distance*100 # [m] to [cm]
                # print('\nDistance: %.1f [cm]' % distance)                             
                # if distance == 0:
                #     print('\nNo Obstacles')
                theta, p = spatial_filter(
                                            roll_filt_sigs[obst_echo - int(5e-4*fs):obst_echo + int(5e-4*fs)], 
                                            fs=fs, nch=roll_filt_sigs.shape[1], d=2.70e-3, 
                                            bw=(low_freq, hi_freq)
                                        )
                p_dB = 10*np.log10(p)
                
                if direct_path != obst_echo:
                    doa_index = np.argmax(p_dB)
                    theta_hat = theta[doa_index]
                    if distance > 0:
                        # print('\nDistance: %.1f [cm] | DoA: %.2f [deg]' % (distance, theta_hat))            
                        line.set_ydata(p_dB)
                        ax.set_ylim(min(p_dB), max(p_dB) + 6)
                        vline.set_xdata([np.deg2rad(theta_hat)])
                        
                        title.set_text('%.1f [cm], %.1f [s]' % (distance, offsets[frame, 0]/fs))
                return line, vline
            except ValueError:
                print('\nNo valid distance or DoA')

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    title = ax.set_title('')
    # Shift axes by -90 degrees
    ax.set_theta_offset(np.pi/2)
    # Limit theta between -90 and 90 degrees
    ax.set_xlim(-np.pi/2, np.pi/2)
    ax.set_ylim(-20, 40)        
    ax.grid(False)
    line = ax.plot(np.linspace(-np.pi/2, np.pi/2, 73), 0*np.sin(np.linspace(-np.pi/2, np.pi/2, 73)))[0]
    vline = ax.axvline(0, 0, 30, color='red', linestyle='--')
    # txt = plt.text(0, 0, '', ha='center', va='center', fontsize=12)
    ani = FuncAnimation(fig, update,  frames=len(offsets), interval=17, cache_frame_data=True, repeat=False)
    plt.show()

except ffmpeg.Error as e:
    print('ffmpeg error:', e.stderr.decode(), file=sys.stderr)
    sys.exit(1)