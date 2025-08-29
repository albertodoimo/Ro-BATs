import ffmpeg
import numpy as np
import librosa
import io
import soundfile as sf
from matplotlib import pyplot as plt
import sys
import os
from scipy import signal

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Define path to video
camera_path = './video_audio_elab/videos/20250516_15-25-28.mp4'
robot_path = './video_audio_elab/audio/20250516_15-25-28.wav'
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

    xcorr_unrolled = signal.correlate(camera_audio, robot_audio, mode='same')
    xcorr = np.roll(xcorr_unrolled, -len(robot_audio) // 2)
    index = np.argmax(np.abs(xcorr))

    frame = int(index / sr * video_fps)
    
    print('Detected frame: %d' % frame)
    robot_audio = np.append(np.zeros(index), robot_audio)
    t = np.linspace(0, len(xcorr) / sr, num=len(xcorr))
    t_robot = np.linspace(0, len(robot_audio) / fs, num=len(robot_audio))
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(t_robot, robot_audio/np.max(np.abs(robot_audio)))
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Amplitude')
    ax1.grid()
    ax2.plot(t, camera_audio/np.max(np.abs(camera_audio)))
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Amplitude')
    ax2.grid() 
    ax3.plot(t, xcorr/np.max(np.abs(xcorr)))
    ax1.axvline(x=index/sr, color='r', linestyle='--', label='Detected delay')
    ax2.axvline(x=index/sr, color='r', linestyle='--', label='Detected delay')        
    ax3.axvline(x=index/sr, color='r', linestyle='--', label='Detected delay')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Amplitude')
    ax3.grid()
    plt.tight_layout()
    plt.show()
except ffmpeg.Error as e:
    print('ffmpeg error:', e.stderr.decode(), file=sys.stderr)
    sys.exit(1)