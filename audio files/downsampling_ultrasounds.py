
#%% CREATE AUDIBLE SOUNDS
#   (if fs decrese time increase proportionally: 
#   192 KHz / 4 = 48 KHz --> 15 sec * 4 = 60 sec
#   => 192000 * 15 = 48000 * 60 = 288000 samples)

import soundfile as sf
import numpy as np
import scipy.signal as signal

audiofile_name = 'video_synced10channel_first15sec_1529543496.WAV'

audio, fs = sf.read(audiofile_name)
print('fs=',fs)

sf.write('dowsampled_'+audiofile_name, audio, samplerate=fs//4)

# %% OTHER METHOD that acts also on the duration of the file

# Downsample or convert frequency if necessary
# For example, you can use signal.resample to downsample
audible_signal = signal.resample(audio, len(audio)//2)

# Normalize the signal
audible_signal /= np.max(np.abs(audible_signal))

sf.write('length_dowsampled_'+ audiofile_name, audible_signal, samplerate=fs//4)
