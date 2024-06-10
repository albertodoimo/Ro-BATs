# -*- coding: utf-8 -*-
"""
Short example to show how to test ADC-DAC latency with sounddevice
==================================================================



Notes
-----
This example uses a queue object to hold audio buffers, and unless you run the entire code chunk, 
the queue object will be empty and the results spectrogr

Created on Wed May 15 06:29:04 2024

@author: theja
"""
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np 
import scipy.signal as signal 
import queue

#%%
tone_durn = 50e-3 # seconds
fs = 44100
t_tone = np.linspace(0, tone_durn, int(fs*tone_durn))
chirp = signal.chirp(t_tone, 18e3, t_tone[-1], 0.5e3)
chirp *= signal.windows.hann(chirp.size)
output_chirp = np.concatenate((chirp, np.zeros((int(fs*0.2)))))

output_tone_stereo = np.float32(np.column_stack((output_chirp, output_chirp)))

# define your sound device (choose the one with 2 in, 2 out)
sd.default.samplerate = fs
i = 0
def callback_sine(indata, outdata, frames, time, status):
    outdata[:] = output_tone_stereo
    all_input_data.put(indata)

# device IDs, you'll need to check the device ID with sd.query_devices()
# and give the input/output device numbers. 

# create a stream, and set the blocksize to match that of the output signal
stream_inout = sd.Stream(samplerate=fs,
                         blocksize=output_chirp.shape[0],
                         device=(4,6),
                         channels=(2,2),
                         callback=callback_sine)

# start the stream 
all_input_data = queue.Queue()
start_time = stream_inout.time

# run the stream for <= 2 seconds
with stream_inout:
    while (stream_inout.time - start_time) <=3:
        pass

# load the input audio 
all_input_audio = []
while not all_input_data.empty():
    all_input_audio.append(all_input_data.get())            
input_audio = np.concatenate(all_input_audio)

#%%
plt.figure()
aa = plt.subplot(211)
plt.specgram(input_audio[:,0], Fs=fs, NFFT=1024, noverlap=512)    
plt.subplot(212, sharex=aa)
t_audio = np.linspace(0, input_audio.shape[0]/fs, input_audio.shape[0])
plt.plot(t_audio, input_audio[:,0])

