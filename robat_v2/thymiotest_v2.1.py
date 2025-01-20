
print('import libraries...')
print('...')

import argparse
import threading
import time
import math

# import pyqtgraph as pg
# import pyqtgraph.opengl as gl
# from pyqtgraph.Qt import QtCore
import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

print('libraries imported\n')

print('initializating audio stream...')
print('...')

import queue
input_audio_queue = queue.Queue()

def get_card(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return i

usb_fireface_index = get_card(sd.query_devices())
print(sd.query_devices())
print('usb_fireface_index=',usb_fireface_index)

fs = 96000
block_size = 4096
channels = 6
mic_spacing = 0.018 #m

#define the input signals features
Sin = sd.InputStream (samplerate=fs, blocksize=block_size,channels=channels,device=usb_fireface_index)
print('Sin fs = ', Sin.samplerate)
print('Sin blocksize = ', Sin.blocksize)
print('Sin channels = ', Sin.channels)
print('Sin latency = ', Sin.latency)
print('Sin devinfo = ', Sin.device)

print('audio stream initialized\n')

tone_durn = 5e-3 # seconds
t_tone = np.linspace(0, tone_durn, int(fs*tone_durn))
chirp = signal.chirp(t_tone, 18e3, t_tone[-1], 0.5e3)
chirp *= signal.windows.hann(chirp.size)
output_chirp = np.concatenate((chirp, np.zeros((int(fs*0.2)))))
output_tone_stereo = np.float32(np.column_stack((output_chirp, output_chirp)))
#print(output_tone_stereo)
#print(np.shape(output_tone_stereo))


# for raspberry longest delay:
# Sin.start()
sd.play(output_chirp, fs)
Sin.start()

in_sig = Sin.read(Sin.blocksize*2)

rec = in_sig
rec = rec[0]

plt.figure()
aa = plt.subplot(211)
# plt.specgram(rec[:,3], Fs=fs, NFFT=1024, noverlap=512)   
plt.specgram(rec[:,3], Fs=fs, NFFT=512, noverlap=256,cmap='viridis')
plt.ylim((500,fs/2))
plt.subplot(212, sharex=aa)
t_audio = np.linspace(0, rec.shape[0]/fs, rec.shape[0])
# plt.plot(t_audio, input_audio[:,0])
plt.plot(t_audio, rec[:,3])
# plt.subplot(313)
# plt.plot(rec[:,2])
plt.show()


