
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
sd.default.channels = channels

tone_durn = 100e-3 # seconds
t_tone = np.linspace(0, tone_durn, int(fs*tone_durn))
chirp = signal.chirp(t_tone, 30e3, t_tone[-1], 40e3)
chirp *= signal.windows.hann(chirp.size)
plt.plot(chirp)
output_chirp = np.concatenate((chirp, np.zeros((int(fs*0.2)))))
output_tone_stereo = np.float32(np.column_stack((output_chirp, output_chirp)))

all_input_audio = sd.playrec(chirp, fs, channels=2)
print(sd.get_status())
print('all_input_audio',all_input_audio)
print(np.shape(all_input_audio))
plt.plot(all_input_audio[0])
plt.show()

# 
# i = 0
# def callback_sine(indata, outdata, frames, time, status):
#     # This function is a callback function for the PyAudio stream
#     # It takes in input data (indata) and outputs a sine wave tone (output_tone_stereo) to the output data (outdata)
#     # It also stores the input data in a queue (all_input_data) for further processing
#     outdata[:] = output_tone_stereo
#     all_input_data.put(indata)
# 
# 

#define the input signals features
# Sin = sd.InputStream (samplerate=fs, blocksize=block_size,channels=channels,device=usb_fireface_index)
# print('Sin fs = ', Sin.samplerate)
# print('Sin blocksize = ', Sin.blocksize)
# print('Sin channels = ', Sin.channels)
# print('Sin latency = ', Sin.latency)
# print('Sin devinfo = ', Sin.device)

# print('audio stream initialized\n')
# stream_inout = sd.Stream(samplerate=fs,
#                          blocksize=output_chirp.shape[0],
#                          device=(usb_fireface_index,usb_fireface_index),
#                          #channels=(1,2),
#                          channels=(6,2),
#                          callback=callback_sine)
# 
# all_input_data = queue.Queue()  # This creates a new Queue object from the queue module and assigns 
# # it to the variable all_input_data this queue will be used to store input data for processing
# 

# start_time = stream_inout.time
# 
# # run the stream for <= 2 seconds
# with stream_inout:  # This opens a context manager for the stream_inout object
#     # This loop runs for 1 second after the stream is started
#     while (stream_inout.time - start_time) <=1:
#         pass


#  
#  # for raspberry longest delay:
#  # Sin.start()
#  sd.play(output_chirp, fs)
#  Sin.start()
#  
#  in_sig = Sin.read(Sin.blocksize*3)
#  rec = in_sig
#  rec = rec[0]
#  
#  #print(np.shape(chirp))
#  print('\n',np.transpose(chirp))
#  # chirp = np.pad(np.transpose(chirp), (0, Sin.blocksize - len(chirp)), mode='constant')
#  #plt.plot(chirp)
#  #plt.show()
#  print('\n padded data = ',chirp)
#  
#  print('\n')
#  print('shape data=',np.shape(chirp))
#  cc = np.correlate(rec[:,2],chirp,'same')
#  
#  plt.plot(cc)
#  plt.show()
#  #midpoint = cc.size/2.0
#  delay = np.argmax(cc) - int(len(chirp)/2)
#  # convert delay to seconds
#  delay *= 1/float(fs)
#  print('delay = ',delay)
#  
#  plt.figure()
#  aa = plt.subplot(211)
#  # plt.specgram(rec[:,3], Fs=fs, NFFT=1024, noverlap=512)   
#  plt.specgram(rec[:,3], Fs=fs, NFFT=512, noverlap=256,cmap='viridis')
#  plt.ylim((500,fs/2))
#  plt.subplot(212, sharex=aa)
#  t_audio = np.linspace(0, rec.shape[0]/fs, rec.shape[0])
#  plt.plot(t_audio, input_audio[:,0])
#  plt.plot(t_audio, rec[:,3])
#  # plt.subplot(313)
#  # plt.plot(rec[:,2])
#  plt.show()
#  
#  

# 
# # load the input audio 
# all_input_audio = []
# while not all_input_data.empty():
#     all_input_audio.append(all_input_data.get())            
# input_audio = np.concatenate(all_input_audio)
# 
plt.figure()
aa = plt.subplot(211)
# plt.specgram(input_audio[:,0], Fs=fs, NFFT=1024, noverlap=512)   
plt.specgram(all_input_audio[:,0], Fs=fs, NFFT=1024, noverlap=512)    
plt.subplot(212, sharex=aa)
t_audio = np.linspace(0, all_input_audio.shape[0]/fs, all_input_audio.shape[0])
# plt.plot(t_audio, input_audio[:,0])
plt.plot(t_audio, all_input_audio[:,0])
plt.show()

