# `pyroomacoustics` demo

# Let's begin by importing the necessary libraries all of which can be installed with `pip`, even `pyroomacoustics`!
import numpy as np
import matplotlib.pyplot as plt
import natsort
import glob
# from scipy.io import wavfile
# from scipy.signal import fftconvolve
import IPython
import pyroomacoustics as pra
import scipy.signal as signal 
import sounddevice as sd

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Construct the new DOA object
        # the max_four parameter is necessary for FRIDA only

c = 343.    # speed of sound
fs = 48000  # sampling frequency
nfft = 512  # FFT size
mic_spacing = 0.003 
channels = 8
block_size  = 4096
freq_range = [10000, 20000]

echo = pra.linear_2D_array(center=[(channels-1)*mic_spacing//2,0], M=channels, phi=0, d=mic_spacing)
# The DOA algorithms require an STFT input, which we will compute for overlapping frames for our 1 second duration signal.

def get_card(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return i

usb_fireface_index = get_card(sd.query_devices())
print(sd.query_devices())
print('usb_fireface_index=',usb_fireface_index)
# 

S = sd.InputStream(samplerate=fs,blocksize=block_size, device=usb_fireface_index, channels=channels, latency='low')
            
def initialization():
    try:
        with S.start():
            for _ in range(1):
                sd.sleep(int(1000))

    except KeyboardInterrupt:
        print("\nStream stopped by user")
    except Exception as e:
        print(f"An error occurred: {e}")


initialization()

def update_polar(frame):
    # Your streaming data source logic goes here

    in_sig,status = S.read(S.blocksize)

    input_audio = np.transpose(in_sig)

    X = pra.transform.stft.analysis(input_audio.T, nfft, nfft // 2)
    X = X.transpose([2, 1, 0])

    doa = pra.doa.algorithms['MUSIC'](echo, fs, nfft, c=c, num_src=6)
    doa.locate_sources(X, freq_range=freq_range)
    print(doa.azimuth_recon * 180 / np.pi) #degrees 

    spatial_resp = doa.grid.values
        
    # normalize   
    min_val = spatial_resp.min()
    max_val = spatial_resp.max()
    spatial_resp = (spatial_resp - min_val) / (max_val - min_val)


    values = np.zeros(360)

    # Update the polar plot
    data.append(values)

    if len(data) > 360:  # Keep only the latest 360 values
        data.pop(0)

    line.set_ydata(spatial_resp)
    return line,


fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

theta = np.linspace(-np.pi,np.pi, 360)
values = np.random.rand(360)
line, =ax.plot(theta, values)
data = []
# Set up the animation
ani = FuncAnimation(fig, update_polar, frames=range(360), blit=False, interval= 50)

plt.show()
