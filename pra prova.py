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
import soundfile as sf


# Construct the new DOA object
        # the max_four parameter is necessary for FRIDA only

c = 343.    # speed of sound
fs = 48000  # sampling frequency
nfft = 512  # FFT size
mic_spacing = 0.003 
channels = 8
block_size  = 4096
freq_range = [2000, 9000]

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


def audio_callback(indata, outdata, frames, time, status):
    input_buffer.append(indata.copy())
    print('input buf callback =', np.shape(input_buffer))

            
def initialization():
    try:
        with sd.Stream(callback=audio_callback,
                    samplerate=fs,
                    blocksize=block_size,
                    device=(usb_fireface_index,usb_fireface_index),
                    channels=channels):
            for _ in range(1):
                sd.sleep(int(1000))

    except KeyboardInterrupt:
        print("\nStream stopped by user")
    except Exception as e:
        print(f"An error occurred: {e}")



def update_polar(frame):
    sd.Stream(callback=audio_callback,
                    samplerate=fs,
                    blocksize=block_size,
                    device=(usb_fireface_index,usb_fireface_index),
                    channels=channels)
    sd.Stream.start()
    # Your streaming data source logic goes here
    input_buffer_1 = np.concatenate(input_buffer)
    print('input buffer shape =', np.shape(input_buffer_1))
    input_audio = np.array(input_buffer_1[-block_size:]) 
    print('input audio shape =', np.shape(input_audio))
    # sf.write('recordings/audio_data.wav', input_buf_av[1,:], samplerate=fs)

    X = pra.transform.stft.analysis(input_audio, nfft, nfft // 2)
    X = X.transpose([2, 1, 0])

    doa = pra.doa.algorithms['MUSIC'](echo, fs, nfft, c=c, num_src=2)
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


input_buffer = []
#S = sd.InputStream(samplerate=fs, blocksize=block_size, device=usb_fireface_index, channels=channels, latency='low', callback=audio_callback)

initialization()

# in_sig,status = S.read(S.blocksize)
#print('\n input buffer = ',np.shape(input_buffer))

#input_buf_av = np.concatenate(input_buffer)
#print('\n input buf  shape = ',np.shape(input_buf_av))

#input_audio_av = np.transpose(input_buf_av)
#print('\n input av = ',input_audio_av)
#print('\n input av shape = ',np.shape(input_audio_av))



#input_audio = np.transpose(in_sig)
#print('input audio = ', input_audio)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

#ax.set_theta_zero_location('E')
ax.set_theta_direction(1)
#ax.set_thetamin(360)
#ax.set_thetamax(0)
#ax.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))
#ax.set_xticklabels(['0°', '30°', '60°', '90°', '120°', '150°', '180°', '210°', '240°', '270°', '300°', '330°'])
data = []
#input_buf_av= []


theta = np.linspace(0, 2*np.pi, 360)
values = np.random.rand(360)
line, =ax.plot(theta, values)

# Set up the animation
ani = FuncAnimation(fig, update_polar, frames=range(360), blit=False, interval= 50)

plt.show()

