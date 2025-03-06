# `pyroomacoustics` demo
import datetime

now = datetime.datetime.now()
print("\nCurrent date and time: \n", now.strftime("%Y-%m-%d %H:%M:%S"))
print('')

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
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Construct the new DOA object
        # the max_four parameter is necessary for FRIDA only

c = 343.    # speed of sound
fs = 96000  # sampling frequency

mic_spacing = 0.018 #m
channels = 5
block_size  = 2048
nfft = block_size//4  # FFT size
print('\nframes per second = ',fs//block_size, '\n' )
freq_range = [2000, 10000]

# This line of code creates a linear 2D microphone array using the pyroomacoustics library.
echo = pra.linear_2D_array(center=[(channels-1)*mic_spacing//2,0], M=channels, phi=0, d=mic_spacing)
# Breaking it down:
# - pra.linear_2D_array() is a function from pyroomacoustics that creates a linear array of microphones
# - center=[0,(channels-1)*mic_spacing//2] sets the center of the array
#   The x-coordinate is 0, and the y-coordinate is calculated based on the number of channels and mic spacing
# - M=channels sets the number of microphones in the array to the value of the 'channels' variable
# - phi=0 sets the orientation of the array 
# - d=mic_spacing sets the distance between adjacent microphones

# This array configuration is likely used for direction of arrival (DOA) estimation or beamforming in acoustic signal processing.


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

S = sd.InputStream(samplerate=fs, blocksize=block_size, device=usb_fireface_index, channels=channels, latency='low')
            
def initialization():
    try:
        with S.start():
            for _ in range(1):
                sd.sleep(int(1000))

    except KeyboardInterrupt:
        print("\nStream stopped by user")
    except Exception as e:
        print(f"An error occurred: {e}")




def update_polar(frame):
    # Your streaming data source logic goes here
    global rec

    in_sig,status = S.read(S.blocksize)
    correction=np.mean(in_sig)
    print(correction)
    in_sig = in_sig-correction
    memory.append(in_sig)
    rec = np.concatenate(memory)
    #print('input audio plot = ', np.shape(rec))

    X = pra.transform.stft.analysis(in_sig, nfft, nfft // 2)
    X = X.transpose([2, 1, 0])

    doa = pra.doa.algorithms['MUSIC'](echo, fs, nfft, c=c, num_src=1)
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
    #print('input audio plot last = ', np.shape(rec))
    return line,


initialization()
memory = []
rec = []

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#ax.set_theta_zero_location('E')
ax.set_theta_direction(1)
#ax.set_thetamin(360)
#ax.set_thetamax(0)
#ax.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))
#ax.set_xticklabels(['0°', '30°', '60°', '90°', '120°', '150°', '180°', '210°', '240°', '270°', '300°', '330°'])
data = []

theta = np.linspace(0, 2*np.pi, 360)
values = np.random.rand(360)
line, = ax.plot(theta, values)

# Set up the animation
ani = FuncAnimation(fig, update_polar, frames=range(360), blit=False, interval= 10)

plt.show()
#print('input audio plot lastlast = ', np.shape(rec))

#sf.write(f'pra_realtime_visual_{now}.wav', rec, samplerate=fs)

