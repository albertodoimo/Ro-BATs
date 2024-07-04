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
fs = 96000  # sampling frequency
nfft = 256  # FFT size
mic_spacing = 0.003 
channels = 8
block_size  = 4096
freq_range = [20, 40000]

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
# # Initialize buffers for recording
# input_buffer = []
# output_buffer = []
# j = 0
# # Callback function
# def audio_callback(indata, outdata, frames, time, status):
#     print('3')
#     input_buffer.append(indata.copy())
#     output_buffer.append(outdata.copy())
# 
# Start the audio stream

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

print('1')

initialization()

def update_polar():
    # Your streaming data source logic goes here

    in_sig,status = S.read(S.blocksize)

    #print('input=', in_sig)
    #print('input=', np.shape(in_sig))
    #input_buf = np.concatenate(in_sig)

    # print('\n input buf  shape = ',np.shape(input_buffer))
    input_audio = np.transpose(in_sig)

    #print('signals =', np.shape(input_audio))
    #print('signals =', input_audio)

    X = pra.transform.stft.analysis(input_audio.T, nfft, nfft // 2)
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
    # line.set_ydata(values)

    # Update the polar plot
    phi_plt = doa.grid.azimuth
    return spatial_resp, phi_plt
    #return spatial_resp

for i in range(50):
    update_polar()
   
    base = 1.
    height = 10.
    true_col = [0, 0, 0]

    spatial_resp, phi_plt = update_polar()
    fig = plt.figure()
    algo_name = 'MUSIC'


    # plot
    ax = fig.add_subplot(111, projection='polar')
    c_phi_plt = np.r_[phi_plt, phi_plt[0]]
    c_dirty_img = np.r_[spatial_resp, spatial_resp[0]]
    ax.plot(c_phi_plt, base + height * c_dirty_img, linewidth=2,
            alpha=0.55, linestyle='-', 
            label="spatial \n spectrum")
    plt.title(algo_name, fontdict={'fontsize': 15}, loc='left')

    plt.legend()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8, bbox_to_anchor=(1.5, 0.6))
    # ax.legend(handles, labels, framealpha=0.5,
    #           scatterpoints=1, loc='upper center', fontsize=10,
    #           ncol=1, bbox_to_anchor=(1.6, 0.5),
    #           handletextpad=.2, columnspacing=1.7, labelspacing=0.1)

    ax.set_xticks(np.linspace(0, 2 * np.pi, num=12, endpoint=False))
    ax.xaxis.set_label_coords(0.5, -0.11)
    ax.set_yticks(np.linspace(0, 1, 2))
    # ax.xaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle=':')
    # ax.yaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle='--')
    ax.set_ylim([0, 1.05 * (base + height)])
    plt.show()


# # Set up the polar plot
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# theta = np.linspace(-np.pi/2, np.pi/2, 360)
# values = np.random.rand(180)
# line, = ax.plot(theta, values)
# #ax.set_thetamin(-90)
# #ax.set_thetamax(90)
# 
# # Set up the animation
# ani = FuncAnimation(fig, update_polar, frames=range(360), blit=True)
# 
# plt.show()


