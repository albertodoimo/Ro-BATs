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
from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
)
dir_obj = CardioidFamily(
    orientation = DirectionVector(azimuth=90, colatitude=90, degrees=True),
    pattern_enum=DirectivityPattern.CARDIOID,
)

#PARAMS

case = 'audible' # ultra or audible
output_sig = 'noise' # noise or sweep

# room reflections order
order = 0 

# source locations
# locations = np.arange(0, 181, 5)
# print(locations)
locations = [50]
print(locations)
# radius distnace of the source from array
distance = 30e-1  # meters

## Direction of Arrival

#Several reference algorithms for direction-of-arrival (DOA) estimation are provided. These methods work in the frequency domain of which there are generally two types: incoherent and coherent methods.

#We provide the following algorithms: SRP-PHAT, MUSIC, CSSM, WAVES, TOPS, and FRIDA.

#Let's perform DOA for two sources.
# Location of sources
azimuth = np.array(locations) / 180. * np.pi


#A few constants and parameters for the algorithm such as the FFT size and the frequency range over which to perform DOA.
c = 343.    # speed of sound
fs = 96000  # sampling frequency
nfft = 256  # FFT size
if case == 'ultra':
    freq_range = [39000, 41000]
elif case == 'audible':
    freq_range = [2000, 9000]
else:
    print('select a case')


#Let's build a 2D room where we will perform our simulation.
snr_db = 1.    # signalxto-noise ratio
sigma2 = 10**(-snr_db / 10) / (4. * np.pi * distance)**2

# Create an anechoic room
room_dim = np.r_[8.,5.]
aroom = pra.ShoeBox(room_dim, fs=fs, max_order=order, sigma2_awgn=sigma2)

# fig, ax = aroom.plot()
# ax.set_xlim([-1, room_dim[0]+1])
# ax.set_ylim([-1, room_dim[1]+1])

if case == 'ultra':
    echo = pra.linear_2D_array(center=room_dim/2, M=7, phi=0, d=0.003)
elif case == 'audible':
    echo = pra.linear_2D_array(center=room_dim/2, M=5, phi=0, d=0.018)
else:
    print('select a case')

echo = np.concatenate((echo, np.array(room_dim/2, ndmin=2).T), axis=1)
aroom.add_microphone_array(pra.MicrophoneArray(echo, aroom.fs))

# fig, ax = aroom.plot()
# ax.set_xlim([-1, room_dim[0]+1]);
# ax.set_ylim([-1, room_dim[1]+1]);

if output_sig == 'sweep':
    tone_durn = 1000e-3 # seconds
    t_tone = np.linspace(0, tone_durn, int(fs*tone_durn))
    if case == 'ultra':
        chirp = signal.chirp(t_tone, 40e3, t_tone[-1], 40e3)
    elif case == 'audible':
        chirp = signal.chirp(t_tone, 4e3, t_tone[-1], 8e3)
    else:
        print('select a case')
    chirp *= signal.windows.hann(chirp.size)
    # output_chirp = np.concatenate((chirp, np.zeros((int(fs*0.2)))))
elif output_sig == 'noise':
    # We'll create two synthetic signals and add them to the room at the specified locations with respect to the array.
    # Add sources of 1 second duration
    rng = np.random.RandomState(23)
    duration_samples = int(fs)
else:
    print('select an output_sig')

for ang in azimuth:
    source_location = room_dim/2  + distance * np.r_[np.cos(ang), np.sin(ang)]
    if output_sig == 'sweep':
        source_signal = chirp
    elif output_sig == 'noise':
        source_signal = rng.randn(duration_samples)
    else:
        print('select an output_sig')
# aroom.add_source(source_location, source_signal, directivity=dir_obj)
aroom.add_source(source_location, source_signal)
  
# Run the simulation
aroom.simulate()

# IPython.display.Audio(data=source_signal, rate=fs)

fig, ax = aroom.plot(figsize=(12, 7))
ax.set_xlim([-1, room_dim[0]+1]);
ax.set_ylim([-1, room_dim[1]+1]);
plt.title('Room layout', fontdict={'fontsize': 15})

# The DOA algorithms require an STFT input, which we will compute for overlapping frames for our 1 second duration signal.
X = pra.transform.stft.analysis(aroom.mic_array.signals.T, nfft, nfft // 2)
X = X.transpose([2, 1, 0])

# Now let's compare a few algorithms!
algo_names = ['SRP', 'MUSIC', 'FRIDA', 'TOPS','CSSM', 'WAVES']
spatial_resp = dict()

# loop through algos
for algo_name in algo_names:
    # Construct the new DOA object
    # the max_four parameter is necessary for FRIDA only
    doa = pra.doa.algorithms[algo_name](echo, fs, nfft, c=c, num_src=2, max_four=4)

    # this call here perform localization on the frames in X
    doa.locate_sources(X, freq_range=freq_range)
    
    # store spatial response
    if algo_name is 'FRIDA':
        spatial_resp[algo_name] = np.abs(doa._gen_dirty_img())
    else:
        spatial_resp[algo_name] = doa.grid.values
        
    # normalize   
    min_val = spatial_resp[algo_name].min()
    max_val = spatial_resp[algo_name].max()
    spatial_resp[algo_name] = (spatial_resp[algo_name] - min_val) / (max_val - min_val)

# Let's plot the estimated spatial spectra and compare it with the true locations!
# plotting param
base = 1.
height = 10.
true_col = [0, 0, 0]

# loop through algos
phi_plt = doa.grid.azimuth
i = 1
fig = plt.figure(figsize=(14, 7))
for algo_name in algo_names:
    # plot
    ax = fig.add_subplot(230 + i, projection='polar')
    c_phi_plt = np.r_[phi_plt, phi_plt[0]]
    c_dirty_img = np.r_[spatial_resp[algo_name], spatial_resp[algo_name][0]]
    ax.plot(c_phi_plt, base + height * c_dirty_img, linewidth=2,
            alpha=0.55, linestyle='-', 
            label="spatial \n spectrum")
    plt.title(algo_name, fontdict={'fontsize': 15}, loc='left')
    
    # plot true loc
    for angle in azimuth:
        ax.plot([angle, angle], [base, base + height], linewidth=2, linestyle='--',
            color=true_col, alpha=0.6)
    K = len(azimuth)
    ax.scatter(azimuth, base + height*np.ones(K), c=np.tile(true_col,
               (K, 1)), s=500, alpha=0.75, marker='*',
               linewidths=0,
               label='true \n locations')

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
    i+=1
    

# music
phi_plt = doa.grid.azimuth
i = 1
fig = plt.figure()
algo_name = 'MUSIC'

# plot
ax = fig.add_subplot(111, projection='polar')
c_phi_plt = np.r_[phi_plt, phi_plt[0]]
c_dirty_img = np.r_[spatial_resp[algo_name], spatial_resp[algo_name][0]]
ax.plot(c_phi_plt, base + height * c_dirty_img, linewidth=2,
        alpha=0.55, linestyle='-', 
        label="spatial \n spectrum")
plt.title(algo_name, fontdict={'fontsize': 15}, loc='left')

# plot true loc
for angle in azimuth:
    ax.plot([angle, angle], [base, base + height], linewidth=2, linestyle='--',
        color=true_col, alpha=0.6)
K = len(azimuth)
ax.scatter(azimuth, base + height*np.ones(K), c=np.tile(true_col,
            (K, 1)), s=500, alpha=0.75, marker='*',
            linewidths=0,
            label='true \n locations')

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
i+=1

plt.show()
# plt.savefig('my_figure.png', dpi=300, facecolor='lightgray', format='png', bbox_inches='tight')

# 
# 
# # And then collect all the .png files
# image_files = natsort.natsorted(glob.glob('fileName*.png'))
# from PIL import Image, ImageDraw, ImageFont
# 
# # Create the frames
# frames = []
# all_solns = []
# for img in image_files:
#     new_frame = Image.open(img)       
#     draw = ImageDraw.Draw(new_frame)
#     # music
# 
#     phi_plt = doa.grid.azimuth
#     i = 1
#     fig = plt.figure()
#     algo_name = 'MUSIC'
# 
#     # plot
#     ax = fig.add_subplot(111, projection='polar')
#     c_phi_plt = np.r_[phi_plt, phi_plt[0]]
#     c_dirty_img = np.r_[spatial_resp[algo_name], spatial_resp[algo_name][0]]
#     ax.plot(c_phi_plt, base + height * c_dirty_img, linewidth=2,
#             alpha=0.55, linestyle='-', 
#             label="spatial \n spectrum")
#     plt.title(algo_name, fontdict={'fontsize': 15}, loc='left')
# 
#     # plot true loc
#     for angle in azimuth:
#         ax.plot([angle, angle], [base, base + height], linewidth=2, linestyle='--',
#             color=true_col, alpha=0.6)
#     K = len(azimuth)
#     ax.scatter(azimuth, base + height*np.ones(K), c=np.tile(true_col,
#                 (K, 1)), s=500, alpha=0.75, marker='*',
#                 linewidths=0,
#                 label='true \n locations')
# 
#     plt.legend()
#     handles, labels = ax.get_legend_handles_labels()
#     ax.legend(handles, labels, fontsize=8, bbox_to_anchor=(1.5, 0.6))
#     # ax.legend(handles, labels, framealpha=0.5,
#     #           scatterpoints=1, loc='upper center', fontsize=10,
#     #           ncol=1, bbox_to_anchor=(1.6, 0.5),
#     #           handletextpad=.2, columnspacing=1.7, labelspacing=0.1)
# 
#     ax.set_xticks(np.linspace(0, 2 * np.pi, num=12, endpoint=False))
#     ax.xaxis.set_label_coords(0.5, -0.11)
#     ax.set_yticks(np.linspace(0, 1, 2))
#     # ax.xaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle=':')
#     # ax.yaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle='--')
#     ax.set_ylim([0, 1.05 * (base + height)])
#     i+=1
#     plt.show()
#     # solns, chunknum = source_solutions.get()
#     frames.append(new_frame)
# 
# 
# # Save into a GIF file that loops forever
# frames[0].save('png_to_gif.gif', format='GIF',
#                 append_images=frames[1:],
#                 save_all=True,
#                 duration=80, loop=0)    
# import os 
# [os.remove(each) for each in image_files]
#         