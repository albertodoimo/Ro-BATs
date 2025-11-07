#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Created: 2025-11-3
Author: Alberto Doimo
email: alberto.doimo@uni-konstanz.de

Description:

Simulation of a circular array in the lab with different DOAs algorithms 
"""

# Let's begin by importing the necessary libraries all of which can be installed with `pip`, even `pyroomacoustics`!
import numpy as np
import matplotlib.pyplot as plt

# from scipy.io import wavfile
# from scipy.signal import fftconvolve

import pyroomacoustics as pra
import scipy.signal as signal 
import os

#PARAMS


def code(locations):
    case = 'audible' # ultra or audible
    output_sig = 'sweep' # noise or sweep

    # room reflections order
    order = 0  
    
    # radius distnace of the source from array
    distance = 2 # meters

    ## Direction of Arrival

    #Several reference algorithms for direction-of-arrival (DOA) estimation are provided. These methods work in the frequency domain of which there are generally two types: incoherent and coherent methods.

    #We provide the following algorithms: SRP-PHAT, MUSIC, CSSM, WAVES, TOPS, and FRIDA.

    #Let's perform DOA for two sources.
    # Location of sources
    azimuth = locations / 180. * np.pi
    print(azimuth) # rad

    #A few constants and parameters for the algorithm such as the FFT size and the frequency range over which to perform DOA.
    c = 343.    # speed of sound
    fs = 48000  # sampling frequency
    nfft = 256  # FFT size
    if case == 'ultra':
        freq_range = [39000, 41000]
    elif case == 'audible':
        freq_range = [1000, 9000]
    else:
        print('select a case')


    #Let's build a 2D room where we will perform our simulation.
    snr_db = 1.    # signalxto-noise ratio
    sigma2 = 10**(-snr_db / 10) / (4. * np.pi * distance)**2

    # Create an anechoic room
    room_dim = np.r_[7.,4.]
    aroom = pra.ShoeBox(room_dim, fs=fs, max_order=order, sigma2_awgn=sigma2)

    fig, ax = aroom.plot()
    ax.set_xlim([-1, room_dim[0]+1])
    ax.set_ylim([-1, room_dim[1]+1])

    if case == 'ultra':
        echo = pra.linear_2D_array(center=room_dim/2, M=8, phi=0, d=0.003)
    elif case == 'audible':
        echo = pra.circular_2D_array(center=room_dim/2, M=6, phi0=0, radius=0.045)
    else:
        print('select a case')
        
    # Concatenate the room center coordinates as a row vector to the echo array
    echo = np.concatenate((echo, np.array(room_dim/2, ndmin=2).T), axis=1)
    
    # Create a MicrophoneArray object using the echo array and the room's sample rate
    aroom.add_microphone_array(pra.MicrophoneArray(echo, aroom.fs))

    # fig, ax = aroom.plot()
    # ax.set_xlim([-1, room_dim[0]+1]);
    # ax.set_ylim([-1, room_dim[1]+1]);

    if output_sig == 'sweep':
        tone_durn = 10e-3 # seconds
        t_tone = np.linspace(0, tone_durn, int(fs*tone_durn))
        if case == 'ultra':
            chirp = signal.chirp(t_tone, 40e3, t_tone[-1], 40e3)
        elif case == 'audible':
            chirp = signal.chirp(t_tone, 0.5e3, t_tone[-1], 10e3)
        else:
            print('select a case')
        chirp *= signal.windows.hann(chirp.size)
        # output_chirp = np.concatenate((chirp, np.zeros((int(fs*0.2)))))
    elif output_sig == 'noise':
        # We'll create two synthetic signals and add them to the room at the specified locations with respect to the array.
        rng = np.random.RandomState(23)
        duration_samples = int(fs)
    else:
        print('select an output_sig')


    # for ang in azimuth:
    #for ang in azimuth:
    source_location = room_dim/2  + distance * np.r_[np.cos(azimuth), np.sin(azimuth)]
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
    print('signals =', aroom.mic_array.signals)
    print('signals =', np.shape(aroom.mic_array.signals))
    X = pra.transform.stft.analysis(aroom.mic_array.signals.T, nfft, nfft // 2)
    X = X.transpose([2, 1, 0])

    # Now let's compare a few algorithms!
    algo_names = ['SRP', 'MUSIC', 'TOPS']
    spatial_resp = dict()
    # loop through algos
    for algo_name in algo_names:
        # Ensure single-source tracking for each frame
        doa = pra.doa.algorithms[algo_name](echo, fs, nfft, c=c, num_src=1, max_four=4)
        doa.locate_sources(X, freq_range=freq_range)

        # retrieve spatial spectrum
        if hasattr(doa, 'grid') and hasattr(doa.grid, 'values'):
            resp = np.asarray(doa.grid.values)
        elif hasattr(doa, 'pseudo_spectrum'):
            resp = np.asarray(doa.pseudo_spectrum)
        elif hasattr(doa, 'spectrum'):
            resp = np.asarray(doa.spectrum)
        elif hasattr(doa, 'spatial_spectrum'):
            resp = np.asarray(doa.spatial_spectrum)
        else:
            try:
                n_angles = doa.grid.azimuth.size
            except Exception:
                n_angles = 360
            resp = np.zeros(n_angles)

        # normalize
        mn = resp.min()
        mx = resp.max()
        if mx > mn:
            resp = (resp - mn) / (mx - mn)
        else:
            resp = np.zeros_like(resp)

        spatial_resp[algo_name] = resp

    # plotting parameters
    base = 1.0
    height = 10.0
    true_col = [0, 0, 0]

    # angle arrays
    # ensure azimuth is an array (single-frame case)
    az = np.atleast_1d(azimuth)
    # try to get phi_plt from doa.grid if available, otherwise build a linear grid
    try:
        phi_plt = doa.grid.azimuth
    except Exception:
        phi_plt = np.linspace(0, 2 * np.pi, num=spatial_resp[algo_names[0]].size, endpoint=False)
    phi_plt = np.atleast_1d(phi_plt)

    # create figure with one subplot per algorithm
    fig = plt.figure(figsize=(14, 7))
    i = 1
    for algo_name in algo_names:
        ax = fig.add_subplot(230 + i, projection='polar')
        c_phi_plt = np.r_[phi_plt, phi_plt[0]]
        c_dirty_img = np.r_[spatial_resp[algo_name], spatial_resp[algo_name][0]]
        ax.plot(c_phi_plt, base + height * c_dirty_img, linewidth=2,
                alpha=0.55, linestyle='-', label="spatial\nspectrum")
        plt.title(algo_name, fontdict={'fontsize': 15}, loc='left')

        # plot true location(s)
        K = az.size
        ax.plot([az, az], [base, base + height], linewidth=2, linestyle='--',
                color=true_col, alpha=0.6)
        ax.scatter(az, base + height * np.ones(K), c=np.tile(true_col, (K, 1)),
                   s=200, alpha=0.9, marker='*', linewidths=0, label='true\nlocations')

        ax.set_xticks(np.linspace(0, 2 * np.pi, num=12, endpoint=False))
        ax.xaxis.set_label_coords(0.5, -0.11)
        ax.set_yticks(np.linspace(0, 1, 2))
        ax.set_ylim([0, 1.05 * (base + height)])
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=8, bbox_to_anchor=(1.5, 0.6))
        i += 1

    # Save a frame for this angle (and update animation GIF)
    try:
        import imageio.v2 as imageio_v2
    except Exception:
        import imageio as imageio_v2  # fallback

    try:
        from natsort import natsorted
        _natsort = True
    except Exception:
        _natsort = False

    # robust script dir (works in interactive sessions too)
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    frames_dir = os.path.join(script_dir, f'frames_{case}')
    os.makedirs(frames_dir, exist_ok=True)

    # compute a stable frame index from locations (works for scalar or array-like)
    loc_arr = np.asarray(locations)
    if loc_arr.size == 0:
        frame_idx = 0
    else:
        frame_idx = int(np.round(float(loc_arr.ravel()[0])))

    frame_path = os.path.join(frames_dir, f'frame_{frame_idx:03d}.png')
    # save current figure
    plt.savefig(frame_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    # build/update GIF from all frames in the folder
    file_list = [f for f in os.listdir(frames_dir) if f.lower().endswith('.png')]
    if _natsort:
        file_list = natsorted(file_list)
    else:
        file_list = sorted(file_list)

    file_paths = [os.path.join(frames_dir, f) for f in file_list]

    images = []
    for fp in file_paths:
        try:
            images.append(imageio_v2.imread(fp))
        except Exception:
            # skip unreadable files
            continue

    gif_path = os.path.join(script_dir, f'animation_{case}.gif')
    if len(images) > 0:

        # duration per frame (seconds)
        imageio_v2.mimsave(gif_path, images, duration=0.08)

