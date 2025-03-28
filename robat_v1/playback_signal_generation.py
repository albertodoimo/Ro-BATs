# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:44:27 2024

@author: theja
"""

import scipy.signal as signal 
import numpy as np 
import matplotlib.pyplot as plt 
import sounddevice as sd
import soundfile as sf

#%%
# make a sweep
durns = np.array([3, 5, 7] )*1e-3
fs = 192000 # Hz

all_sweeps = []
for durn in durns:
    t = np.linspace(0, durn, int(fs*durn))
    start_f, end_f = 15e3, 95e3
    sweep = signal.chirp(t, start_f, t[-1], end_f)
    sweep *= signal.windows.tukey(sweep.size, 0.95)
    sweep *= 0.8
    sweep_padded = np.pad(sweep, pad_width=[int(fs*0.1)]*2, constant_values=[0,0])
    all_sweeps.append(sweep_padded)
    
sweeps_combined = np.concatenate(all_sweeps)
sf.write('playback_sweeps.wav', sweeps_combined, samplerate=fs)
