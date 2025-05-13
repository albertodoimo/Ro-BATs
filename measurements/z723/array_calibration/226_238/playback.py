# -*- coding: utf-8 -*-
"""
Extended speaker playback script
================================
Script that runs a speaker playback file every X seconds to mimic a playback
experiment run over an hour or two. 

Required inputs
---------------
* A WAV file
* Have to check the sounddevice device number or full string which is there in the output of sounddevice.query_devices()
* An ASIO-based audio interface
* A microphone to record the playback

Generated outputs
-----------------
* Multiple WAV files (depending on how long the experiment is run)

Created on Mon Apr  7 08:41:54 2025

@author: theja
"""
import sounddevice as sd
import soundfile as sf
import os
import datetime as dt
import time 
import sounddevice as sd
import numpy as np
import scipy.signal as signal
import time
from matplotlib import pyplot as plt   

def get_card(device_list):
    """
    Get the index of the ASIO card in the device list.
    Parameters:
    - device_list: list of devices (usually = sd.query_devices())

    Returns: index of the card in the device list
    """
    for i, each in enumerate(device_list):
        dev_name = each['name']
        name = 'Fireface' in dev_name
        if name:
            return i
    return None

def read_audiofile(filepath):
    audio, fs = sf.read(filepath)
    return audio, fs

if __name__ == "__main__":
    print(sd.query_devices())
    sd.default.device = get_card(sd.query_devices()) # or whatever the device string ID is when you check the output of 'sd.query_devices'
    print('selected device:', sd.default.device)

    wait_time = 0 # seconds

    pbkfile_path = './1_20k_5sweeps.wav'
    
    audio, fs = read_audiofile(pbkfile_path)
    #siren, fs = read_audiofile('./start_notify.wav')
    
    try:
        nchannels = 9
    except:
        nchannels = 1 
        audio = audio.reshape(-1,1)
    
    start_time = time.time()    

    # Playback
    
    sd.play(audio, samplerate=fs, blocking=True) 

    time.sleep(5)
