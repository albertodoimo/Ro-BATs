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

def hhmmss_to_seconds(time_str):
    """Get seconds from time.
    Credit to: https://stackoverflow.com/a/6402859/4955732
    """
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

def read_audiofile(filepath):
    audio, fs = sf.read(filepath)
    return audio, fs

def generate_ISOstyle_time_now():
    '''
    generates timestamp in yyyy-mm-ddTHH-MM-SS format
    
    based on  https://stackoverflow.com/a/28147286/4955732
    '''
    current_timestamp = dt.datetime.now().replace(microsecond=0).isoformat()
    current_timestamp = current_timestamp.replace(':', '-')
    current_timestamp = current_timestamp.replace('T', '_')
    return current_timestamp

if __name__ == "__main__":
    print(sd.query_devices())
    sd.default.device = get_card(sd.query_devices()) # or whatever the device string ID is when you check the output of 'sd.query_devices'
    print('selected device:', sd.default.device)


    # how long to run the playbacks in HH:MM:SS
    duration = "01:00:00" 
    
    # gap between playbacks in seconds
    wait_time = 60 # seconds

    pbkfile_path = './playback_sweeps_1_95.wav'
    current_date = dt.datetime.now().strftime("%Y-%m-%d")
    DIR = f"./{current_date}/"  # Directory to save the first sweeps
    os.makedirs(DIR, exist_ok=True)  # Create the directory if it doesn't exist
    
    audio, fs = read_audiofile(pbkfile_path)
    #siren, fs = read_audiofile('./start_notify.wav')
    
    try:
        nchannels = 9
    except:
        nchannels = 1 
        audio = audio.reshape(-1,1)
    
    start_time = time.time()    

#name by which sounddevice refers to the audio-interface by
while time.time() <= start_time + hhmmss_to_seconds(duration):
    # Playback and record simultaneously
    
    #sd.play(siren, samplerate=fs, blocking=True) # play the start notification sound
    # wait for 5 seconds before starting the playback
    #time.sleep(5)
    rec_audio = sd.playrec(audio, samplerate=fs, input_mapping=[9], blocking=True) # input_mapping=[9]: record only channel 9
    # # save recording 
    current_filename = 'playback_recording'+generate_ISOstyle_time_now()+'.wav'
    sf.write(DIR + current_filename, rec_audio, fs)
    
    # wait a bit before starting the next play-rec
    time.sleep(wait_time)