#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Created: 2025-11-12
Author: Alberto Doimo
email: alberto.doimo@uni-konstanz.de

Description:

Audio reproduction of camera+sync signal and target localization/repulsion signal from sounddevice
"""

import sounddevice as sd
import numpy as np
import soundfile


# Load wav files

# List available audio devices
print("Available audio devices:")
print(sd.query_devices())

# Select your device by index (change this number based on the list above)
device_index = 0  # Change to your desired device index

# Load the stereo wav file for channels 1-2
sample_rate_1, audio_data_stereo = soundfile.read('stereo_file.wav')

# Load the mono/stereo wav file for the other channel
sample_rate_2, audio_data_other = soundfile.read('other_file.wav')

# Ensure audio_data_other is mono or select one channel
if audio_data_other.ndim > 1:
    audio_data_other = audio_data_other[:, 0]

# Make sure both files have the same length
max_length = max(len(audio_data_stereo), len(audio_data_other))
if len(audio_data_stereo) < max_length:
    audio_data_stereo = np.pad(audio_data_stereo, ((0, max_length - len(audio_data_stereo)), (0, 0)))
if len(audio_data_other) < max_length:
    audio_data_other = np.pad(audio_data_other, (0, max_length - len(audio_data_other)))

# Create output array with 3 channels
output_channels = 3
output_audio = np.zeros((max_length, output_channels))

# Assign stereo file to channels 0-1 (channels 1-2 in 1-indexed)
output_audio[:, 0:2] = audio_data_stereo

# Assign other file to channel 2 (channel 3 in 1-indexed)
output_audio[:, 2] = audio_data_other

# Normalize to float32 range [-1, 1]
output_audio = output_audio.astype(np.float32) / np.max(np.abs(output_audio))

# Play the audio
print(f"Playing audio on device {device_index}...")
sd.play(output_audio, samplerate=sample_rate_1, device=device_index)
sd.wait()
print("Playback finished.")
