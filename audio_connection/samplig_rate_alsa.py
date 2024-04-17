import subprocess

print('install libraries...')

subprocess.run(
    'pip install sounddevice',shell=True)
print('install libraries...')
subprocess.run(
    'pip install sounddevice matplotlib PyQt5 matplotlib sounddevice argparse time math',shell=True)

print('libraries installed')

import sounddevice as sd

S = sd.InputStream()
print('fs = ', S.samplerate)
print('blocksize = ', S.blocksize)
print('channels = ', S.channels)
print('latency = ', S.latency)
# print('devinfo = ', S.device[0])
