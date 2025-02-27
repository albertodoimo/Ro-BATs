import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

if __name__ == '__main__':

    file_name = '0.wav'
    
    y, fs = sf.read(file_name)
    print(fs)

    dur = len(y) / fs
    t = np.linspace(0, dur, len(y))

    plt.figure()
    plt.suptitle('Original recording ' + file_name)
    plt.subplot(2, 1, 1)
    plt.plot(t, y)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.subplot(2, 1, 2)
    plt.specgram(y, Fs=fs, NFFT=64, noverlap=32)
    plt.show()