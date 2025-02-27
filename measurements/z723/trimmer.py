import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

if __name__ == '__main__':

    file_name = '0.wav'
    
    cut_dir = 'cut_sweeps'
    y, fs = sf.read(file_name)

    y_cut = y[int(0.343*fs):int(0.353*fs)]
    sf.write(cut_dir + '/' + file_name, y_cut, fs)
    if int(file_name[0:2]) > 0 and int(file_name[0:2]) < 180:
        y_symm = y_cut
        file_name_symm = str(360 - int(file_name[0:3])) + 'deg.wav'
        sf.write(cut_dir + '/' + file_name_symm, y_symm, fs)

    y, fs = sf.read(cut_dir + '/' + file_name)
    print(fs)
    dur = len(y) / fs
    t = np.linspace(0, dur, len(y))

    plt.figure()
    plt.suptitle('Trimmed sweep ' + file_name + file_name_symm)
    plt.subplot(2, 1, 1)
    plt.plot(t, y)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.subplot(2, 1, 2)
    plt.specgram(y, Fs=fs, NFFT=64, noverlap=32)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()