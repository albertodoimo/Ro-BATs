#%%
 
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
# %%
if __name__ == '__main__':

    file_name = '180 full.wav'
    new_name = '180.wav'
    cut_dir = 'noise_floor'
    file_name_symm = '180.wav'
    y, fs = sf.read('audiofiles/'+ file_name)

    y_cut = y[int(0.5*fs):int(0.55*fs)]
    sf.write(cut_dir + '/' + new_name, y_cut, fs)

    y_symm = y_cut

    sf.write(cut_dir + '/' + file_name_symm, y_symm, fs)

    y, fs = sf.read(cut_dir + '/' + new_name)
    print(fs)
    dur = len(y) / fs
    t = np.linspace(0, dur, len(y))

    plt.figure()
    plt.suptitle('Trimmed sweep ' + new_name + file_name_symm)
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
# %%
