import sounddevice as sd
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt

# Configuration
fs = 96000  # Sampling rate
duration = 1  # Duration to capture for each DOA estimation
mic_positions = np.array([[0, 0], [0.1, 0]])  # Microphone positions

# Create a room and add microphones
room = pra.ShoeBox([1, 1], fs=fs, max_order=0)
room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))

# Initialize plot
plt.ion()
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
line, = ax.plot([], [], 'bo')

def update_plot(doa_estimation):
    theta = np.radians(doa_estimation)
    r = np.ones_like(theta)
    line.set_data(theta, r)
    ax.set_ylim(0, 1)
    plt.draw()
    plt.pause(0.001)

def compute_doa(indata):
    # Perform STFT
    X = pra.transform.stft.analysis(indata.T, 512, 256)

    # Compute spatial covariance matrix
    R = np.einsum('...dt,...et->...de', X, X.conj())
    R = np.mean(R, axis=2)

    # Perform MUSIC DOA estimation
    doa = pra.doa.algorithms['MUSIC'](mic_positions, fs, nfft=512, c=343)
    doa.locate_sources(R)
    
    # Return DOA estimation
    return doa.azimuth_recon * 180 / np.pi

def callback(indata, frames, time, status):
    if status:
        print(status)
    
    indata = indata[:, 0:len(mic_positions)]  # Select relevant mic channels
    doa_estimation = compute_doa(indata)
    update_plot(doa_estimation)

# Open the input stream
with sd.InputStream(channels=len(mic_positions), callback=callback, samplerate=fs, blocksize=int(fs * duration)):
    print("Running... Press Ctrl+C to stop.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Stopped.")


