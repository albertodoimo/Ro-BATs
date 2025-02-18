import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import stft
from matplotlib.animation import FuncAnimation

# Define DAS filter function
def das_filter_v2(y, fs, nch, d, bw, theta=np.linspace(-90, 90, 36), c=343):    
    win_len = 256
    f_spec_axis, _, spectrum = stft(y, fs=fs, window=np.ones((win_len, )), nperseg=win_len, noverlap=win_len-1, axis=0)
    bands = f_spec_axis[(f_spec_axis > bw[0]) & (f_spec_axis < bw[1])]
    bands = np.array([bands[0], bands[len(bands)//3], bands[2*len(bands)//3], bands[-1]])
    p = np.zeros_like(theta, dtype=complex)
    
    for f_c in bands:
        w_s = (2*np.pi*f_c*d*np.sin(np.deg2rad(theta))/c)        
        a = np.exp(np.outer(np.linspace(0, nch-1, nch), -1j*w_s))     
        spec = spectrum[f_spec_axis == f_c, :, :].squeeze()
        cov_est = np.cov(spec, bias=True)
        
        for i, _ in enumerate(theta):
            p[i] += a[:, i].T.conj() @ cov_est @ a[:, i]/(nch**2)
    
    mag_p = np.abs(p)/len(bands)
    return theta, mag_p

# Constants
c = 343.0  # speed of sound
fs = 48000  # sampling frequency
mic_spacing = 0.018
channels = 5
block_size = 2048
freq_range = [1000, 20000]

def get_card(device_list):
    for i, each in enumerate(device_list):
        if 'MCHStreamer' in each['name']:
            return i
usb_fireface_index = get_card(sd.query_devices())

S = sd.InputStream(samplerate=fs, blocksize=block_size, device=usb_fireface_index, channels=channels, latency='low')

def initialization():
    try:
        with S.start():
            print("Stream started")

    except KeyboardInterrupt:
        print("\nStream stopped by user")
    except Exception as e:
        print(f"An error occurred: {e}")

def update_polar(frame):
    global rec
    in_sig0, status = S.read(S.blocksize)
    correction=np.mean(in_sig0)
    #print(correction)
    in_sig = in_sig0-correction
    print(np.mean(in_sig))
    theta, spatial_resp = das_filter_v2(in_sig, fs, channels, mic_spacing, freq_range, theta=np.linspace(0, 360, 360))
    spatial_resp = (spatial_resp - spatial_resp.min()) / (spatial_resp.max() - spatial_resp.min())
    line.set_ydata(spatial_resp)
    return line,

initialization()
memory, rec = [], []

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi / 2)  # Rotate the plot by 90 degrees
theta = np.linspace(-np.pi, np.pi, 360)
values = np.random.rand(360)
line, = ax.plot(theta, values)
ani = FuncAnimation(fig, update_polar, frames=range(360), blit=False, interval=10)
plt.show()
