import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.signal as signal
from matplotlib.animation import FuncAnimation
from das_v2 import das_filter_v2

# Define DAS filter function
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
    theta, spatial_resp = das_filter_v2(in_sig, fs, channels, mic_spacing, freq_range, theta=np.linspace(90, -90, 37))
    print(spatial_resp)
    #spatial_resp = 10 * np.log10(spatial_resp)
    #print(spatial_resp)
    spatial_resp = (spatial_resp - spatial_resp.min()) / (spatial_resp.max() - spatial_resp.min())
    #window = signal.windows.tukey(37,alpha=1)
    #spatial_resp = spatial_resp * window
    line.set_ydata(spatial_resp)
    print(line.get_ydata())
    return line,

initialization()
memory, rec = [], []
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi / 2)  # Rotate the plot by 90 degrees
theta = np.linspace(-np.pi/2, np.pi/2, 37)
ax.set_thetamin(-90)
ax.set_thetamax(90)
ax.set_xticks(np.pi/180. * np.linspace(-90, 90, 19), labels=np.arange(-90, 91, 10))
#ax.set_yticks([1e-11, 1e-5])
values = np.random.rand(37)
#values = np.zeros(37)
line, = ax.plot(theta, values)
ani = FuncAnimation(fig, update_polar, frames=range(37), blit=False, interval=10)
plt.show()
