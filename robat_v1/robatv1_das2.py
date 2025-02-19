import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from matplotlib.animation import FuncAnimation
from das_v2 import das_filter_v2
import soundfile as sf

#%%
# Constants

audio_path = input_path = '/Users/alberto/Documents/UNIVERSITA/MAGISTRALE/tesi/robat video-foto/tracking results/final test/RUN2/2024-10-24__16-44-12/' + 'spr_20241024_164412.wav'

data, samplerate = sf.read(audio_path) 
data = data[:int(len(data)/100),:]
#audio_buffer = sf.blocks(audio_path, blocksize=block_size, overlap=0)
#plt.figure()
#plt.plot(data)
#plt.show()

c = 343.0  # speed of sound
fs = samplerate  # sampling frequency
mic_spacing = 0.018
channels = 7
block_size = 2048
freq_range = [1000, 20000]


def get_card(device_list):
    for i, each in enumerate(device_list):
        if 'MCHStreamer' in each['name']:
            return i
usb_fireface_index = get_card(sd.query_devices())

#%%

def update_polar(frame):
    global rec
    correction=np.mean(buffer)
    in_sig = buffer-correction  #set imput signal to zero to eliminate negative offset 
    print(np.shape(buffer))
    #print(buffer)
    theta, spatial_resp = das_filter_v2(in_sig, fs, channels, mic_spacing, freq_range, theta=np.linspace(0, 360, 360))
    #spatial_resp = (spatial_resp - spatial_resp.min()) / (spatial_resp.max() - spatial_resp.min())
    line.set_ydata(spatial_resp)
    return line,

memory, rec = [], []

#%%
if __name__ == "__main__":
    current_frame = 0
    def callback(indata, frames, time, status):
        global current_frame, buffer
        if status:
            print(status)
        chunksize = min(len(data) - current_frame, frames)
        indata[:chunksize] = data[current_frame:current_frame + chunksize]
        buffer = indata[:chunksize]
        print(buffer)

        if chunksize < frames:
            indata[chunksize:] = 0
            raise sd.CallbackAbort()
        current_frame += chunksize

        correction=np.mean(buffer)
        in_sig = buffer-correction  #set imput signal to zero to eliminate negative offset 
        print(np.shape(buffer))
        #print(buffer)
        line, = ax.plot(theta, values)
        theta, spatial_resp = das_filter_v2(in_sig, fs, channels, mic_spacing, freq_range, theta=np.linspace(0, 360, 360))
        #spatial_resp = (spatial_resp - spatial_resp.min()) / (spatial_resp.max() - spatial_resp.min())
        line.set_ydata(spatial_resp)
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2)  # Rotate the plot by 90 degrees


    try: 
            stream = sd.InputStream(samplerate=fs,
                                   blocksize=block_size,
                                   device=5,
                                   channels=channels,
                                   callback=callback,
                                   latency='low')
                
            with stream:
                while stream.active:
                    pass

            current_frame = 0
            

    except KeyboardInterrupt:
        print('Interrupted by user')
        
        
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi / 2)  # Rotate the plot by 90 degrees
theta = np.linspace(-np.pi, np.pi, 360)
values = np.random.rand(360)
line, = ax.plot(theta, values)
ani = FuncAnimation(fig, update_polar, frames=range(360), blit=False, interval=10)
plt.show()


