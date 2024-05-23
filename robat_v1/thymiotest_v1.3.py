import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy import signal
import sounddevice as sd

# Parameters
sample_rate =96000  # Sample rate in Hz
duration = 10e-3  # Duration of the sine sweep in seconds. max = blocksize/fs
num_repeats = 5 # Number of buffer/blocks to repeat
channels = (6,1)  # in/out
block_size = 4096  # Block size for the stream

def get_card(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return i

usb_fireface_index = get_card(sd.query_devices())
print(sd.query_devices())
print('usb_fireface_index=',usb_fireface_index)

# Generate a sine sweep
def generate_sine_sweep(duration, fs):
    # t = np.linspace(0, duration, int(sample_rate * duration))
    # sweep = np.sin(2 * np.pi * 200 * (t**2) * 100)  # Simple quadratic sine sweep
    # print(np.shape(sweep))
    # plt.plot(sweep)
    # plt.show()
    # return sweep
    t_tone = np.linspace(0, duration, int(fs*duration))
    chirp = signal.chirp(t_tone, 40e3, t_tone[-1], 40e3)
    chirp *= signal.windows.hann(chirp.size)
    print(np.shape(chirp))
    # plt.plot(chirp)
    # plt.show()
    # output_chirp = np.concatenate((chirp, np.zeros((int(fs*0.2)))))
    output_chirp = np.pad(np.transpose(chirp), (0, block_size - len(chirp)), mode='constant')
    output_tone_stereo = np.float32(np.column_stack((output_chirp, output_chirp)))
    # plt.plot(output_chirp)
    # plt.title('in data')
    # plt.show()
    print(np.shape(output_chirp))
    return output_chirp

sine_sweep = generate_sine_sweep(duration, sample_rate)

# Initialize buffers for recording
input_buffer = []
output_buffer = []

# Callback function
def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(f"Status: {status}", flush=True)
    outdata[:] = sine_sweep[:frames].reshape(-1, 1)  # Play the sine sweep
    input_buffer.append(indata.copy())
    output_buffer.append(outdata.copy())

# Start the audio stream
try:
    with sd.Stream(callback=audio_callback,
                   samplerate=sample_rate,
                   blocksize=block_size,
                   device=(usb_fireface_index,usb_fireface_index),
                   channels=channels):
        for _ in range(num_repeats):
            sd.sleep(int((block_size/sample_rate)*1000))
            print(int((block_size/sample_rate)*1000))  # Wait for the duration of the sweep
        print("Sine sweep playback finished")

except KeyboardInterrupt:
    print("\nStream stopped by user")
except Exception as e:
    print(f"An error occurred: {e}")

# Convert buffers to numpy arrays
input_audio = np.concatenate(input_buffer)
print('input audio = ', np.shape(input_audio))
output_audio = np.concatenate(output_buffer)

t_audio = np.linspace(0, input_audio.shape[0]/sample_rate, input_audio.shape[0])

# Plot the input and output audio
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(t_audio, input_audio)
#plt.legend(input_audio)
plt.title('Input Audio Waveform')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.subplot(2, 1, 2)
plt.plot(t_audio, output_audio)
# plt.legend(output_audio)
plt.title('Output Audio Waveform')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

# Plot the spectrograms
plt.figure()
aa = plt.subplot(211)
# plt.specgram(input_audio[:,0], Fs=fs, NFFT=1024, noverlap=512)   
plt.specgram(input_audio[:,2], Fs=sample_rate, NFFT=1024, noverlap=512)    
plt.subplot(212, sharex=aa)
t_audio = np.linspace(0, input_audio.shape[0]/sample_rate, input_audio.shape[0])
# plt.plot(t_audio, input_audio[:,0])
plt.plot(t_audio, input_audio[:,2])
plt.show()

