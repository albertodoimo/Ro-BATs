import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy import signal
import sounddevice as sd

# Parameters
sample_rate = 96000  # Sample rate in Hz
fs = sample_rate
duration = 1e-3  # Duration of the sine sweep in seconds. max = blocksize/fs
num_repeats = 1  # Number of buffer/blocks to repeat
nchannels = 7
channels = (nchannels,1)  # in/out
block_size = 2048  # Block size for the streamß
mic_spacing = 0.018 
# initial_delay = 110 # ms
central_mic = 4

print('loading functions...')
#%%
# def calc_rms(in_sig):
#     '''
#     
#     '''
#     rms_sig = np.sqrt(np.mean(in_sig**2))
#     return(rms_sig)
# 

def calc_delay(two_ch,fs):
    '''
    Parameters
    ----------
    two_ch : (Nsamples, 2) np.array
        Input audio buffer
    ba_filt : (2,) tuple
        The coefficients of the low/high/band-pass filter
    fs : int, optional
        Frequency of sampling in Hz. Defaults to 44.1 kHz
    
    Returns
    -------
    delay : float
        The time-delay in seconds between the arriving audio across the 
        channels. 
    '''
    for each_column in range(2):
        two_ch[:,each_column] = two_ch[:,each_column]

    cc = np.correlate(two_ch[:,0],two_ch[:,1],'same')
    midpoint = cc.size/2.0
    delay = np.argmax(cc) - midpoint
    # convert delay to seconds
    delay *= 1/float(fs)
    # if np.abs(delay)< 5.5*10**-5:
    #     delay = 0
    # else:
    #     delay = delay
    return delay

def calc_multich_delays(multich_audio,fs):
    '''s
    Calculates peak delay based with reference of 
    channel 1. 
    '''
    nchannels = multich_audio.shape[1]
    delay_set = []
    for each in range(1, nchannels):
        delay_set.append(calc_delay(multich_audio[:,[0,each]],fs))
    # print(delay_set)
    return np.array(delay_set)

def avar_angle(delay_set,nchannels,mic_spacing):
    '''
    calculates the mean angle of arrival to the array
    with channel 1 as reference
    '''
    theta = []
    for each in range(0, nchannels-1):
        theta.append(np.arcsin((delay_set[each]*343)/((each+1)*mic_spacing))) # rad
    # print('theta=',theta)
    avar_theta = np.mean(theta)
    return avar_theta

print('functions loaded')

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
    print('chirp len = ',np.shape(chirp))
    # plt.plot(chirp)
    # plt.show()
    output_chirp = np.concatenate((chirp, np.zeros((int(fs*0.2)))))
    # output_chirp = np.pad(np.transpose(chirp), (0, block_size - len(chirp)), mode='constant')
    # output_tone_stereo = np.float32(np.column_stack((output_chirp, output_chirp)))
    # plt.plot(output_chirp)
    # plt.title('in data')
    # plt.show()
    print('output chirp shape',np.shape(output_chirp))
    return output_chirp

sine_sweep = generate_sine_sweep(duration, fs)

# Initialize buffers for recording
input_buffer = []
output_buffer = []
j = 0
# Callback function
def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(f"Status: {status}", flush=True)
    outdata[:] = sine_sweep[:frames].reshape(-1, 1)  # Play the sine sweep
    # print('indata buffer=', np.shape(indata))
    # j +=1
    # print('j= ',j)
    input_buffer.append(indata.copy())
    output_buffer.append(outdata.copy())

# Start the audio stream
def initialization():
    try:
        with sd.Stream(callback=audio_callback,
                    samplerate=sample_rate,
                    blocksize=block_size,
                    device=(usb_fireface_index,usb_fireface_index),
                    channels=channels):
            for _ in range(num_repeats):
                sd.sleep(int(1000))

    except KeyboardInterrupt:
        print("\nStream stopped by user")
    except Exception as e:
        print(f"An error occurred: {e}")

def update():
    try:
        with sd.Stream(callback=audio_callback,
                    samplerate=sample_rate,
                    blocksize=block_size,
                    device=(usb_fireface_index,usb_fireface_index),
                    channels=channels):
            for _ in range(num_repeats):
                sd.sleep(int((block_size/sample_rate+initial_delay)*1000))
                print(int((block_size/sample_rate+initial_delay)*1000))  # Wait for the duration of the sweep
            print("Sine sweep playback finished")

    except KeyboardInterrupt:
        print("\nStream stopped by user")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Convert buffers to numpy arrays
    print('input buffer shape =', np.shape(input_buffer))
#    j = np.shape(input_buffer[1])
    # print('j= ',j)

    # print(input_buffer)
    input_audio = np.zeros((block_size*6, 7))
    # print('input audio = ', input_audio)
    print('input audio shape = ', np.shape(input_audio))
    jj = 5
    input_buffer_int = np.concatenate(input_buffer)
    # print('input buf int = ', input_buffer_int)
    print('input buf int shape = ', np.shape(input_buffer_int))
    #for jj in range(6):
    print(jj)
    print(jj*block_size)
    print(block_size*(jj+1))
    input_audio = input_buffer_int[jj*block_size:block_size*(jj+1),:]
    # plt.plot(input_audio)
    # plt.show()
    # print('input audio = ', input_audio)
    print('input audio shape = ', np.shape(input_audio))
    output_audio = np.concatenate(output_buffer)
        
    # Filter input signal
    # delay_crossch = calc_multich_delays(in_sig,ba_filt,fs)
    # delay_crossch = calc_multich_delays(in_sig[:,[2,3,4,5]],ba_filt,fs)
    delay_crossch = calc_multich_delays(input_audio[:,[2,3,4,5,6]],fs)

    # calculate avarage angle
    avar_theta = avar_angle(delay_crossch,nchannels-2,mic_spacing)
    # print('avarage theta rad',avar_theta)
    print('\n avarage theta deg = ',np.rad2deg(avar_theta))
    print()
 
initialization()
input_buf_av = np.concatenate(input_buffer)
print('\n input buffer = ',np.shape(input_buffer))
print('\n input buf  shape = ',np.shape(input_buf_av))
input_audio_av = np.transpose(input_buf_av)
print('\n input av = ',input_audio_av)
print('\n input av shape = ',np.shape(input_audio_av))
mean = np.mean(input_audio_av[central_mic])
print('mean = ', mean)
thr = 0.001
initial_delay = 0

for sample in range(40000):
    k = 5
    # print('sample = ',sample)
    # print(np.abs(input_buf_av[sample,central_mic]))
    if np.abs(input_buf_av[sample,central_mic]) > np.abs(mean) + thr and np.abs(input_buf_av[sample+k,central_mic]) > np.abs(mean) + thr:
        initial_delay = sample/sample_rate
        #print('initial delay = ', initial_delay)
        break

#input_buffer = []
#output_buffer = []

for i in range (1):
    update()
    i+=1

#  # PLOTS

# Convert buffers to numpy arrays
input_audio = np.concatenate(input_buffer - mean)
print('input audio plot = ', np.shape(input_audio))
output_audio = np.concatenate(output_buffer)

t_audio = np.linspace(0, input_audio.shape[0]/sample_rate, input_audio.shape[0])

# Plot the input and output audio
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(t_audio, input_audio[:,[2,3,4,5,6]])
# plt.legend(input_audio)
plt.title('Input Audio Waveform')
plt.xlabel('sec')
plt.ylabel('Amplitude')
plt.subplot(2, 1, 2)
plt.plot(t_audio, output_audio)
# plt.legend(output_audio)
plt.title('Output Audio Waveform')
plt.xlabel('sec')
plt.ylabel('Amplitude')

# Plot the spectrograms
plt.figure()
aa = plt.subplot(211)
# plt.specgram(input_audio[:,0], Fs=fs, NFFT=1024, noverlap=512)   
plt.specgram(input_audio[:,central_mic], Fs=sample_rate, NFFT=1024, noverlap=512)    
plt.subplot(212, sharex=aa)
t_audio = np.linspace(0, input_audio.shape[0]/sample_rate, input_audio.shape[0])
# plt.plot(t_audio, input_audio[:,0])
plt.plot(t_audio, input_audio[:,central_mic])
plt.show()

