import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy import signal
import sounddevice as sd

# Parameters
sample_rate = 96000  # Sample rate in Hz
fs = sample_rate
duration = 100e-3  # Duration of the sine sweep in seconds. max = blocksize/fs
num_repeats = 10  # Number of buffer/blocks to repeat
nchannels = 7
channels = (nchannels,1)  # in/out
block_size = 4096  # Block size for the stream
mic_spacing = 0.018 #m


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
    chirp = signal.chirp(t_tone, 3e3, t_tone[-1], 10e3)
    chirp *= signal.windows.hann(chirp.size)
    print(np.shape(chirp))
    # plt.plot(chirp)
    # plt.show()
    output_chirp = np.concatenate((chirp, np.zeros((int(fs*0.2)))))
    # output_chirp = np.pad(np.transpose(chirp), (0, block_size - len(chirp)), mode='constant')
    # output_tone_stereo = np.float32(np.column_stack((output_chirp, output_chirp)))
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
def update():
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
        
    # Filter input signal
    # delay_crossch = calc_multich_delays(in_sig,ba_filt,fs)
    # delay_crossch = calc_multich_delays(in_sig[:,[2,3,4,5]],ba_filt,fs)
    delay_crossch = calc_multich_delays(input_audio[:,[2,3,4,5,6]],fs)

    # calculate avarage angle
    avar_theta = avar_angle(delay_crossch,nchannels-2,mic_spacing)
    print(avar_theta)


update()
#  # PLOTS

#  # Convert buffers to numpy arrays
#  input_audio = np.concatenate(input_buffer)
#  print('input audio = ', np.shape(input_audio))
#  output_audio = np.concatenate(output_buffer)
#  
#  t_audio = np.linspace(0, input_audio.shape[0]/sample_rate, input_audio.shape[0])
#  
#  # Plot the input and output audio
#  plt.figure(figsize=(10, 8))
#  plt.subplot(2, 1, 1)
#  plt.plot(t_audio, input_audio[:,[2,3,4,5,6]])
#  # plt.legend(input_audio)
#  plt.title('Input Audio Waveform')
#  plt.xlabel('sec')
#  plt.ylabel('Amplitude')
#  plt.subplot(2, 1, 2)
#  plt.plot(t_audio, output_audio)
#  # plt.legend(output_audio)
#  plt.title('Output Audio Waveform')
#  plt.xlabel('sec')
#  plt.ylabel('Amplitude')
#  
#  # Plot the spectrograms
#  plt.figure()
#  aa = plt.subplot(211)
#  # plt.specgram(input_audio[:,0], Fs=fs, NFFT=1024, noverlap=512)   
#  plt.specgram(input_audio[:,4], Fs=sample_rate, NFFT=1024, noverlap=512)    
#  plt.subplot(212, sharex=aa)
#  t_audio = np.linspace(0, input_audio.shape[0]/sample_rate, input_audio.shape[0])
#  # plt.plot(t_audio, input_audio[:,0])
#  plt.plot(t_audio, input_audio[:,4])
#  plt.show()
#  
