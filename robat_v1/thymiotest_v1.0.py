
print('import libraries...')
print('...')

import argparse
import threading
import time
import math

# import pyqtgraph as pg
# import pyqtgraph.opengl as gl
# from pyqtgraph.Qt import QtCore
import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# thymio
from thymiodirect import Connection 
from thymiodirect import Thymio

print('libraries imported\n')


print('loading functions...') 
print('...')

def calc_delay(two_ch,fs):
    for each_column in range(2):
        two_ch[:,each_column] = two_ch[:,each_column]

    cc = np.correlate(two_ch[:,0],two_ch[:,1],'same')
    midpoint = cc.size/2.0
    delay = np.argmax(cc) - midpoint
    # convert delay to seconds
    delay *= 1/float(fs)
    return delay

def calc_multich_delays(multich_audio,fs):
    nchannels = multich_audio.shape[1]
    delay_set = []
    for each in range(1, nchannels):
        delay_set.append(calc_delay(multich_audio[:,[0,each]],fs))
    # print(delay_set)
    return np.array(delay_set)

def avar_angle(delay_set,nchannels,mic_spacing):
    theta = []
    for each in range(0, nchannels-1):
        theta.append(np.arcsin((delay_set[each]*343)/((each+1)*mic_spacing))) # rad
    # print('theta=',theta)
    avar_theta = np.mean(theta)
    return avar_theta

print('functions loaded\n')

print('initializating audio stream...')
print('...')
#%% Set up the audio-stream of the laptop, along with how the 
# incoming audio buffers will be processed and thresholded.
import queue
input_audio_queue = queue.Queue()

def get_card(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return i

usb_fireface_index = get_card(sd.query_devices())
# print('usb_fireface_index=',usb_fireface_index)
fs = 96000
block_size = 4096
# block_size = 1024*2
#block_size = 8192
channels = 6
mic_spacing = 0.018 #m

bp_freq = np.array([100,45000.0]) # the min and max frequencies
# to be 'allowed' in Hz.

#%%
# define the input signals features
Sin = sd.InputStream (blocksize=block_size,channels=channels, latency='low',device=usb_fireface_index)
# print('Sin fs = ', Sin.samplerate)
# print('Sin blocksize = ', Sin.blocksize)
# print('Sin channels = ', Sin.channels)
# print('Sin latency = ', Sin.latency)
# print('Sin devinfo = ', Sin.device)
# #Sin.start()
# print('in stream initialized\n')

# fs = Sin.samplerate

Sout = sd.OutputStream(samplerate=fs, blocksize=block_size,channels=1, device=usb_fireface_index)
print('Sout fs = ',Sout.samplerate)
print('Sout blocksize = ', Sout.blocksize)
print('Sout channels = ', Sout.channels)
print('Sout latency = ', Sout.latency)
print('Sout devinfo = ', Sout.device)
# Sout.start()
print('out stream initialized\n')

tone_durn = 50e-3 # seconds
t_tone = np.linspace(0, tone_durn, int(fs*tone_durn))
chirp = signal.chirp(t_tone, 18e3, t_tone[-1], 0.5e3)
chirp *= signal.windows.hann(chirp.size)
output_chirp = np.concatenate((chirp, np.zeros((int(fs*0.2)))))
output_tone_stereo = np.float32(np.column_stack((output_chirp, output_chirp)))
#print(np.shape(output_tone_stereo))

all_input_data = queue.Queue()
i = 0
def callback_sine(indata, outdata, frames, time, status):
    outdata[:] = output_tone_stereo
    all_input_data.put(indata)

#----------------------- FROM PARSER ---------------------------------------------------
# def int_or_str(text):
#     """Helper function for argument parsing."""
#     try:
#         return int(text)
#     except ValueError:
#         return text
# 
# 
# parser = argparse.ArgumentParser(add_help=False)
# parser.add_argument(
#     '-l', '--list-devices', action='store_true',
#     help='show list of audio devices and exit')
# args, remaining = parser.parse_known_args()
# if args.list_devices:
#     print(sd.query_devices())
#     parser.exit(0)
# parser = argparse.ArgumentParser(
#     description=__doc__,
#     formatter_class=argparse.RawDescriptionHelpFormatter,
#     parents=[parser])
# # parser.add_argument(
# #     'filename', metavar='FILENAME',
# #     help='audio file to be played back')
# parser.add_argument(
#     '-d', '--device', type=int_or_str,
#     help='output device (numeric ID or substring)')
# args = parser.parse_args(remaining)
# 
# event = threading.Event()
# 
# try:
#     data, fs = sf.read('30-40k_3ms.wav', always_2d=True)
# 
#     current_frame = 0
# 
#     def callback(outdata, frames, time, status):
#         global current_frame
#         if status:
#             print(status)
#         chunksize = min(len(data) - current_frame, frames)
#         outdata[:chunksize] = data[current_frame:current_frame + chunksize]
#         if chunksize < frames:
#             outdata[chunksize:] = 0
#             raise sd.CallbackStop()
#         current_frame += chunksize
# 
#     stream = sd.OutputStream(
#         samplerate=fs, device=args.device, channels=data.shape[1],
#         callback=callback, finished_callback=event.set)
#     
#     with stream:
#         event.wait()  # Wait until playback is finished
# 
#     Sin.start()
#     in_sig,status = Sin.read(Sin.blocksize)
#     # 
#     delay_crossch = calc_multich_delays(in_sig[:,[2,3,4,5]],fs)
#     avar_theta = avar_angle(delay_crossch,channels-2,mic_spacing)
#     print('angle = ',np.rad2deg(avar_theta))
# 
# except KeyboardInterrupt:
#     parser.exit('\nInterrupted by user')
#     Sin.stop()
# except Exception as e:
#     parser.exit(type(e).__name__ + ': ' + str(e))
    
# -------------------------------------------------------------------------------------------------------------
# data, fs = sf.read('2sec_sweep.wav', always_2d=True)

try:
    # data, fs = sf.read('30-40k_3ms.wav', always_2d=True)
    #data, fs = sf.read('2sec_sweep.wav', always_2d=True)
    # print('data = ',data)
    # print(np.shape(data))
    # data, fs = sf.read('1-80k_3ms.wav', always_2d=True)
    #sd.play(data, fs)
    # stream_inout = sd.RaStream(samplerate=fs,
    #                      blocksize=output_chirp.shape[0],
    #                      device=(usb_fireface_index,usb_fireface_index),
    #                      #channels=(1,2),
    #                      channels=(6,2),
    #                      callback=callback_sine)
    # print('latencies: ', stream_inout.latencies)
    sd.play(output_tone_stereo, fs)
    #print(stream_inout.time)
    Sin.start()
    #print(stream_inout.time)
    in_sig = Sin.read(Sin.blocksize)
    # print(in_sig)
    #delay_crossch = calc_multich_delays(in_sig[:,[2,3,4,5]],fs)
    #print('delay_crossch = ', delay_crossch)
    #avar_theta = avar_angle(delay_crossch,channels-2,mic_spacing)
    #print('angle = ',np.rad2deg(avar_theta))
    # sd.wait()
except KeyboardInterrupt:
    Sin.stop()

# t = QtCore.QTimer()
# t.timeout.connect(update)
# t.start(5)

# rec = update()
rec = in_sig
rec = rec[0]
#print(rec)
#print('rec shape = ',np.shape(rec))
#print(np.shape(rec[:,2]))
#print('\n')
# #data = sf.read('30-40k_3ms.wav', always_2d=True)
# data = sf.read('1-80k_3ms.wav', always_2d=True)
# # data = sf.read('2sec_sweep.wav', always_2d=True)
# data = data[0]
# print(np.transpose(data))
# data = np.pad(np.transpose(data), (0, block_size*2 - len(data)), mode='constant')
# print(data[0,:])
# data = data[0,:]
# print(np.shape(data))
# print('\n')
# cc = np.correlate(rec[:,2],data,'same')
# # midpoint = cc.size/2.0
# delay = np.argmax(cc) 
# # convert delay to seconds
# # delay *= 1/float(fs)
# print('delay = ',delay)

fig, ax = plt.subplots()
ax.plot(rec[:,2])
plt.show()

plt.figure()
aa = plt.subplot(311)
# plt.specgram(input_audio[:,0], Fs=fs, NFFT=1024, noverlap=512)   
plt.specgram(rec[:,2], Fs=fs, NFFT=1024, noverlap=512)    
plt.subplot(312, sharex=aa)
t_audio = np.linspace(0, rec.shape[0]/fs, rec.shape[0])
# plt.plot(t_audio, input_audio[:,0])
plt.plot(t_audio, rec[:,2])
plt.subplot(313)
plt.plot(rec[:,2])
plt.show()

# delay = 6500
# delay *= 1/float(fs)
# print(delay)


# THYMIO -----------------------------------------------------------------------------------

# def main(use_sim=False, ip='localhost', port=2001):
#     ''' Main function '''
# 
#     try:
#         # Configure Interface to Thymio robot
#         # simulation
#         if use_sim:
#             th = Thymio(use_tcp=True, host=ip, tcp_port=port, 
#                         on_connect=lambda node_id: print(f' Thymio {node_id} is connected'))
#         # real robot
#         else:
#             port = Connection.serial_default_port()
#             th = Thymio(serial_port=port, 
#                         on_connect=lambda node_id: print(f'Thymio {node_id} is connected'))
# 
#         # Connect to Robot
#         th.connect()
#         robot = th[th.first_node()]  
# 
#         while True:
#             avar_theta_deg = update()
#             detectsCollision = max([robot['prox.horizontal'][i] > 1200 for i in range(5)])
#             print('avarage theta deg', avar_theta_deg)
# 
#             match avar_theta_deg:
#                 case theta if theta < -30:
#                     robot["leds.top"] = [0, 0, 255]
#                     time.sleep(0.005)
#                     robot['motor.left.target'] = 50
#                     robot['motor.right.target'] = 600
#                     time.sleep(0.005)
#                 case theta if -30 <= theta < -5:
#                     robot["leds.top"] = [0, 255, 255]
#                     time.sleep(0.005)
#                     robot['motor.left.target'] = 200
#                     robot['motor.right.target'] = 500
#                     time.sleep(0.005)
#                 case theta if -5 <= theta <= 5:
#                     robot["leds.top"] = [255, 255, 255]
#                     time.sleep(0.005)
#                     robot['motor.left.target'] = 50
#                     robot['motor.right.target'] = 50
#                     time.sleep(0.005)
#                 case theta if 5 < theta <= 30:
#                     robot["leds.top"] = [255, 255, 0]
#                     time.sleep(0.005)
#                     robot['motor.right.target'] = 200
#                     robot['motor.left.target'] = 500
#                     time.sleep(0.005)
#                 case theta if theta > 30:
#                     robot["leds.top"] = [255, 0, 0]
#                     time.sleep(0.005)
#                     robot['motor.right.target'] = 50
#                     robot['motor.left.target'] = 600
#                     time.sleep(0.005)
#                 case _:
#                     pass
#     
#     except Exception as err:
#         # Stop robot
#         robot['motor.left.target'] = 0
#         robot['motor.right.target'] = 0 
#         robot["leds.top"] = [0,0,0]
#         print(err)
#     except KeyboardInterrupt:
#         robot['motor.left.target'] = 0
#         robot['motor.right.target'] = 0
#         robot["leds.top"] = [0,0,0]
#         print("Press Ctrl-C again to end the program")
# 
# if __name__ == '__main__':
#     # Parse commandline arguments to cofigure the interface for a simulation (default = real robot)
#     parser = argparse.ArgumentParser(description='Configure optional arguments to run the code with simulated Thymio. '
#                                                     'If no arguments are given, the code will run with a real Thymio.')
#     
#     # Add optional arguments
#     parser.add_argument('-s', '--sim', action='store_true', help='set this flag to use simulation')
#     parser.add_argument('-i', '--ip', help='set the TCP host ip for simulation. default=localhost', default='localhost')
#     parser.add_argument('-p', '--port', type=int, help='set the TCP port for simulation. default=2001', default=2001)
# 
#     # Parse arguments and pass them to main function
#     args = parser.parse_args()
#     main(args.sim, args.ip, args.port)