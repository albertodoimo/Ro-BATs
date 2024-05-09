
print('libraries installed')
print('import libraries...')

import argparse
import time
import math

# import pyqtgraph as pg
# import pyqtgraph.opengl as gl
# from pyqtgraph.Qt import QtCore
import sounddevice as sd
import numpy as np
from scipy import signal
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# thymio
from thymiodirect import Connection 
from thymiodirect import Thymio

print('libraries imported')

print('loading functions...') 

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

print('functions loaded')

print('initializating audio stream...')
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
Sin = sd.InputStream(samplerate=fs,blocksize=block_size,channels=channels, latency='low',device=usb_fireface_index)
print('Sin fs = ', Sin.samplerate)
print('Sin blocksize = ', Sin.blocksize)
print('Sin channels = ', Sin.channels)
print('Sin latency = ', Sin.latency)
print('Sin devinfo = ', Sin.device)
Sin.start()
print('in stream initialized')
Sout = sd.OutputStream(samplerate=fs, blocksize=block_size,channels=1, device=usb_fireface_index)
print('Sout fs = ',Sout.samplerate)
print('Sout blocksize = ', Sout.blocksize)
print('Sout channels = ', Sout.channels)
print('Sout latency = ', Sout.latency)
print('Sout devinfo = ', Sout.device)
Sout.start()
print('out stream initialized')

def update():
    try:
        
        in_sig,status = Sin.read(Sin.blocksize)

        delay_crossch = calc_multich_delays(in_sig[:,[2,3,4,5]],fs)
        avar_theta = avar_angle(delay_crossch,channels-2,mic_spacing)

    except KeyboardInterrupt:
        Sin.stop()
    return np.rad2deg(avar_theta)


def main(use_sim=False, ip='localhost', port=2001):
    ''' Main function '''

    try:
        # Configure Interface to Thymio robot
        # simulation
        if use_sim:
            th = Thymio(use_tcp=True, host=ip, tcp_port=port, 
                        on_connect=lambda node_id: print(f' Thymio {node_id} is connected'))
        # real robot
        else:
            port = Connection.serial_default_port()
            th = Thymio(serial_port=port, 
                        on_connect=lambda node_id: print(f'Thymio {node_id} is connected'))

        # Connect to Robot
        th.connect()
        robot = th[th.first_node()]  

        while True:
            avar_theta_deg = update()
            detectsCollision = max([robot['prox.horizontal'][i] > 1200 for i in range(5)])
            print('avarage theta deg', avar_theta_deg)

            match avar_theta_deg:
                case theta if theta < -30:
                    robot["leds.top"] = [0, 0, 255]
                    time.sleep(0.005)
                    robot['motor.left.target'] = 50
                    robot['motor.right.target'] = 600
                    time.sleep(0.005)
                case theta if -30 <= theta < -5:
                    robot["leds.top"] = [0, 255, 255]
                    time.sleep(0.005)
                    robot['motor.left.target'] = 200
                    robot['motor.right.target'] = 500
                    time.sleep(0.005)
                case theta if -5 <= theta <= 5:
                    robot["leds.top"] = [255, 255, 255]
                    time.sleep(0.005)
                    robot['motor.left.target'] = 50
                    robot['motor.right.target'] = 50
                    time.sleep(0.005)
                case theta if 5 < theta <= 30:
                    robot["leds.top"] = [255, 255, 0]
                    time.sleep(0.005)
                    robot['motor.right.target'] = 200
                    robot['motor.left.target'] = 500
                    time.sleep(0.005)
                case theta if theta > 30:
                    robot["leds.top"] = [255, 0, 0]
                    time.sleep(0.005)
                    robot['motor.right.target'] = 50
                    robot['motor.left.target'] = 600
                    time.sleep(0.005)
                case _:
                    pass
    
    except Exception as err:
        # Stop robot
        robot['motor.left.target'] = 0
        robot['motor.right.target'] = 0 
        robot["leds.top"] = [0,0,0]
        print(err)
    except KeyboardInterrupt:
        robot['motor.left.target'] = 0
        robot['motor.right.target'] = 0
        robot["leds.top"] = [0,0,0]
        print("Press Ctrl-C again to end the program")

if __name__ == '__main__':
    # Parse commandline arguments to cofigure the interface for a simulation (default = real robot)
    parser = argparse.ArgumentParser(description='Configure optional arguments to run the code with simulated Thymio. '
                                                    'If no arguments are given, the code will run with a real Thymio.')
    
    # Add optional arguments
    parser.add_argument('-s', '--sim', action='store_true', help='set this flag to use simulation')
    parser.add_argument('-i', '--ip', help='set the TCP host ip for simulation. default=localhost', default='localhost')
    parser.add_argument('-p', '--port', type=int, help='set the TCP port for simulation. default=2001', default=2001)

    # Parse arguments and pass them to main function
    args = parser.parse_args()
    main(args.sim, args.ip, args.port)