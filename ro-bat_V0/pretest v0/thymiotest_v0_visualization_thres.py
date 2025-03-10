import subprocess

print('install libraries...')
#subprocess.run(
#    'pip install thymiodirect sounddevice numpy scipy argparse',shell=True)

print('libraries installed')
print('import libraries...')

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore
from scipy import signal 
import sounddevice as sd

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import argparse
import time
import math

# thymio
from thymiodirect import Connection 
from thymiodirect import Thymio

print('libraries imported')

print('loading functions...')
#%%
def calc_rms(in_sig):
    '''
    
    '''
    rms_sig = np.sqrt(np.mean(in_sig**2))
    # print('rms =', rms_sig)
    return(rms_sig)

def calc_rms_avar(in_sig,ch):
    rms_sig = []
    #print(ch)
    #print('empty rms =', rms_sig)
    for i in range(ch):
        #print(i)
        rms_sig.append(np.sqrt(np.mean(in_sig[:,i]**2)))
        #print('rms =', rms_sig)
    
    avar_rms = np.mean(rms_sig)
    #print('rms avar =', avar_rms)
    return(avar_rms)

def calc_delay(two_ch,ba_filt,fs):
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
        two_ch[:,each_column] = signal.lfilter(ba_filt[0],ba_filt[1],two_ch[:,each_column])

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

def calc_multich_delays(multich_audio, ba_filt,fs):
    '''s
    Calculates peak delay based with reference of 
    channel 1. 
    '''
    nchannels = multich_audio.shape[1]
    delay_set = []
    for each in range(1, nchannels):
        delay_set.append(calc_delay(multich_audio[:,[0,each]],ba_filt,fs))
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
    avar_theta = np.mean(theta)
    print('theta=',theta)
    return avar_theta

app = pg.mkQApp("Realtime angle-of-arrival plot")
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('Realtime angle-of-arrival plot')
w.setCameraPosition(distance=25)

g = gl.GLGridItem()
w.addItem(g)

mypos = np.random.normal(0,1,size=(20,3))*5
mypos[:,2] = np.abs(mypos[:,2])
sp_my = gl.GLScatterPlotItem(pos=mypos, color=(1,1,1,1), size=10)
w.addItem(sp_my)

print('functions loaded')

print('initializating audio stream...')
#%% Set up the audio-stream of the laptop, along with how the 
# incoming audio buffers will be processed and thresholded.
import queue

# np.random.seed(78464)

input_audio_queue = queue.Queue()

def get_card(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return i

usb_fireface_index = get_card(sd.query_devices())
print('usbfireface index:', usb_fireface_index)

fs = 96000
block_size = 1024
#block_size = 8192
# channels = 5
channels = 5
mic_spacing = 0.003 #m
central_mic = 3

bp_freq = np.array([100,45000.0]) # the min and max frequencies
# to be 'allowed' in Hz.

ba_filt = signal.butter(2, bp_freq/float(fs*0.5),'bandpass')

#%%
# define the input signals features
S = sd.InputStream(samplerate=fs,blocksize=block_size, device=usb_fireface_index, channels=channels, latency='low')

S.start()

# creation the guide vector x values
all_xs = np.linspace(-10,10,S.blocksize)
# print('all_xs',all_xs.shape)
#threshold = 0.1

# creation of the guide matrix [blocksize * 3] with x = all_xs, y = 0 , z = 0 values 
guidepos = np.column_stack((all_xs, np.zeros(S.blocksize), np.zeros(S.blocksize)))
# creation of the line in the 3d space

# guideline = gl.GLScatterPlotItem(pos=guidepos, color=(1,0,1,1), size=10) # Guideline color is purple
# 
# w.addItem(guideline)

print('audio stream initialized')
#%%

in_sig,status = S.read(S.blocksize)
threshold = calc_rms_avar(in_sig, channels)
print('thresh=', threshold)
# # LINE PLOT
# # Calculate multichannel delays
def update():
    global sp_my, all_xs, threshold, S, ba_filt
    
    try:
        in_sig,status = S.read(S.blocksize) #[buffer,chan]
        #print('in_sig=', in_sig)
        
        # Filter input signal
        # delay_crossch = calc_multich_delays(in_sig,ba_filt,fs)
        

        # print('delay',delay_crossch)
        # calculate aavarage angle

        # avar_theta = avar_angle(delay_crossch,channels,mic_spacing)
        # print('avarage theta rad',avar_theta)
        # print('avarage theta deg',np.rad2deg(avar_theta))

        # Calculate RMS
        # rms_sig = calc_rms(in_sig[:,0])
        rms_sig = calc_rms(in_sig[:,central_mic])
        #print('\nrms ch 3=', rms_sig)
        # threshold = calc_rms_avar(in_sig, channels)
        #print('\nrms tresh =', threshold)
        if rms_sig > threshold*1.09:

            delay_crossch = calc_multich_delays(in_sig,ba_filt,fs)
        

            # print('delay',delay_crossch)
            # calculate aavarage angle

            avar_theta = avar_angle(delay_crossch,channels,mic_spacing)
            
            # Scale movement
            movement_amp_factor = 5e4
            all_zs = np.tile(rms_sig*movement_amp_factor*1e-3, S.blocksize)
            
            # Scale delay (delay_crossch[3] gives the best central accuracy w.r.t the center of the array )

            #all_delay = np.tile(-delay_crossch[3]*movement_amp_factor, S.blocksize)
            all_delay = np.tile(-delay_crossch[2]*movement_amp_factor, S.blocksize) 

            # print(all_delay)
            # all_delay = np.tile(avar_theta*movement_amp_factor, S.blocksize)
            # print(all_delay)
            
            # Add delay to signal            
            all_ys = in_sig[:,0]+all_delay[0]
            xyz = np.column_stack((all_xs,all_ys,all_zs))
            
        else:
            # when there's no/low signal at the mics
            # Set y values to 0
            y = np.zeros(S.blocksize)
            z= y.copy()
            xyz = np.column_stack((all_xs,y,z))
      
        sp_my.setData(pos=xyz)
        
    except KeyboardInterrupt:
        S.stop()
        
t = QtCore.QTimer()
t.timeout.connect(update)
t.start(5)


# POLAR PLOT

# Function to update the polar plot
def update_polar(frame):
    # Your streaming data source logic goes here
    
    in_sig,status = S.read(S.blocksize)
    rms_sig = calc_rms(in_sig[:,central_mic])
    # threshold = calc_rms_avar(in_sig, channels)
    if rms_sig > threshold*1.09:    
        # Filter input signal
        delay_crossch = calc_multich_delays(in_sig,ba_filt,fs)
        # print('delay',delay_crossch)
        # calculate aavarage angle

        avar_theta = avar_angle(delay_crossch,channels,mic_spacing)
        # print('avarage theta rad',avar_theta)
        # print('avarage theta deg',np.rad2deg(avar_theta))

        values = np.zeros(180)
        for i in range(len(values)):
            if round(np.rad2deg(avar_theta)+90) == i:
                values[i] = 1
            else:
                values[i] = 0

        # Update the polar plot
        line.set_ydata(values)
        return line,

# Set up the polar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
theta = np.linspace(-np.pi/2, np.pi/2, 180)
values = np.random.rand(180)
line, = ax.plot(theta, values)
ax.set_thetamin(-90)
ax.set_thetamax(90)

# Set up the animation
ani = FuncAnimation(fig, update_polar, frames=range(180), blit=False,interval= 50)

plt.show()



# # Thymio 
# # # %%------------------------------------------------------
# # 
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
#         # Delay to allow robot initialization of all variables
#         time.sleep(1)
#                 
#         # b) print all variables
#         # Main loop
#             # Calculate multichannel delays
# 
#         def update():
#             global sp_my, all_xs, threshold, S, ba_filt
#             print('thymio update cycle started...')
#             try:
#                 in_sig,status = S.read(S.blocksize)
#                 
#                 # Filter input signal
#                 delay_crossch = calc_multich_delays(in_sig,ba_filt,fs)
#                 # print('delay',delay_crossch)
#                 # calculate aavarage angle
# 
#                 avar_theta = avar_angle(delay_crossch,channels,mic_spacing)
#                 avar_theta_deg = np.rad2deg(avar_theta)
#                 #print('avarage theta rad',avar_theta)
#                 print('avarage theta deg',avar_theta_deg)
# 
#                  # get lights turn on based on proximity sensors
#                 if avar_theta_deg <-40:
#                     robot["leds.top"] = [0, 0, 255]
#                 if avar_theta_deg >= -40 and avar_theta_deg<-10:
#                     robot["leds.top"] = [0, 255, 255]
#                 if avar_theta_deg <=10 and avar_theta >=-10:
#                     robot["leds.top"] = [255, 255, 255]
#                 if avar_theta_deg <= 40 and avar_theta_deg>10:
#                     robot["leds.top"] = [255,255,0]
#                 if avar_theta_deg >40 :
#                     robot["leds.top"] = [255, 0, 0]   
#                 else:
#                     robot["leds.top"] = [0,0,0]
#                 
#             except KeyboardInterrupt:
#                 S.stop()
#             print('thymio update cycle interrupted')
# 
#     except Exception as err:
#         # Stop robot
#         robot['motor.left.target'] = 0
#         robot['motor.right.target'] = 0 
#         print(err)
# 
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
# # %%
