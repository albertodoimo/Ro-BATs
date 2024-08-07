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
from scipy.fftpack import fft, ifft
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
    return(rms_sig)


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
    plt.plot(cc)
    plt.show()
    plt.title('cc')
    midpoint = cc.size/2.0
    delay = np.argmax(cc) - midpoint
    # convert delay to seconds
    delay *= 1/float(fs)
    # if np.abs(delay)< 5.5*10**-5:
    #     delay = 0
    # else:
    #     delay = delay 
    return delay


def gcc_phat(sig, fs):
    # Compute the cross-correlation between the two signals
    refsig = sig[:,0]
    sig = sig[:,1]
    
    n = sig.shape[0] + refsig.shape[0]
    SIG = fft(sig, n=n)
    REFSIG = fft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    cc = np.fft.ifft(R / np.abs(R))
    max_shift = int(np.floor(n / 2))
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    #plt.plot(cc)
    #plt.show()
    #plt.title('gcc-phat')
    shift = np.argmax(np.abs(cc)) - max_shift
    return shift / float(fs)

def calc_multich_delays(multich_audio, ba_filt,fs):
    '''s
    Calculates peak delay based with reference of 
    channel 1. 
    '''
    nchannels = multich_audio.shape[1]
    delay_set = []
    delay_set_gcc = []
    for each in range(1, nchannels):
        delay_set.append(calc_delay(multich_audio[:,[0,each]],ba_filt,fs))
        delay_set_gcc.append(gcc_phat(multich_audio[:,[0,each]],fs))
    #print('delay=',delay_set)
    #print('delay gcc=',delay_set_gcc)
    return np.array(delay_set), np.array(delay_set_gcc)

def calc_multich_delays_center(multich_audio, ba_filt,fs):
    '''s
    Calculates peak delay based with reference of 
    channel 1. 
    '''
    nchannels = multich_audio.shape[1]
    delay_set_right = []
    delay_set_left = []

    for each in range(0, (nchannels-1)//2):
        delay_set_left.append(calc_delay(multich_audio[:,[(nchannels-1)//2,each]],ba_filt,fs))
    for each in range(((nchannels-1)//2)+1, (nchannels)):
        delay_set_right.append(calc_delay(multich_audio[:,[(channels-1)//2,each]],ba_filt,fs))

    #print('delay_set_right=', delay_set_right)
    #print('delay_set_left=' ,delay_set_left)
    return delay_set_left, delay_set_right

def avar_angle(delay_set,nchannels,mic_spacing):
    '''
    calculates the mean angle of arrival to the array
    with channel 1 as reference
    '''
    theta = []
    for each in range(0, nchannels-1):
        theta.append(np.arcsin((delay_set[each]*343)/((each+1)*mic_spacing))) # rad
    #print('theta=',theta)
    avar_theta = np.mean(theta)
    return avar_theta

def avar_angle_center(delay_set_l, delay_set_r,nchannels,mic_spacing):
    '''
    calculates the mean angle of arrival to the array
    with channel 1 as reference
    '''
    theta_l = []
    theta_r = [] 

    for each in range(0, (nchannels-1)//2):
        theta_l.append(np.arcsin((delay_set_l[each]*343)/((each+1)*mic_spacing))) # rad
    #print('theta_l=',theta_l)
    for each in range(0, (nchannels-1)//2):
        #print('each=' ,each)
        theta_r.append(np.arcsin((delay_set_r[each]*343)/((each+1)*mic_spacing))) # rad
    #print('theta_r=',theta_r)

    avar_theta_l = np.mean(theta_l)
    avar_theta_r = np.mean(theta_r)
    return avar_theta_l, avar_theta_r

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

fs = 48000
# block_size = 4096
block_size = 2048
# channels = 5
channels = 7
mic_spacing = 0.015 #m

bp_freq = np.array([100,20000.0]) # the min and max frequencies
# to be 'allowed' in Hz.

ba_filt = signal.butter(2, bp_freq/float(fs*0.5),'bandpass')

#%%
# define the input signals features
S = sd.InputStream(samplerate=fs,blocksize=block_size, device=usb_fireface_index, channels=channels, latency='low')

S.start()

# creation the guide vector x values
all_xs = np.linspace(-10,10,S.blocksize)
# print('all_xs',all_xs.shape)
threshold = 1e-5

# creation of the guide matrix [blocksize * 3] with x = all_xs, y = 0 , z = 0 values 
guidepos = np.column_stack((all_xs, np.zeros(S.blocksize), np.zeros(S.blocksize)))
# creation of the line in the 3d space

# guideline = gl.GLScatterPlotItem(pos=guidepos, color=(1,0,1,1), size=10) # Guideline color is purple
# 
# w.addItem(guideline)

print('audio stream initialized')
#%%

# # LINE PLOT
# # Calculate multichannel delays
def update():
    global sp_my, all_xs, threshold, S, ba_filt
    
    try:
        in_sig,status = S.read(S.blocksize)
        
        # Filter input signal
        delay_crossch_l, delay_crossch_r  = calc_multich_delays_center(in_sig,ba_filt,fs)
        avar_theta_l, avar_theta_r = avar_angle_center(delay_crossch_l, delay_crossch_r,channels,mic_spacing)

        #print('avar_theta_l',avar_theta_l)
        #print('avar_theta_r',avar_theta_r)

        #print('avarage theta deg l',np.rad2deg(avar_theta_l))
        #print('avarage theta deg r ',np.rad2deg(avar_theta_r))
        delay_crossch, delay_crossch_gcc = calc_multich_delays(in_sig,ba_filt,fs)

        

        # print('delay',delay_crossch)
        # calculate aavarage angle

        avar_theta = avar_angle(delay_crossch,channels,mic_spacing)
        avar_theta_gcc = avar_angle(delay_crossch_gcc,channels,mic_spacing)
        #print('avarage theta rad',avar_theta)
        print('avarage theta deg',np.rad2deg(avar_theta))
        print('avarage theta deg gcc',np.rad2deg(avar_theta))

        # Calculate RMS
        # rms_sig = calc_rms(in_sig[:,0])
        rms_sig = calc_rms(in_sig[:,2])
         
        if rms_sig > threshold:
            
            # Scale movement
            movement_amp_factor = 5e4
            all_zs = np.tile(rms_sig*movement_amp_factor*1e-3, S.blocksize)
            
            # Scale delay (delay_crossch[3] gives the best central accuracy w.r.t the center of the array )

            #all_delay = np.tile(-delay_crossch[3]*movement_amp_factor, S.blocksize)
           # all_delay = np.tile(-delay_crossch[2]*movement_amp_factor, S.blocksize) 

            # print(all_delay)
            # all_delay = np.tile(avar_theta*movement_amp_factor, S.blocksize)
            # print(all_delay)
            
            # Add delay to signal            
            #all_ys = in_sig[:,0]+all_delay[0]
            #xyz = np.column_stack((all_xs,all_ys,all_zs))
            
        else:
            # when there's no/low signal at the mics
            # Set y values to 0
            y = np.zeros(S.blocksize)
            z= y.copy()
            #xyz = np.column_stack((all_xs,y,z))
      
        #sp_my.setData(pos=xyz)
        
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
    #print('line = ',values)
    #print('line shape= ',np.shape(values))
    return line,

# Set up the polar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
theta = np.linspace(-np.pi/2, np.pi/2, 180)
values = np.random.rand(180)
line, = ax.plot(theta, values)
ax.set_thetamin(-90)
ax.set_thetamax(90)

# Set up the animation
#ani = FuncAnimation(fig, update_polar, frames=range(180), blit=True, interval= 10)

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
