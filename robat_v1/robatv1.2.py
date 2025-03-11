
# Robat v0 code to use with different arrays(PDM passive array or I2S array).
#
# FEATURES:
# > Compute pra alg for angle detection
# > Calculates dB values 
# > Triggers computation of angle only over threshold
# > Save recordings every x minutes 
# > Save file data from angle/polar plot in csv and xml files

print('import libraries...')

import numpy as np
import pyroomacoustics as pra
import scipy.signal as signal 
import sounddevice as sd
import soundfile as sf
import argparse
import queue
import datetime
import time
import random
import os
import threading

from scipy.ndimage import gaussian_filter1d
from thymiodirect import Connection 
from thymiodirect import Thymio
from functions.das_v2 import das_filter_v2
from functions.music import music
from functions.get_card import get_card 
from functions.pow_two_pad_and_window import pow_two_pad_and_window
from functions.check_if_above_level import check_if_above_level
from functions.calc_multich_delays import calc_multich_delays
from functions.avar_angle import avar_angle
from functions.bandpass import bandpass
from functions.save_data_to_csv import save_data_to_csv
from functions.save_data_to_xml import save_data_to_xml
from RobotMove import RobotMove

print('imports done')

# Get the index of the USB card
usb_fireface_index = get_card(sd.query_devices())
print(sd.query_devices())
print('usb_fireface_index=',usb_fireface_index)

# Parameters for the DOA algorithms
trigger_level = -55 # dB level ref max pdm
critical_level = -43 # dB level pdm critical distance
c = 343   # speed of sound
fs = 48000
rec_samplerate = 44000
block_size = 512
queue_size = block_size
analyzed_buffer = fs/block_size #theoretical buffers analyzed each second 
channels = 5
mic_spacing = 0.018 #m
nfft = 512
ref = channels//2 #central mic in odd array as ref
#print('ref=',ref) 
#ref= 0 #left most mic as reference
critical = []
# print('critical', np.size(critical))

# Possible algorithms for computing DOA: PRA (pyroomacoustics), CC, DAS
method = 'DAS'

doa_name = 'MUSIC'
doa_name = 'SRP'

# Parameters for the CC algorithm
avar_theta = None
theta_values   = []

# Parameters for the PRA algorithm
echo = pra.linear_2D_array(center=[(channels-1)*mic_spacing//2,0], M=channels, phi=0, d=mic_spacing)

# Parameters for the DAS algorithm
theta_das = np.linspace(-90, 90, 61) # angles resolution for DAS spectrum
N_peaks = 1 # Number of peaks to detect in DAS spectrum

# Parameters for the chirp signal
rand = random.uniform(0.8, 1.2)
duration_out = 100e-3  # Duration in seconds
duration_in = rand * 0.5  # Duration in seconds
duration_in = 500e-3  # Duration in seconds
amplitude = 0.1 # Amplitude of the chirp

# Generate a chirp signal
low_freq = 1e3 # [Hz]
hi_freq =  20e3 # [Hz]
t_tone = np.linspace(0, duration_out, int(fs*duration_out))
chirp = signal.chirp(t_tone, low_freq, t_tone[-1], hi_freq)
sig = pow_two_pad_and_window(chirp, fs = fs, show=False)
silence_dur = 50 # [ms]
silence_samples = int(silence_dur * fs/1000)
silence_vec = np.zeros((silence_samples, ))
full_sig = np.concatenate((sig, silence_vec))
print('len = ', len(full_sig))
stereo_sig = np.hstack([full_sig.reshape(-1, 1), full_sig.reshape(-1, 1)])
data = amplitude * np.float32(stereo_sig)

# Calculate highpass and lowpass frequencies based on the array geometry
auto_hipas_freq = int(343/(2*(mic_spacing*(channels-1))))
print('HP frequency:', auto_hipas_freq)
auto_lowpas_freq = int(343/(2*mic_spacing))
print('LP frequency:', auto_lowpas_freq)
highpass_freq, lowpass_freq = [auto_hipas_freq ,auto_lowpas_freq]
freq_range = [highpass_freq, lowpass_freq]
freq_range = [hi_freq, low_freq]

# Thymio movement parameters

norm_coefficient = 48000/1024 #most used fractional value, i.e. normalization value 
max_speed = 200 #to be verified 

# Straight speed
speed = 0.5 * max_speed * analyzed_buffer *(1/norm_coefficient) #generic speed of robot while moving 
speed = int(speed)
speed = 150 
#print('\nspeed = ',speed, '\n')

# Turning speed
prop_turn_speed = 50
turn_speed = 100 
waiturn = 1000 #turning time ms

left_sensor_threshold = 300
right_sensor_threshold = 300	

# Create queues for storing data
shared_audio_queue = queue.Queue()
angle_queue = queue.Queue()
qqq = queue.Queue()

# Stream callback function
class AudioProcessor:
    def __init__(self, fs, channels, block_size, data, args, trigger_level, critical_level, mic_spacing, ref, highpass_freq, lowpass_freq, theta_das, N_peaks):
        self.fs = fs
        self.channels = channels
        self.block_size = block_size
        self.data = data
        self.args = args
        self.trigger_level = trigger_level
        self.critical_level = critical_level
        self.mic_spacing = mic_spacing
        self.ref = ref
        self.highpass_freq = highpass_freq
        self.lowpass_freq = lowpass_freq
        self.theta_das = theta_das
        self.N_peaks = N_peaks
        self.q = queue.Queue()
        self.qq = queue.Queue()
        self.qqq = queue.Queue()
        self.shared_audio_queue = queue.Queue()
        self.current_frame = 0

        print('args.filename = ', args.filename)
    def continuos_recording(self):
        with sf.SoundFile(args.filename, mode='x', samplerate=args.rec_samplerate,
                            channels=args.channels, subtype=args.subtype) as file:
            with sd.InputStream(samplerate=fs, device=usb_fireface_index,channels=channels, callback=audio_processor.callback_in, blocksize=block_size):
                    while True:
                        file.write(self.shared_audio_queue.get())
        
    def callback_out(self, outdata, frames, time, status):
        if status:
            print(status)
        chunksize = min(len(self.data) - self.current_frame, frames)
        outdata[:chunksize] = self.data[self.current_frame:self.current_frame + chunksize]
        if chunksize < frames:
            outdata[chunksize:] = 0
            self.current_frame = 0  # Reset current_frame after each iteration
            raise sd.CallbackStop()
        self.current_frame += chunksize
            
    def callback_in(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        #correction=np.mean(indata)
        #print(correction)
        #in_sig = indata-correction
        self.shared_audio_queue.put((indata).copy())
        self.q.put((indata).copy())
        self.args.buffer = ((indata).copy())
        
    def callback(self, indata, outdata, frames, time, status):
        if status:
            print(status)
        chunksize = min(len(self.data) - self.current_frame, frames)
        outdata[:chunksize] = self.data[self.current_frame:self.current_frame + chunksize]
        if chunksize < frames:
            outdata[chunksize:] = 0
            self.current_frame = 0  # Reset current_frame after each iteration
            raise sd.CallbackStop()
        self.current_frame += chunksize
        self.q.put((indata).copy())
        self.args.buffer = ((indata).copy())

    def record(self, rec_counter, time1):
        q_contents = [self.q.get() for _ in range(self.q.qsize())]

        #print('q_contents = ', np.shape(q_contents))
        rec = np.concatenate(q_contents)
        #print('rec = ', np.shape(rec))


        rec2besaved = rec[:, :self.channels]
        save_path = '/home/thymio/robat_py/robat_v0_files/'
        #save_path = ''
        # Create folder with args.filename (without extension)
        folder_name = str(time1) + '_MULTIWAV'  
        folder_path = os.path.join(save_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        save_path = folder_path

        timenow = datetime.datetime.now()
        time2 = timenow.strftime('%Y-%m-%d__%H-%M-%S')
        full_path = os.path.join(save_path, str(rec_counter)+'__'+time2+'.wav')
        with sf.SoundFile(full_path, mode='x', samplerate=rec_samplerate,
                    channels=args.channels, subtype=args.subtype) as file:
            file.write(rec2besaved)
        print(f'\nsaved to {full_path}\n')

    def update(self):
        correction=np.mean(self.args.buffer)
        in_sig = self.args.buffer-correction
        ref_channels = in_sig
        #print(np.shape(in_sig))
        #print(in_sig)
        #print('ref_channels=', np.shape(ref_channels))
        ref_channels_bp = bandpass(ref_channels,self.highpass_freq,self.lowpass_freq,self.fs)
        #print('ref_channels_bp=', np.shape(ref_channels_bp))
        above_level,dBrms_channel = check_if_above_level(ref_channels_bp,self.trigger_level)
        #print(above_level)
        av_above_level = np.mean(dBrms_channel)
        #print(av_above_level)
        ref_sig = in_sig[:,self.ref]
        delay_crossch= calc_multich_delays(in_sig,ref_sig,self.fs,self.ref)

        # calculate avarage angle
        avar_theta = avar_angle(delay_crossch,self.channels,self.mic_spacing,self.ref)
        time3 = datetime.datetime.now()
        avar_theta1 = np.array([avar_theta, time3.strftime('%H:%M:%S.%f')[:-3]])

        #print('avarage theta',avar_theta1)

        self.qqq.put(avar_theta1.copy()) # store detected angle value
        #if av_above_level > critical_level:
        #    avar_theta = critical
        #    return avar_theta
            
        if av_above_level > self.trigger_level:


            #calculate avarage angle
            #avar_theta = avar_angle(delay_crossch,channels,mic_spacing,ref)
            #avar_theta = avar_angle(delay_crossch_gcc,channels,mic_spacing,ref)
            
            print('avarage theta deg = ', np.rad2deg(avar_theta))
            return np.rad2deg(avar_theta), av_above_level
        else:
            avar_theta = None
            return avar_theta, av_above_level

    def update_polar(self):
        # Your streaming data source logic goes here

        in_sig = self.args.buffer
        #print('buffer', np.shape(in_sig))

        X = pra.transform.stft.analysis(in_sig, nfft, nfft // 2)
        X = X.transpose([2, 1, 0])

        doa = pra.doa.algorithms[doa_name](echo, self.fs, nfft, c=c, num_src=2, max_four=2)
        doa.locate_sources(X, freq_range=freq_range)
        #print('azimuth_recon=',doa.azimuth_recon) # rad value of detected angles
        theta_pra_deg = (doa.azimuth_recon * 180 / np.pi) 
        #print('theta=',theta_pra_deg) #degrees value of detected angles

        spatial_resp = doa.grid.values # 360 values for plot
        #print('spat_resp',spatial_resp) 

        # normalize   
        min_val = spatial_resp.min()
        max_val = spatial_resp.max()
        spatial_resp = (spatial_resp - min_val) / (max_val - min_val)
        time3 = datetime.datetime.now()

        spatial_resp = np.hstack([spatial_resp, np.array(time3.strftime('%H:%M:%S.%f')[:-3])])
        

        self.qq.put(spatial_resp.copy())
        self.qqq.put(doa.azimuth_recon.copy())

        ref_channels = in_sig
        #print('ref_channels=', np.shape(ref_channels))
        ref_channels_bp = bandpass(ref_channels,self.highpass_freq,self.lowpass_freq,self.fs)
        #print('ref_channels_bp=', np.shape(ref_channels_bp))
        above_level,dBrms_channel = check_if_above_level(ref_channels_bp,self.trigger_level)
        #print(above_level)
        av_above_level = np.mean(dBrms_channel)
        #print(av_above_level)

        if av_above_level > self.critical_level:
            theta_pra = critical
            print('critical')
            return theta_pra

        elif av_above_level > self.trigger_level:

            if theta_pra_deg[0]<=180:
                theta_pra = 90-theta_pra_deg[0]
                print('pra theta deg 0', theta_pra)
                return theta_pra
            elif theta_pra_deg[1]<=180:
                theta_pra = 90-theta_pra_deg[1]
                print('pra theta deg 1', theta_pra)
                return theta_pra
            else:
                theta_pra = None
                return theta_pra

    def update_das(self):
        # Update  with the DAS algorithm
        in_buffer = self.shared_audio_queue.get()
        in_sig = in_buffer-np.mean(in_buffer)

        #print('buffer', np.shape(in_sig))
        #starttime = time.time()
        #theta, spatial_resp = das_filter_v2(in_sig, fs, channels, mic_spacing, freq_range, theta=theta_das)
        theta, spatial_resp = music(in_sig, self.fs, self.channels, self.mic_spacing, [self.highpass_freq, self.lowpass_freq], theta=self.theta_das, show = False)
        
        #print(time.time()-starttime)
        # find the spectrum peaks 

        spatial_resp = gaussian_filter1d(spatial_resp, sigma=4)
        peaks, _ = signal.find_peaks(spatial_resp)  # Adjust height as needed
        # peak_angle = theta_das[np.argmax(spatial_resp)]
        peak_angles = theta[peaks]
        N = self.N_peaks # Number of peaks to keep

        # Sort peaks by their height and keep the N largest ones
        peak_heights = spatial_resp[peaks]
        top_n_peak_indices = np.argsort(peak_heights)[-N:]  # Indices of the N largest peaks # Indices of the N largest peaks
        top_n_peak_indices = top_n_peak_indices[::-1]
        peak_angles = theta[peaks[top_n_peak_indices]]  # Corresponding angles
        print('peak angles', peak_angles)

        return peak_angles
        print('Angle = ', peak_angles)
        #qq.put(spatial_resp.copy())



if __name__ == '__main__':

    def int_or_str(text):
        """Helper function for argument parsing."""
        try:
            return int(text)
        except ValueError:
            return text

    # Parse commandline arguments to configure the interface for a simulation (default = real robot)
    parser = argparse.ArgumentParser(description='Configure optional arguments to run the code with simulated Thymio. '
                                                    'If no arguments are given, the code will run with a real Thymio.', add_help=False)
    # Add optional arguments
    parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])
    parser.add_argument(
        'filename', nargs='?', metavar='FILENAME',
        help='audio file to store recording to')
    parser.add_argument(
        '-d', '--device', type=int_or_str,
        help='input device (numeric ID or substring)')
    parser.add_argument(
        '-r', '--samplerate', type=int, help='sampling rate')
    parser.add_argument(
        '-c', '--channels', type=int, default=channels, help='number of input channels')
    parser.add_argument(
        '-t', '--subtype', type=str, help='sound file subtype (e.g. "PCM_24")')
    args = parser.parse_args(remaining)

    # Parse arguments and pass them to main function
    args = parser.parse_args()
    args.buffer = np.zeros((block_size, channels))

    # Set initial parameters for the audio processing
    startime = datetime.datetime.now()
    args.samplerate = fs
    args.rec_samplerate = rec_samplerate
    args.angle = None
    av_above_level = 0

    # Create folder for saving recordings
    time1 = startime.strftime('%Y-%m-%d__%H-%M-%S')
    save_path = '/home/thymio/robat_py/'
    folder_name = str(time1)  
    folder_path = os.path.join(save_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    name = 'MULTIWAV_' + str(time1) + '.wav'
    args.filename = os.path.join(folder_path, name)

    if args.samplerate is None:  
        print('error!: no samplerate set! Using default')
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = int(device_info['default_samplerate'])
    if args.filename is None:
        timenow = datetime.datetime.now()
        time1 = timenow.strftime('%Y-%m-%d__%H-%M-%S')
        args.filename = 'MULTIWAV_' + str(time1) + '.wav'
    print(args.samplerate)

    # Create instances of the AudioProcessor and RobotMove classes
    audio_processor = AudioProcessor(fs, channels, block_size, data, args, trigger_level, critical_level, mic_spacing, ref, highpass_freq, lowpass_freq, theta_das, N_peaks)
    robot_move = RobotMove(angle_queue, speed, turn_speed, waiturn, left_sensor_threshold, right_sensor_threshold, critical_level, av_above_level, trigger_level, ground_sensors_bool = False)
    
    # Create threads for the audio input and recording
    inputstream_thread = threading.Thread(target=
        audio_processor.continuos_recording, daemon = True)
    inputstream_thread.start()

    move_thread = threading.Thread(target=
        robot_move.audio_move, daemon = True)
    move_thread.start()
    angle_queue.put(None)

    try:
        #audio_processor.continuos_recording()
        while True:
            current_frame = 0 
            start_time = time.time()      
            with sd.OutputStream(samplerate=fs,
                                blocksize=0, 
                                device=usb_fireface_index, 
                                channels=2,
                                callback=audio_processor.callback_out,
                                latency='low') as out_stream:
                                while out_stream.active:
                                    #robot_move.stop()
                                    pass
            print('out time =', time.time() - start_time)                       
            #start_time = time.time()
            
            
            #print('in time =', time.time() - start_time)
            if method == 'PRA':
                time_start = time.time()
                args.angle = audio_processor.update_polar()
                #print('Angle = ', args.angle)
                time_end = time.time()
                #print('delta update pra',time_end - time_start,'sec')

            elif method == 'CC':
                args.angle, av_above_level = audio_processor.update()
                #args.angle, av_above_level = AudioProcessor.update()
                if isinstance(args.angle, (int, float, np.number)):
                    if np.isnan(args.angle):
                        args.angle = None
                #else:
                #    print("Warning: angle is not a numerical type")
                    
                #print('Angle = ', angle)
                #print('av ab level = ', av_above_level)
                #angle = int(angle)
                
                #print('updatestart=',datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S'))
                time_end = time.time()
                #print('delta update cc',time_end - time_start,'sec')

            elif method == 'DAS':

                time_start = time.time()

                in_buffer = audio_processor.shared_audio_queue.get()
                in_sig = in_buffer-np.mean(in_buffer)
                ref_channels = in_sig
                #print('ref_channels=', np.shape(ref_channels))
                ref_channels_bp = bandpass(ref_channels,highpass_freq,lowpass_freq,fs)
                #print('ref_channels_bp=', np.shape(ref_channels_bp))
                above_level, dBrms_channel = check_if_above_level(ref_channels_bp,trigger_level)
                #print(above_level)
                av_above_level = np.mean(dBrms_channel)
                #print(av_above_level)

                if av_above_level > critical_level:
                    print('critical')
                    angle_queue.put(audio_processor.update_das())

                elif av_above_level > trigger_level:
                    #args.angle = audio_processor.update_das()
                    angle_queue.put(audio_processor.update_das())
                    print(angle_queue.get())
                else:
                    #args.angle = None
                    angle_queue.put(None)

                print('Angle = ', angle_queue.get())
                print('av ab level = ', av_above_level)
                #args.angle = int(args.angle)
                
                #print('updatestart=',datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S'))
                time_end = time.time()
                #print('delta update das',time_end - time_start,'sec')
                start_time_rec = time.time()

            
            else:
                print('No valid method provided')

            time_start_robot = time.time()
            if args.angle != None:
                prop_turn_speed = 0.5 * 1/90 * (speed*int(args.angle)) #velocity of the wheels while turning 
                prop_turn_speed = int(prop_turn_speed)

                print('\nprop_turn_speed = ',prop_turn_speed, '\n')

            #PROPORTIONAL THYMIO MOVEMENT
            #robot_move.angle = args.angle
            #robot_move.speed = speed
            #robot_move.prop_turn_speed = prop_turn_speed
            #robot_move.waiturn = waiturn
            #robot_move.av_above_level = av_above_level

            #robot_move.audio_move()

                #print('delta robot',time.time() - time_start_robot,'sec')
        else:
            print('in time =', time.time() - start_time)
            robot_move.stop()

    except Exception as e:
        parser.exit(type(e).__name__ + ': ' + str(e))
        robot_move.stop()
    except Exception as err:
        # Stop robot
        robot_move.stop()
        print('err:',err)
    except KeyboardInterrupt:
        robot_move.stop()
        print('\nRecording finished: ' + repr(args.filename)) 
        parser.exit(0)  
        # Stop robot
        