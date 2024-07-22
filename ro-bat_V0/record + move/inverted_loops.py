#!/usr/bin/env python3
"""Create a recording with arbitrary duration.

The soundfile module (https://python-soundfile.readthedocs.io/)
has to be installed!

"""
import argparse
import tempfile
import queue
import sys
import json
import datetime
import time


import sounddevice as sd
import soundfile as sf
import numpy  as np # Make sure NumPy is loaded before it is used in the callback
#assert numpy  # avoid "imported but unused" message (W0611)

q = queue.Queue()

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())
    args.buffer = (indata.copy())
    print('buffer=',np.shape(args.buffer))

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

def get_card(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return i

usb_fireface_index = get_card(sd.query_devices())
print('usb_fireface_index=',usb_fireface_index)
fs = 48000
block_size = 4096
channels = 4
mic_spacing = 0.015 #m

   
def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


def update():
    try:
        in_sig = args.buffer
        #print(in_sig)
        print('in_sig = ',np.shape(in_sig))
        #for line in process.stdout:
        #    print('Output:', line.strip())
    
        delay_crossch = calc_multich_delays(in_sig[:,:channels],fs)        # print('delay',delay_crossch)

        # calculate avarage angle
        avar_theta = avar_angle(delay_crossch,channels-1,mic_spacing)
        #print('avarage theta rad',avar_theta)
        # print('avarage theta deg',np.rad2deg(avar_theta))
        
    except KeyboardInterrupt:

        print("\nupdate function stopped\n")

    return np.rad2deg(avar_theta)


def main(use_sim=False, ip='localhost', port=2001):
    ''' Main function '''
    try:
        global startime, process
                    # Configure Interface to Thymio robot
                    # simulation
            #       if use_sim:
            #            th = Thymio(use_tcp=True, host=ip, tcp_port=port, 
            #                        on_connect=lambda node_id: print(f' Thymio {node_id} is connected'))
                    # real robot
            #        else:
            #            port = Connection.serial_default_port()
            #            th = Thymio(serial_port=port, 
            #                        on_connect=lambda node_id: print(f'Thymio {node_id} is connected'))

                    # Connect to Robot
                    # th.connect()
            #        robot = th[th.first_node()]

        startime = datetime.datetime.now()
        print("\nREC START TIME: \n", startime.strftime("%Y-%m-%d %H:%M:%S"))
        print('')
        #initialization()
        #process = subprocess.Popen(["python3", "fieldrecorder_trigger_robat.py"])
        # process = subprocess.Popen(["python3", "rec_unlimited.py"],stdout=subprocess.PIPE, text=True)
        #if process.returncode == 0:
            # Deserialize the JSON output back to a Python array
        #    array = json.loads(process.stdout.strip())
        #    return array
        #print('array = ', array)
        if args.samplerate is None:    
            device_info = sd.query_devices(args.device, 'input')
            # soundfile expects an int, sounddevice provides a float:
            args.samplerate = int(device_info['default_samplerate'])
        if args.filename is None:
            #args.filename = tempfile.mktemp(prefix='delme_rec_unlimited_',suffix='.wav', dir='')
            timenow = datetime.datetime.now()
            time1 = timenow.strftime('%Y-%m-%d_%H-%M-%S')
            args.filename = 'test_' + str(time1) + '.wav'

        # Make sure the file is opened before recording anything:
        with sf.SoundFile(args.filename, mode='x', samplerate=args.samplerate,
                        channels=args.channels, subtype=args.subtype) as file:
            with sd.InputStream(samplerate=args.samplerate, device=usb_fireface_index,
                                channels=args.channels, callback=callback):
                print('#' * 80)
                print('press Ctrl+C to stop the recording')
                print('#' * 80)
                while True:
                    file.write(q.get())
        
                    avar_theta_deg = update()
                    avar_theta_deg = avar_theta_deg*1.25
                    print('avarage theta deg', avar_theta_deg)
                    
        #            ground_sensors = robot['prox.ground.reflected']
        #            ground_sensors_max = 1000
        #            # Adjust these threshold values as needed
        #            ground_sensors = robot['prox.ground.reflected']
        #            #print('ground = ',robot['prox.ground.reflected'])
        #            # Adjust these threshold values as needed
        #            left_sensor_threshold = 80
        #            right_sensor_threshold = 80
        #            direction = random.choice(['left', 'right'])
        #            if ground_sensors[0] > left_sensor_threshold  and ground_sensors[1]> right_sensor_threshold:
        #                # Both sensors detect the line, turn left
        #                if direction == 'left':
        #                    robot['motor.left.target'] = -150
        #                    robot['motor.right.target'] = 150   
        #                    time.sleep(waiturn) 
        #                else:
        #                    robot['motor.left.target'] = 150
        #                    robot['motor.right.target'] = -150
        #                    time.sleep(waiturn)
        #                # robot['motor.left.target'] = -50 + random.choice([, 100])
        #                # robot['motor.right.target'] = -50 + random.choice([-100, 100])
        #            elif ground_sensors[1] > right_sensor_threshold:
        #                # Only right sensor detects the line, turn left
        #                robot['motor.left.target'] = -150
        #                robot['motor.right.target'] = 150
        #                time.sleep(waiturn)
        #            elif ground_sensors[0] > left_sensor_threshold:
        #                # Only left sensor detects the line, turn right
        #                robot['motor.left.target'] = 150 
        #                robot['motor.right.target'] = -150 
        #                time.sleep(waiturn)
        #            else:       
        #                match avar_theta_deg:
        #                    case theta if theta < -30:
        #                        robot["leds.top"] = [0, 0, 255]
        #                        time.sleep(wait)
        #                        robot['motor.left.target'] = 400
        #                        robot['motor.right.target'] = 20
        #                        time.sleep(wait)
        #                    case theta if -30 <= theta < -1:
        #                        robot["leds.top"] = [0, 255, 255]
        #                        time.sleep(wait)
        #                        robot['motor.left.target'] = 300
        #                        robot['motor.right.target'] = 20
        #                        time.sleep(wait)
        #                    case theta if -1 <= theta <= 1:
        #                        robot["leds.top"] = [255, 255, 255]
        #                        time.sleep(wait)
        #                        robot['motor.left.target'] = 200
        #                        robot['motor.right.target'] = 200
        #                        time.sleep(wait)
        #                    case theta if 1 < theta <= 30:
        #                        robot["leds.top"] = [255, 255, 0]
        #                        time.sleep(wait)
        #                        robot['motor.right.target'] = 300
        #                        robot['motor.left.target'] = 20
        #                        time.sleep(wait)
        #                    case theta if theta > 30:
        #                        robot["leds.top"] = [255, 0, 0]
        #                        time.sleep(wait)
        #                        robot['motor.right.target'] = 400
        #                        robot['motor.left.target'] = 20
        #                        time.sleep(wait)
        #                    case _:
        #                        pass 
        #
#    except Exception as err:
#        # Stop robot
#        robot['motor.left.target'] = 0
#        robot['motor.right.target'] = 0 
#        robot["leds.top"] = [0,0,0]
#        print(err)
    except KeyboardInterrupt:

        stoptime = datetime.datetime.now()
        print("\nREC START TIME: \n", startime.strftime("%Y-%m-%d %H:%M:%S"))
        print("\nSTOP REC TIME: \n", stoptime.strftime("%Y-%m-%d %H:%M:%S"))
    
#        robot['motor.left.target'] = 0
#        robot['motor.right.target'] = 0
#        robot["leds.top"] = [0,0,0]

        print("\nPress Ctrl-C again to end the program\n")

        #print("fieldrecorder_trigger_robat.py has been terminated \n FILE IN PATH: \n\n" + path ) 

if __name__ == '__main__':
    # Parse commandline arguments to cofigure the interface for a simulation (default = real robot)
    parser = argparse.ArgumentParser(description='Configure optional arguments to run the code with simulated Thymio. '
                                                    'If no arguments are given, the code will run with a real Thymio.', add_help=False)
    
    # Add optional arguments
    parser.add_argument('-s', '--sim', action='store_true', help='set this flag to use simulation')
    parser.add_argument('-i', '--ip', help='set the TCP host ip for simulation. default=localhost', default='localhost')
    parser.add_argument('-p', '--port', type=int, help='set the TCP port for simulation. default=2001', default=2001)


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
        '-c', '--channels', type=int, default=4, help='number of input channels')
    parser.add_argument(
        '-t', '--subtype', type=str, help='sound file subtype (e.g. "PCM_24")')
    
    args = parser.parse_args(remaining)


    # Parse arguments and pass them to main function
    args = parser.parse_args()
    args.buffer = np.zeros((block_size, channels))
    main(args.sim, args.ip, args.port)


    


#%% Set up the audio-stream of the laptop, along with how the 
# incoming audio buffers will be processed and thresholded.
