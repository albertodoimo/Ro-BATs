
import argparse
import datetime
import threading
import queue
import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy import signal

# thymio
from thymiodirect import Connection 
from thymiodirect import Thymio

print('libraries imported')

print('loading functions...')

def calc_delay(two_ch, fs):
    cc = np.correlate(two_ch[:, 0], two_ch[:, 1], 'same')
    midpoint = cc.size // 2
    delay = np.argmax(cc) - midpoint
    delay *= 1 / float(fs)
    return delay

def calc_multich_delays(multich_audio, fs):
    nchannels = multich_audio.shape[1]
    delay_set = []
    for each in range(1, nchannels):
        delay_set.append(calc_delay(multich_audio[:, [0, each]], fs))
    return np.array(delay_set)

def avar_angle(delay_set, nchannels, mic_spacing):
    theta = []
    for each in range(0, nchannels - 1):
        theta.append(np.arcsin((delay_set[each] * 343) / ((each + 1) * mic_spacing)))
    avar_theta = np.mean(theta)
    return avar_theta

print('functions loaded')

print('initializing audio stream...')

def get_card(device_list):
    for i, each in enumerate(device_list):
        if 'MCHStreamer' in each['name']:
            return i
    raise ValueError('Audio device not found')

try:
    usb_fireface_index = get_card(sd.query_devices())
    print('usb_fireface_index =', usb_fireface_index)
except ValueError as e:
    print(e)
    exit(1)

fs = 48000
block_size = 1024
channels = 5
mic_spacing = 0.018

def initialization():
    global S
    S = sd.InputStream(samplerate=fs,
                       blocksize=block_size,
                       device=usb_fireface_index,
                       channels=channels)
    S.start()

#path = '/path/to/your/recordings/'

print('audio stream initialized')

q = queue.Queue()
recording = True
recording_thread = None

def record_audio():
    global recording
    while recording:
        q.put(S.read(block_size)[0])

def start_recording():
    global recording_thread
    recording_thread = threading.Thread(target=record_audio, daemon=True)
    recording_thread.start()

def stop_recording():
    global recording
    recording = False
    if recording_thread is not None:
        recording_thread.join()

def update():
    try:
        in_sig, status = S.read(S.blocksize)
        delay_crossch = calc_multich_delays(in_sig[:, :channels], fs)
        avar_theta = avar_angle(delay_crossch, channels - 1, mic_spacing)
    except KeyboardInterrupt:
        stop_recording()
        empty_qcontentsintolist()
        save_qcontents_aswav()
        print("\nupdate function stopped\n")
        S.stop()
    return np.rad2deg(avar_theta)

def empty_qcontentsintolist():
    global q_contents
    q_contents = [q.get() for _ in range(q.qsize())]

def save_qcontents_aswav():
    print('Saving file now...')
    rec = np.concatenate(q_contents)
    rec2besaved = rec[:, :channels]
    main_filename = 'MULTIWAV.WAV'
    try:
        sf.write(main_filename, rec2besaved, fs)
        print('File saved')
    except IOError:
        print('Could not save file !!')

def main(use_sim=False, ip='localhost', port=2001):
    try:
        global startime
        startime = datetime.datetime.now()
        print("\nREC START TIME: \n", startime.strftime("%Y-%m-%d %H:%M:%S"))
        initialization()
        start_recording()
        while True:
            avar_theta_deg = update()
            avar_theta_deg *= 1.25
            print('average theta deg', avar_theta_deg)
    except KeyboardInterrupt:
        stoptime = datetime.datetime.now()
        print("\nREC START TIME: \n", startime.strftime("%Y-%m-%d %H:%M:%S"))
        print("\nSTOP REC TIME: \n", stoptime.strftime("%Y-%m-%d %H:%M:%S"))
        stop_recording()
        empty_qcontentsintolist()
        save_qcontents_aswav()
        print("\nPress Ctrl-C again to end the program\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure optional arguments to run the code with simulated Thymio. '
                                                 'If no arguments are given, the code will run with a real Thymio.')
    parser.add_argument('-s', '--sim', action='store_true', help='set this flag to use simulation')
    parser.add_argument('-i', '--ip', help='set the TCP host ip for simulation. default=localhost', default='localhost')
    parser.add_argument('-p', '--port', type=int, help='set the TCP port for simulation. default=2001', default=2001)
    args = parser.parse_args()
    main(args.sim, args.ip, args.port)
