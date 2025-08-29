# file to record all the mics in the system in a single file

#from functions.get_card import get_card 

#!/usr/bin/env python3
"""Create a recording with arbitrary duration.

The soundfile module (https://python-soundfile.readthedocs.io/)
has to be installed!

"""
import argparse
import tempfile
import queue
import sys
import datetime 
import os
import threading

import sounddevice as sd
import soundfile as sf
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)


def get_card(device_list):
    """
    Get the index of the ASIO card in the device list.
    Parameters:
    - device_list: list of devices (usually = sd.query_devices())

    Returns: index of the card in the device list
    """
    devices = []
    for i, each in enumerate(device_list):
        dev_name = each['name']
        name = 'MCHStreamer' in dev_name
        if name:
            devices.append(i)
    return devices

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

class AudioProcessor:
    def __init__(self, fs, channels, device):
        self.fs = fs
        self.channels = channels
        self.device = device
        self.q1 = queue.Queue()
        self.q2 = queue.Queue()

    def callback1(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        self.q1.put(indata.copy())

    def callback2(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        self.q2.put(indata.copy())

#     def continuos_recording(self):
#         with sf.SoundFile(self.filename, mode='x', samplerate=self.rec_samplerate,
#                             channels=self.channels, subtype=self.subtype) as file:
#             with sd.InputStream(samplerate=self.fs, device=self.usb_fireface_index,channels=self.channels, callback=self.callback_in, blocksize=self.block_size):
#                 while True:
#                     self.buffer = self.shared_audio_queue.get()
#                     file.write(self.buffer)
                        
#     input_stream1 = sd.InputStream(samplerate=self.fs, device=self.usb_fireface_index,channels=self.channels, callback=self.callback_in, blocksize=self.block_size)

    
#     input_stream2(self):
#         with sd.InputStream(samplerate=self.fs, device=self.usb_fireface_index ,channels=self.channels, callback=self.callback_in, blocksize=self.block_size) as in_stream:


parser = argparse.ArgumentParser(add_help=False)

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
    '-c', '--channels', type=int, default=1, help='number of input channels')
parser.add_argument(
    '-t', '--subtype', type=str, help='sound file subtype (e.g. "PCM_24")')
parser.add_argument(
    '-a', '--angle', type=str, help='angle')
parser.add_argument(
    '-dir', '--directory', type=str)
args = parser.parse_args(remaining)

# q = queue.Queue()
# q1 = queue.Queue()
# q2 = queue.Queue()

# def callback1(indata, frames, time, status):
#     """This is called (from a separate thread) for each audio block."""
#     if status:
#         print(status, file=sys.stderr)
#     q1.put(indata.copy())

# def callback2(indata, frames, time, status):
#     """This is called (from a separate thread) for each audio block."""
#     if status:
#         print(status, file=sys.stderr)
#     q2.put(indata.copy())


# Create folder for saving recordings
timenow = datetime.datetime.now()
time = timenow.strftime('%Y-%m-%d')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
if not args.directory:
    save_path = './pc_audio_recordings/'
    folder_name = str(time)
else:
    save_path = args.directory
    folder_name = str(time)
folder_path = os.path.join(save_path, folder_name)
os.makedirs(folder_path, exist_ok=True)

time1 = timenow.strftime('%Y-%m-%d__%H-%M-%S')
if not args.angle:
    name = 'MULTIWAV_' + str(time1) + '.wav'
else:
    name = args.angle + '.wav'
args.filename = os.path.join(folder_path, name)
args.device = get_card(sd.query_devices())
print(sd.query_devices())
print('device = ', args.device)
args.samplerate = 48000
args.channels = 16

audio_processor = AudioProcessor(args.samplerate, args.channels, args.device)

in1 = sd.InputStream(samplerate=args.samplerate, device=args.device[0], blocksize=32,
                            channels=args.channels, callback=audio_processor.callback1)
in2 = sd.InputStream(samplerate=args.samplerate, device=args.device[1], blocksize=32,
                            channels=args.channels, callback=audio_processor.callback2)
try:
    if args.samplerate is None:  
        print('error!: no samplerate set! Using default')
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = int(device_info['default_samplerate'])
    if args.filename is None:
        timenow = datetime.datetime.now()
        time1 = timenow.strftime('%Y-%m-%d__%H-%M-%S')
        args.filename = 'MULTIWAV_' + str(time1) + '.wav'
    print('fs=',args.samplerate)

    # Make sure the file is opened before recording anything:
    with sf.SoundFile(args.filename, mode='x', samplerate=args.samplerate,
                      channels=32, subtype=args.subtype) as file:
        with in1, in2:
            print('#' * 80)
            print('press Ctrl+C to stop the recording')
            print('#' * 80)
            while True:
                # Concatenate q1 and q2 data
                data1 = audio_processor.q1.get()
                data2 = audio_processor.q2.get()
                combined = numpy.concatenate((data1, data2), axis=1) 
                file.write(combined)


# def callback(indata, outdata, frames, time, status):
#     if status:
#         print(status)
#     outdata[:] = indata
#     q.put(indata.copy())

# inputstream_thread = threading.Thread(target=audio_processor.input_stream, daemon = True)
# inputstream_thread.start()

# try:
#     if args.samplerate is None:  
#         print('error!: no samplerate set! Using default')
#         device_info = sd.query_devices(args.device, 'input')
#         args.samplerate = int(device_info['default_samplerate'])
#     if args.filename is None:
#         timenow = datetime.datetime.now()
#         time1 = timenow.strftime('%Y-%m-%d__%H-%M-%S')
#         args.filename = 'MULTIWAV_' + str(time1) + '.wav'

#     print('fs=',args.samplerate)
#     with sf.SoundFile(args.filename, mode='x', samplerate=args.samplerate,
#                       channels=16, subtype=args.subtype) as file:
#         with sd.Stream(device=(args.device[0], args.device[1]),
#                     samplerate=args.samplerate,
#                     channels=args.channels, callback=callback), in1:
#             print('#' * 80)
#             print('press Return to quit')
#             print('#' * 80)
#             input()
#             while True:
#                 data = q1.get()
#                 file.write(data)
except KeyboardInterrupt:
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))



