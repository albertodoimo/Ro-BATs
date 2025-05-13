
# This script records a specific channel of an ASIO device (e.g., Fireface) and saves it to a file. 

import os
import soundfile as sf
import sounddevice as sd
import datetime as dt

def get_card(device_list):
    """
    Get the index of the ASIO card in the device list.
    Parameters:
    - device_list: list of devices (usually = sd.query_devices())

    Returns: index of the card in the device list
    """
    for i, each in enumerate(device_list):
        dev_name = each['name']
        name = 'Fireface' in dev_name
        if name:
            return i
    return None


def generate_ISOstyle_time_now():
    '''
    generates timestamp in yyyy-mm-ddTHH-MM-SS format
    
    based on  https://stackoverflow.com/a/28147286/4955732
    '''
    current_timestamp = dt.datetime.now().replace(microsecond=0).isoformat()
    current_timestamp = current_timestamp.replace(':', '-')
    current_timestamp = current_timestamp.replace('T', '_')
    return current_timestamp

    
if __name__ == "__main__":

    print(sd.query_devices())
    sd.default.device = get_card(sd.query_devices()) # or whatever the device string ID is when you check the output of 'sd.query_devices'
    print('selected device:', sd.default.device)
    fs = int(sd.query_devices(sd.default.device, 'input')['default_samplerate'])
    print('sample rate:', fs)

    current_date = dt.datetime.now().strftime("%Y-%m-%d")
    DIR = f"./{current_date}/"  # Directory to save the first sweeps
    os.makedirs(DIR, exist_ok=True)  # Create the directory if it doesn't exist
    duration = 10  # seconds
    rec_audio = sd.rec(int(duration * fs), channels=1, mapping=[9], blocking=True) # input_mapping=[9]: record only channel 9
    # # save recording 
    current_filename = 'ref_tone_gras'+generate_ISOstyle_time_now()+'.wav'
    sf.write(DIR + current_filename, rec_audio, fs)