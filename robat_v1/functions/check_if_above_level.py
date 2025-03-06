import numpy as np
import sounddevice as sd
from functions.calc_dBrms import calc_dBrms

def check_if_above_level(mic_inputs,trigger_level):
    """Checks if the dB rms level of the input recording buffer is above
    threshold. If any of the microphones are above the given level then 
    recording is initiated. 
    
    Parameters:
    - mic_inputs : Nsamples x Nchannels np.array. Data from soundcard
    - trigger_level :  If the input data buffer has a dB rms >= this value then True will be returned. 
                
    Returns:
        
        above_level : Boolean. True if the buffer dB rms is >= the trigger_level
    """ 

    dBrms_channel = np.apply_along_axis(calc_dBrms, 0, mic_inputs)   
    #print('dBrms_channel=',dBrms_channel)     
    above_level = np.any( dBrms_channel >= trigger_level)
    #print('above level =',above_level)
    
    return(above_level,dBrms_channel)
