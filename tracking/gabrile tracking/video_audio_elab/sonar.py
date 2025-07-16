from scipy import signal
import numpy as np

def sonar(signals, discarded_samples, max_index, fs, C_AIR=343):
    envelopes = np.abs(signal.hilbert(signals, axis=0))
    mean_envelope = np.mean(envelopes, axis=1)
    
    idxs, _ = signal.find_peaks(mean_envelope, prominence=12)
    try:
        emission_peak = idxs[0]

        peaks_positions = []
        enough = True
        for i in np.arange(envelopes.shape[1]):
            idxs, _ = signal.find_peaks(
                envelopes[emission_peak + discarded_samples:min(emission_peak + max_index, len(envelopes[emission_peak:])), i]
                , prominence=2.5)
            if idxs.any():
                peaks_positions.append(idxs[0] + emission_peak + discarded_samples)
            else:
                enough = False
                break  

        if enough:
            earliest_peak = np.min(peaks_positions)            
            dist = (earliest_peak - emission_peak)/fs*C_AIR/2 + 0.025               
            return dist, emission_peak, earliest_peak
            
        else:
            return 0, emission_peak, emission_peak
        
    except IndexError:
        raise ValueError
        