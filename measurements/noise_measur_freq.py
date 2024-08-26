import numpy as np
import matplotlib.pyplot as plt
import soundfile as wav
from scipy.signal import butter, lfilter, freqz

def compute_rms(audio_signal):
    """Compute the RMS of the audio signal."""
    return np.sqrt(np.mean(audio_signal**2))

def rms_to_db(rms_value):
    """Convert RMS value to decibels."""
    return 20 * np.log10(rms_value) if rms_value > 0 else -np.inf

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design a Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply a Butterworth bandpass filter to the data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    
    return lfilter(b, a, data)

def plot_polar_db(angles, db_values, ref_db, ukon_number, freq_bands):
    """Plot polar plot of RMS amplitudes in dB with frequency bands."""
    angles_rad = np.deg2rad(angles)
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # Plot RMS values
    ax.plot(angles_rad, db_values - db_values[0], 'bo-', label='Overall RMS')
    
    # Plot frequency bands
    for band, db_band in freq_bands.items():
        ax.plot(angles_rad, db_band - freq_bands_db[band][0], label=f'Band {band}')
        #ax.plot(angles_rad, db_band-db_band[0], label=f'Band {band}')
    
    # Improve plot aesthetics
    ax.set_title(f'Thymio Speaker Polar Plot of RMS Amplitsudes\nfor ukon{ukon_number}')
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_theta_offset(np.pi / 2)  # Start from the top
    #ax.set_ylim(-6, 2)
    #ax.set_yticks(np.arange(-6, 1, 1))
    ax.set_xticks(np.deg2rad(np.arange(0, 210, 30)))
    ax.set_xticklabels([f"{angle}°" for angle in range(0, 210, 30)])
    ax.legend(loc='upper right')
    
    plt.savefig(f'polar plots/rms_ukon{ukon_number}_freq.png', dpi=300, bbox_inches='tight')
    plt.show()

R = 0.5  # Reference distance (if used for calculations)
angles = np.arange(0, 210, 30)

freq_bands = {
    '0-2kHz': (1, 2000),
    '2-4kHz': (2000, 4000),
    '4-6kHz': (4000, 6000),
    '6-8kHz': (6000, 8000)
}

for i in range(1, 7):
    db_values = []
    rms_values = []
    freq_bands_db = {band: [] for band in freq_bands.keys()}
    ukon_number = i

    for angle in angles:
        # Load the WAV file for the given angle
        try:
            audio_signal, sample_rate = wav.read(f'/Users/alberto/Documents/UNIVERSITA/MAGISTRALE/tesi/github/white-noise-characterisation/ukon{ukon_number}/{angle}.wav')

        except FileNotFoundError:
            print(f"File for ukon{ukon_number} at angle {angle} not found.")
            continue

        # Compute overall RMS and convert to dB
        rms_value = compute_rms(audio_signal)
        print('\nrms value=',rms_value)
        #plt.figure()
        #plt.plot(audio_signal,label=str(angle))
        #plt.legend()

        rms_values.append(rms_value)
        #print(rms_values)
        db_value = rms_to_db(rms_value)
        db_values.append(db_value)
        
        # Compute RMS for each frequency band
        for band, (lowcut, highcut) in freq_bands.items():
            filtered_signal = butter_bandpass_filter(audio_signal, lowcut, highcut, sample_rate) #time signal 
            
            # Compute FFT
            #n = len(filtered_signal)
            #fft_result = np.fft.fft(filtered_signal)
            #freqs = np.fft.fftfreq(n, 1/sample_rate)
            # Calculate magnitude spectrum
            #magnitude_spectrum = np.abs(fft_result)

            #plt.figure(1)
            #plt.clf()
            #for order in [3, 6, 9]:
            #    b, a = butter_bandpass(lowcut, highcut, sample_rate, order=order)
            #    w, h = freqz(b, a, fs=sample_rate, worN=2000)
            #    plt.plot(w, abs(h), label="order = %d" % order)
            #plt.plot([0, 0.5 * sample_rate], [np.sqrt(0.5), np.sqrt(0.5)],
            #        '--', label='sqrt(0.5)')
            #plt.xlabel('Frequency (Hz)')
            #plt.ylabel('Gain')
            #plt.grid(True)
            #plt.legend(loc='best')
            #plt.show()
        
            band_rms = compute_rms(filtered_signal)
            band_db = rms_to_db(band_rms)
            freq_bands_db[band].append(band_db)
        if angle == 0:
            print('\nzero')
            print(f'\nfreq_bands_db_ref=',freq_bands_db)

        print(f'\nfreq_bands_db {angle}=',freq_bands_db)

    print('\nshape=',np.shape(freq_bands_db))

    freq_bands_db_ref = {band: values[0] for band, values in freq_bands_db.items()}
    print(f'\nfreq_bands_db_ref=', freq_bands_db_ref)
    print('\ndb values = ',db_values)

    angles_rad = np.deg2rad(angles)
    plt.figure(figsize = (10, 5))
    plt.suptitle(f'Thymio Speaker Polar Plot of RMS Amplitudes for ukon{ukon_number}', fontsize=16)


    ax = plt.subplot(121, polar=True)
    # Plot RMS values
    #ax.plot(angles_rad, db_values - db_values[0], 'bo-', label='Overall RMS')
    ax.plot(angles_rad, db_values , 'b-', label='Overall RMS')
    # Improve plot aesthetics
    #ax.set_title(f'Thymio Speaker Polar Plot of RMS Amplitudes\nfor ukon{ukon_number}')
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_theta_offset(np.pi / 2)  # Start from the top
    #ax.set_ylim(-2, 2)
    #ax.set_yticks([3,2,1,0,-1,-2,-3])
    ax.set_thetamax(np.pi)
    ax.set_thetamin(0)
    ax.set_xticks(np.deg2rad(np.arange(0, 181, 30)))
    ax.set_xticklabels([f"{angle}°" for angle in range(0, 181, 30)])
    ax.legend(loc='upper right')
    
    # Plot frequency bands
    ax = plt.subplot(122, polar=True)

    for band, db_band in freq_bands_db.items():
        print('\nref=',freq_bands_db_ref[band])
        print('\nband=',db_band)
        #ax.plot(angles_rad, db_band - freq_bands_db_ref[band], label=f'Band {band}')
        ax.plot(angles_rad, db_band , label=f'{band}')
        #ax.plot(angles_rad, db_band-db_band[0], label=f'Band {band}')
    
    # Improve plot aesthetics
    #ax.set_title(f'Thymio Speaker Polar Plot of RMS Amplitudes\nfor ukon{ukon_number}')
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_theta_offset(np.pi / 2)  # Start from the top
    #ax.set_ylim(-2, 2)
    #ax.set_yticks([3,2,1,0,-1,-2,-3])
    ax.set_thetamax(np.pi)
    ax.set_thetamin(0)
    ax.set_xticks(np.deg2rad(np.arange(0, 181, 30)))
    ax.set_xticklabels([f"{angle}°" for angle in range(0, 181, 30)])
    ax.legend(loc='upper right')
    
    plt.savefig(f'polar plots/{ukon_number}/rms_ukon{ukon_number}.png', dpi=300, bbox_inches='tight')
    #plt.show()
    
    #if db_values:
    #    #ref_db_all = db_values[0] # Reference value for normalization
    #    ref_db = freq_bands_db[band][0] # Reference value for normalization 
    #    angles_rad = np.deg2rad(angles)
#
#
    #    plot_polar_db(angles, db_values, ref_db, ukon_number, freq_bands_db)
    #else:
    #    print(f"No valid data for ukon{ukon_number}")
#
