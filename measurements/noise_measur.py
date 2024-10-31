import numpy as np
import matplotlib.pyplot as plt
import soundfile as wav
from scipy.signal import butter, lfilter, freqz

import matplotlib as mpl
import soundfile as wav

# Function to compute RMS of audio signal
# Set font to Times New Roman
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['text.usetex'] = False  # Use if LaTeX is not required

labelsize = 18
legendsize = 13
R = 0.5  # Reference distance (if used for calculations)
angles = np.arange(0, 210, 30)
line_styles = ['-', '--', '-.', ':']

freq_bands = {
    '0-1': (1, 1000),
    '1-2': (1000, 2000),
    '2-3': (2000, 3000),
    '3-4': (3000, 4000)
}

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


for i in range (1,7):
    angles = np.arange(0, 210, 30)
    # angles = np.concatenate((np.arange(0, 210, 30), np.arange(-150, 30, 30)))
    #print(angles)
    db_values = []
    rms_values = []
    fftsig = []
    meansig = []
    norm_mean = []
    SPLs = []
    ukon_number = i
    #print(i)
    one_Pa, sample_rate = wav.read(f'/Users/alberto/Documents/UNIVERSITA/MAGISTRALE/tesi/github/white-noise-characterisation/1Pa-tone/1Pa-tone_cut.wav')
    #plt.figure()
    #plt.plot(one_Pa) 
    #plt.show()

    for angle in angles:
        # Load the WAV file for the given angle
        audio_signal, sample_rate = wav.read(f'/Users/alberto/Documents/UNIVERSITA/MAGISTRALE/tesi/github/white-noise-characterisation/ukon{ukon_number}/{angle}.wav')
        
        #fftsig = np.fft.fft(audio_signal)  
        #plt.plot(fftsig) 
        #plt.show()
        #meansig = np.mean(4*np.pi*R*abs(fftsig))
        #print(meansig)
        #norm_mean = (meansig/np.max(meansig))
        #print(norm_mean)
        #db_scale = 20*np.log10(norm_mean)        # Compute the RMS value of the audio signal
        rms_value = compute_rms(audio_signal)
        print('rms value=',rms_value)
        #plt.figure()
        #plt.plot(audio_signal,label=str(angle))
        #plt.legend()

        rms_values.append(rms_value)
        #print(rms_values)
        db_value = rms_to_db(rms_value)
        #plt.fft(audio_signal[0], Fs=sample_rate, scale='linear', cmap='inferno')
        #plt.show()
        db_values.append(db_value)
    print('db values = ',db_values)
    
    #print('rms_values)=',rms_values)
    rms_ref_spl = compute_rms(one_Pa)
    print('rms_ref_spl=',rms_ref_spl)
    ref_db_1pa = rms_to_db(rms_ref_spl)
    print('ref db=',rms_to_db(rms_ref_spl))

    SPLs = 20*np.log10(rms_values/rms_ref_spl) 
    #print(f'SPL at {angle} degrees is {SPL} dB')
    print('SPLs=',SPLs)


    # Convert angles to radians for the polar plot
    angles_rad = np.deg2rad(angles)
    ref_db = db_values[0]

    # Plot the RMS values on a polar plot
    plt.figure(figsize = (18, 8))
    plt.suptitle(f'Thymio II robot loudspeaker amplitudes polar plot for UKON {ukon_number}\n', fontsize=25)


    ax = plt.subplot(132, polar = True)
    ax.plot(angles_rad, db_values-ref_db, 'k-', label='All bands')  # blue dots and lines
    #print(f'db values {angle}= ',db_value)

    # Optional: Improve plot aesthetics
    #ax.set_title(f'Thymio speaker Polar Plot of dB RMS Amplitudes at Different Angles for ukon{ukon_number}')
    ax.set_theta_direction(-1)  # clockwise
    ax.set_theta_offset(np.pi / 2)  # start from the top
    #ax.set_ylim(-6,2)
    ax.set_thetamax(np.pi)
    ax.set_thetamin(0)
    ax.set_ylabel('dB RMS',fontsize=labelsize)
    ax.set_yticks([3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10])
    ax.set_yticklabels(['','','','0','','','','','-5','','','','','-10'],fontsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)
    # Set ticks at every 30 degrees
    ax.set_xticks(np.deg2rad(np.arange(0, 181, 30)))
    ax.set_xticklabels([f"{angle}°" for angle in range(0, 181, 30)],fontsize=labelsize)
    ax.legend(loc='upper right',fontsize=legendsize)
    #plt.show(False)
    #plt.savefig(f'polar plots/ukon{ukon_number}/rms_ukon{ukon_number}.png', dpi=300, bbox_inches='tight')
    #plt.show(block=True)

    # Plot the SPL values on a polar plot
    #plt.figure(figsize = (8, 8))
    ax = plt.subplot(131, polar=True)
    ax.plot(angles_rad, 94+SPLs,'k-', label='All bands') 

    # Optional: Improve plot aesthetics
    #ax.set_title(f'Thymio speaker Polar Plot of SPL Amplitudes at Different Angles for ukon{ukon_number} \n reference signal: 1kHz 1Pa tone, 94 dB SPL')
    ax.set_title(f'Reference signal: 1kHz, 94 dB SPL\n',fontsize=labelsize)
    ax.set_theta_direction(-1)  # clockwise
    ax.set_theta_offset(np.pi / 2)  # start from the top
    ax.set_ylim(0,70)
    ax.set_thetamax(np.pi)
    ax.set_thetamin(0)
    ax.set_ylabel('dB SPL ref 20 $\mu$Pa',fontsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)
    #ax.set_yticks([5,4,3,2,1,0,-1,-2,-3,-4,-5])
    # Set ticks at every 30 degrees 
    ax.set_xticks(np.deg2rad(np.arange(0, 181, 30)))
    ax.set_xticklabels([f"{angle}°" for angle in range(0, 181, 30)],fontsize=labelsize)
    ax.legend(loc='upper right',fontsize=legendsize)
    #plt.savefig(f'polar plots/ukon{ukon_number}/SPL_ukon{ukon_number}.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)

#________________________________
# from noise_measur_freq.py


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

        rms_values.append(rms_value)
        #print(rms_values)
        db_value = rms_to_db(rms_value)
        db_values.append(db_value)
        
        # Compute RMS for each frequency band
        for band, (lowcut, highcut) in freq_bands.items():
            filtered_signal = butter_bandpass_filter(audio_signal, lowcut, highcut, sample_rate) #time signal 

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
    #plt.figure(figsize = (12,8))
    #plt.suptitle(f'Thymio Speaker Polar Plot of dB RMS Amplitudes for ukon{ukon_number}', fontsize=20)


#    ax = plt.subplot(121, polar=True)
#    # Plot RMS values
#    #ax.plot(angles_rad, db_values - db_values[0], 'bo-', label='Overall RMS')
#    ax.plot(angles_rad, db_values , 'b-', label='Overall dB RMS')
#    # Improve plot aesthetics
#    #ax.set_title(f'Thymio Speaker Polar Plot of RMS Amplitudes\nfor ukon{ukon_number}')
#    ax.set_theta_direction(-1)  # Clockwise
#    ax.set_theta_offset(np.pi / 2)  # Start from the top
#    #ax.set_ylim(-2, 2)
#    #ax.set_yticks([3,2,1,0,-1,-2,-3])
#    ax.set_thetamax(np.pi)
#    ax.set_thetamin(0)
#    ax.set_ylabel('dB RMS',fontsize=labelsize)
#    ax.set_xticks(np.deg2rad(np.arange(0, 181, 30)))
#    ax.set_xticklabels([f"{angle}°" for angle in range(0, 181, 30)],fontsize=labelsize)
#    ax.legend(loc='upper right',fontsize=labelsize)
    
    # Plot frequency bands
    ax = plt.subplot(133, polar=True)
    line_styles = ['-', '--', '-.', ':']

    for (band, db_band), linestyle in zip(freq_bands_db.items(), line_styles):
        print('\nref0=',freq_bands_db['0-1'][0])
        print('\nref=',freq_bands_db_ref[band])
        print(f'\n delta band {band} kHz',db_band-db_band[0])
        #ax.plot(angles_rad, db_band - db_band[0], linewidth=0.5, label=f'Band {band}')
        #ax.plot(angles_rad, db_band , label=f'{band}')
        ax.plot(angles_rad, db_band-freq_bands_db['0-1'][0], 'k',  linestyle=linestyle, label=f'{band}')
    
    # Improve plot aesthetics
    #ax.set_title(f'Thymio Speaker Polar Plot of RMS Amplitudes\nfor ukon{ukon_number}')
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_theta_offset(np.pi / 2)  # Start from the top
    #ax.set_ylim(-1, 1)
    #ax.set_yticks(fontsize=15)
    ax.set_thetamax(np.pi)
    ax.set_thetamin(0)
    ax.set_ylabel('dB RMS',fontsize=labelsize)
    ax.set_yticks([3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10])
    ax.set_yticklabels(['','','','0','','','','','-5','','','','','-10'],fontsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)
    ax.set_xticks(np.deg2rad(np.arange(0, 181, 30)))
    ax.set_xticklabels([f"{angle}°" for angle in range(0, 181, 30)],fontsize=labelsize)
    ax.legend(loc='upper right', title="Frequency \nBand [kHz] ", fontsize=legendsize, title_fontsize= legendsize)
    
    #plt.savefig(f'polar plots/ukon{ukon_number}/sum_ukon{ukon_number}.png', dpi=300, bbox_inches='tight')
    plt.show()
