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
legendsize = 10
R = 0.5  # Reference distance (if used for calculations)
angles = np.arange(0, 210, 30)
line_styles = ['-', '--', '-.', ':']

freq_bands = {
    '0-1': (70, 1000),
    '1-2': (1000, 2000),
    '2-3': (2000, 3000),
    '3-4': (3000, 4000)
}

frequencies = [700, 1400,2100, 2800, 3500]

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

def get_amplitude_at_frequency(data, sample_rate, target_freq):
    # Perform FFT
    N = len(data)
    yf = np.fft.fft(data)
    xf = np.fft.fftfreq(N, 1 / sample_rate)
    
    # Find the index of the closest frequency in the FFT result
    idx = np.argmin(np.abs(xf - target_freq))
    
    # Compute the amplitude at the target frequency
    amplitude = np.abs(yf[idx]) / N
    return 20 * np.log10(amplitude)  # Convert to dB

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
        #db_scale = 20*np.log10(norm_mean)        
        # 
        # # Compute the RMS value of the audio signal
        rms_value = compute_rms(audio_signal[:,0])
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
    ax.plot(angles_rad, db_values-ref_db, label='All bands', linewidth=2)  # blue dots and lines
    #print(f'db values {angle}= ',db_value)

    # Optional: Improve plot aesthetics
    #ax.set_title(f'Thymio speaker Polar Plot of dB RMS Amplitudes at Different Angles for ukon{ukon_number}')
    ax.set_theta_direction(-1)  # clockwise
    ax.set_theta_offset(np.pi / 2)  # start from the top
    #ax.set_ylim(-6,2)
    ax.set_thetamax(np.pi)
    ax.set_thetamin(0)
    ax.set_ylabel('dB RMS',fontsize=labelsize)
    ax.set_yticks([10, 5, 0,-5,-10,-15,-20])
    #ax.set_yticklabels(['','0','-10', '-20'],fontsize=labelsize)
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
    ax.plot(angles_rad, 94+SPLs, label='All bands', linewidth=2) 

    # Optional: Improve plot aesthetics
    #ax.set_title(f'Thymio speaker Polar Plot of SPL Amplitudes at Different Angles for ukon{ukon_number} \n reference signal: 1kHz 1Pa tone, 94 dB SPL')
    ax.set_title(f'Reference signal: 1kHz, 94 dB SPL\n',fontsize=labelsize)
    ax.set_theta_direction(-1)  # clockwise
    ax.set_theta_offset(np.pi / 2)  # start from the top
    ax.set_ylim(45,75)
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
    #plt.show(block=False)

#________________________________
# from noise_measur_freq.py


    db_values = []
    rms_values = []
    amp_bands = {band: [] for band in freq_bands.keys()}
    freq_bands_db = {band: [] for band in freq_bands.keys()}
    amplitudes = {freq: [] for freq in frequencies}
    ukon_number = i

    for angle in angles:
        # Load the WAV file for the given angle
        try:
            audio_signal, sample_rate = wav.read(f'/Users/alberto/Documents/UNIVERSITA/MAGISTRALE/tesi/github/white-noise-characterisation/ukon{ukon_number}/{angle}.wav')

        except FileNotFoundError:
            print(f"File for ukon{ukon_number} at angle {angle} not found.")
            continue

        # Compute overall RMS and convert to dB
        rms_value = compute_rms(audio_signal[:,0])
        print('\nrms value=',rms_value)

        rms_values.append(rms_value)
        #print(rms_values)
        db_value = rms_to_db(rms_value)
        db_values.append(db_value)
        amplitude = 4*np.pi*R*np.abs(np.fft.fft(audio_signal[:,0]))


        for freq in frequencies:
            amplitude = get_amplitude_at_frequency(audio_signal[:,0], sample_rate, freq)
            amplitudes[freq].append(amplitude)

        print(f'amp',amplitudes)

        #plt.figure()
        # Compute RMS for each frequency band
        for band, (lowcut, highcut) in freq_bands.items():
            filtered_signal = butter_bandpass_filter(audio_signal[:,0], lowcut, highcut, sample_rate) #time signal
            
            print('freq',lowcut, highcut)
            freqs = np.fft.fftfreq(len(filtered_signal), 1/sample_rate)
            band1 = compute_rms(np.abs(np.fft.fft(filtered_signal)))
            #print('band1',band1)
            #plt.plot(freqs, np.abs(audio_signal[:,0]))
            #plt.plot(freqs, band1)
            #plt.show()

            amplitude_bands = 4*np.pi*R*np.abs(np.fft.fft(filtered_signal)) 
            amplitude = 4*np.pi*R*np.abs(np.fft.fft(audio_signal[:,0]))
            amplitude = butter_bandpass_filter(amplitude, lowcut, highcut, sample_rate)
            amp_bands[band].append(amplitude)

            freqs = np.fft.fftfreq(len(filtered_signal), 1/sample_rate)
            #plt.plot(freqs, np.mean(amplitude))
            #plt.show()
            
            band_rms = compute_rms(band1)
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
        ax.plot(angles_rad, db_band - db_band[0], linewidth=2, label=f'Band {band}')
        #ax.plot(angles_rad, db_band , label=f'{band}')
        #ax.plot(angles_rad, db_band-freq_bands_db['0-1'][0], 'k',  linestyle=linestyle, label=f'{band}')
    #for (i,freq),linestyle in zip(enumerate(frequencies), line_styles):
    #    ax.plot(angles_rad, amplitudes[freq]-amplitudes[freq][0], 'k',linestyle=linestyle, label=f"{freq / 1000:.0f}")
    #for i,freq in enumerate(frequencies):
    #    ax.plot(angles_rad, amplitudes[freq]-amplitudes[freq][0], label=f"{freq / 1000 }")
    
    # Improve plot aesthetics
    #ax.set_title(f'Thymio Speaker Polar Plot of RMS Amplitudes\nfor ukon{ukon_number}')
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_theta_offset(np.pi / 2)  # Start from the top
    ax.set_ylim(-20, 10)
    #ax.set_yticks(fontsizse=15)
    ax.set_thetamax(np.pi)
    ax.set_thetamin(0)
    ax.set_ylabel('dB RMS',fontsize=labelsize)
    ax.set_yticks([10,5,0,-5,-10,-15,-20])
    #ax.set_yticklabels(['5','','','','','0','','','','','-5','','','','','-10',-20],fontsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)
    ax.set_xticks(np.deg2rad(np.arange(0, 181, 30)))
    ax.set_xticklabels([f"{angle}°" for angle in range(0, 181, 30)],fontsize=labelsize)
    ax.legend(loc='upper right', title="Frequency \nBand [kHz] ", fontsize=legendsize, title_fontsize= legendsize)
    
    plt.savefig(f'polar plots/ukon{ukon_number}/sum_ukon{ukon_number}.png', dpi=300, bbox_inches='tight')
    plt.show()
