import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

# Function to compute RMS of audio signal

R = 0.5 #M
def compute_rms(audio_signal):
    return np.sqrt(np.mean(audio_signal**2))

def rms_to_db(rms_value):
    return 20 * np.log10(rms_value)

for i in range (1,7):
    angles = np.arange(0, 210, 30)
    # angles = np.concatenate((np.arange(0, 210, 30), np.arange(-150, 30, 30)))
    print(angles)
    db_values = []
    rms_values = []
    fftsig = []
    meansig = []
    norm_mean = []
    ukon_number = i
    print(i)
    for angle in angles:
        # Load the WAV file for the given angle
        sample_rate, audio_signal = wav.read(f'/Users/alberto/Documents/UNIVERSITA/MAGISTRALE/tesi/github/white-noise-characterisation/ukon{ukon_number}/{angle}.wav')
        
        #fftsig = np.fft.fft(audio_signal)  
        #plt.plot(fftsig) 
        #plt.show()
        #meansig = np.mean(4*np.pi*R*abs(fftsig))
        #print(meansig)
        #norm_mean = (meansig/np.max(meansig))
        #print(norm_mean)
        #db_scale = 20*np.log10(norm_mean)        # Compute the RMS value of the audio signal
        rms_value = compute_rms(audio_signal)
        rms_values.append(rms_value)
        db_value = rms_to_db(rms_value)
        db_values.append(db_value)
        #print(f'rms values {angle} degrees= ',rms_value)
        print(f'db values {angle}= ',db_value)
    # Convert angles to radians for the polar plot
    angles_rad = np.deg2rad(angles)
    ref_db = db_values[0]
    # Plot the RMS values on a polar plot
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles_rad, db_values-ref_db, 'b*-')  # blue dots and lines
    print(f'db values {angle}= ',db_value)

    # Optional: Improve plot aesthetics
    ax.set_title(f'Thymio speaker Polar Plot of RMS Amplitudes at Different Angles for ukon{ukon_number}')
    ax.set_theta_direction(-1)  # clockwise
    ax.set_theta_offset(np.pi / 2)  # start from the top
    ax.set_ylim(-6,2)
    ax.set_yticks([2,0,-2,-4])
    # Set ticks at every 30 degrees 
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 30)))
    ax.set_xticklabels([f"{angle}°" for angle in range(0, 360, 30)])
    #plt.show(False)
    plt.savefig(f'polar plots/rms_ukon{ukon_number}.png', dpi=300, bbox_inches='tight')
    plt.show()