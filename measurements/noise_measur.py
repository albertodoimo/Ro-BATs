import numpy as np
import matplotlib.pyplot as plt
import soundfile as wav

# Function to compute RMS of audio signal

R = 0.5 #M
def compute_rms(audio_signal):
    return np.sqrt(np.mean(audio_signal**2))

def rms_to_db(rms_value):
    return 20 * np.log10(rms_value)

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


    #rms_ref_spl = compute_rms(one_Pa)
    #print('ref spl',rms_ref_spl)

    # Plot the RMS values on a polar plot
    plt.figure(figsize = (8, 8))
    ax = plt.subplot(111, polar = True)
    ax.plot(angles_rad, db_values-ref_db, 'b-')  # blue dots and lines
    #print(f'db values {angle}= ',db_value)

    # Optional: Improve plot aesthetics
    ax.set_title(f'Thymio speaker Polar Plot of RMS Amplitudes at Different Angles for ukon{ukon_number}')
    ax.set_theta_direction(-1)  # clockwise
    ax.set_theta_offset(np.pi / 2)  # start from the top
    #ax.set_ylim(-6,2)
    ax.set_thetamax(np.pi)
    ax.set_thetamin(0)
    ax.set_yticks([5,4,3,2,1,0,-1,-2,-3,-4,-5])
    # Set ticks at every 30 degrees
    ax.set_xticks(np.deg2rad(np.arange(0, 181, 30)))
    ax.set_xticklabels([f"{angle}°" for angle in range(0, 181, 30)])
    #plt.show(False)
    plt.savefig(f'polar plots/{ukon_number}/rms_ukon{ukon_number}.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)



    # Plot the SPL values on a polar plot
    plt.figure(figsize = (8, 8))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles_rad, 94+SPLs,'b-')  

    # Optional: Improve plot aesthetics
    ax.set_title(f'Thymio speaker Polar Plot of SPL Amplitudes at Different Angles for ukon{ukon_number} \n reference signal: 1kHz 1Pa tone, 94 dB SPL')
    ax.set_theta_direction(-1)  # clockwise
    ax.set_theta_offset(np.pi / 2)  # start from the top
    ax.set_ylim(0,70)
    ax.set_thetamax(np.pi)
    ax.set_thetamin(0)
    #ax.set_yticks([5,4,3,2,1,0,-1,-2,-3,-4,-5])
    # Set ticks at every 30 degrees 
    ax.set_xticks(np.deg2rad(np.arange(0, 181, 30)))
    ax.set_xticklabels([f"{angle}°" for angle in range(0, 181, 30)])
    #plt.savefig(f'polar plots/{ukon_number}/SPL_ukon{ukon_number}.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)

    # # Optional: Improve plot aesthetics
    # ax.set_title(f'Thymio speaker Polar Plot of RMS Amplitudes at Different Angles for ukon{ukon_number}')
    # ax.set_theta_direction(-1)  # clockwise
    # ax.set_theta_offset(np.pi / 2)  # start from the top
    # ax.set_ylim(-6,2)
    # ax.set_yticks([1,0,-1,-2,-3,-4,-5])
    # # Set ticks at every 30 degrees 
    # ax.set_xticks(np.deg2rad(np.arange(0, 360, 30)))
    # ax.set_xticklabels([f"{angle}°" for angle in range(0, 360, 30)])
    # #plt.show(False)
    # plt.savefig(f'polar plots/rms_ukon{ukon_number}.png', dpi=300, bbox_inches='tight')
    # plt.show()
