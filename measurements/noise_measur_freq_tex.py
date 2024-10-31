import numpy as np
import matplotlib.pyplot as plt
import soundfile as wav
from scipy.signal import butter, lfilter

# Set LaTeX rendering for text
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "text.latex.preamble": r"\usepackage{amsmath}"
})

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

R = 0.5  # Reference distance (if used for calculations)
angles = np.arange(0, 210, 30)

freq_bands = {
    '0-1kHz': (1, 1000),
    '1-2kHz': (1000, 2000),
    '2-3Hz': (2000, 3000),
    '3-4kHz': (3000, 4000)
}

for i in range(1, 7):
    db_values = []
    freq_bands_db = {band: [] for band in freq_bands.keys()}
    ukon_number = i

    for angle in angles:
        try:
            audio_signal, sample_rate = wav.read(f'/Users/alberto/Documents/UNIVERSITA/MAGISTRALE/tesi/github/white-noise-characterisation/ukon{ukon_number}/{angle}.wav')
        except FileNotFoundError:
            print(f"File for ukon{ukon_number} at angle {angle} not found.")
            continue

        rms_value = compute_rms(audio_signal)
        db_value = rms_to_db(rms_value)
        db_values.append(db_value)
        
        for band, (lowcut, highcut) in freq_bands.items():
            filtered_signal = butter_bandpass_filter(audio_signal, lowcut, highcut, sample_rate)
            band_rms = compute_rms(filtered_signal)
            band_db = rms_to_db(band_rms)
            freq_bands_db[band].append(band_db)

    freq_bands_db_ref = {band: values[0] for band, values in freq_bands_db.items()}
    angles_rad = np.deg2rad(angles)
    
    plt.figure(figsize=(20, 10))
    plt.suptitle(r'Thymio Speaker Polar Plot of dB RMS Amplitudes for {}$'.format(ukon_number), fontsize=16)

    # Plot for Overall dB RMS
    ax = plt.subplot(121, polar=True)
    ax.plot(angles_rad, db_values, 'b-', label=r'\textbf{Overall dB RMS}')
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_ylabel(r'\textbf{dB RMS}', fontsize=14)
    ax.set_xticks(np.deg2rad(np.arange(0, 181, 30)))
    ax.set_xticklabels([r"${}\degree$".format(angle) for angle in range(0, 181, 30)])
    ax.legend(loc='upper right')

    # Plot for frequency bands
    ax = plt.subplot(122, polar=True)
    for band, db_band in freq_bands_db.items():
        ax.plot(angles_rad, db_band - freq_bands_db_ref[band], '-', label=r'\textbf{Band} ' + band)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_ylabel(r'\textbf{dB RMS}', fontsize=14)
    ax.set_xticks(np.deg2rad(np.arange(0, 181, 30)))
    ax.set_xticklabels([r"${}\degree$".format(angle) for angle in range(0, 181, 30)])
    ax.legend(loc='upper right')

    plt.show()
