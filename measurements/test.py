import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile  # Assuming noise recordings are in WAV format
from scipy.fft import fft

# Configuration
angles_deg = np.arange(0, 181, 30)  # Angles from 0° to 180° in 30° steps
angles_rad = np.deg2rad(angles_deg)  # Convert angles to radians

# Frequency of interest (in Hz)
frequencies = [1000, 2000, 3000, 4000]  # Frequencies at 51 kHz and 102 kHz

# Data storage for amplitude at each frequency
amplitudes = {freq: [] for freq in frequencies}

# Function to compute amplitude at a given frequency using FFT
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

# Process each angle's recording
for angle in angles_deg:
    # Load the noise recording for the current angle
    filename = f"/Users/alberto/Documents/UNIVERSITA/MAGISTRALE/tesi/github/white-noise-characterisation/ukon3/{angle}.wav" 
  
    sample_rate, data = wavfile.read(filename)
    
    # If stereo, use only one channel
    if len(data.shape) > 1:
        data = data[:, 0]
    
    # Calculate and store amplitude for each frequency of interest
    for freq in frequencies:
        amplitude = get_amplitude_at_frequency(data, sample_rate, freq)
        amplitudes[freq].append(amplitude)

print(f'amp',amplitudes)

# Normalize amplitudes to the 0° angle
for freq in frequencies:
    zero_deg_amplitude = amplitudes[freq][0]  # Amplitude at 0°
    print('0 deg amp', zero_deg_amplitude)
    amplitudes[freq] = [amp - zero_deg_amplitude for amp in amplitudes[freq]]

# Plotting
plt.figure()
ax = plt.subplot(111, projection='polar')

# Plot each frequency on the polar plot
for i, freq in enumerate(frequencies):
    ax.plot(angles_rad, amplitudes[freq], label=f"{freq / 1000:.0f} kHz")

# Additional plot settings
ax.set_theta_zero_location("N")  # Zero degrees at the top
ax.set_yticks([0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-20,-40])
ax.set_theta_direction(-1)       # Plot clockwise
ax.set_rlabel_position(225)      # Position for radial labels
plt.legend(loc="lower left")

# Show plot
plt.show()
