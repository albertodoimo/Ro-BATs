import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

def lms_adaptive_filter(x, d, M=4, mu=0.1):
    """
    Implements an LMS adaptive filter.
    
    Parameters:
        x  : Input signal (reference noise)
        d  : Desired signal (signal + noise)
        M  : Filter order (number of taps)
        mu : Step size (learning rate)
        
    Returns:
        y  : Filter output (estimated noise)
        e  : Error signal (cleaned signal)
        w  : Adaptive filter weights over iterations
    """
    N = len(x)
    w = np.zeros(M)  # Initialize filter weights
    y = np.zeros(N)  # Filter output
    e = np.zeros(N)  # Error signal (clean signal)
    
    for n in range(M, N):
        x_n = x[n:n-M:-1]  # Input vector (last M samples)
        y[n] = np.dot(w, x_n)  # Filter output
        e[n] = d[n] - y[n]  # Error signal
        w += 2 * mu * e[n] * x_n  # Weight update rule
        
    return y, e, w

# Simulate signal + noise
np.random.seed(42)
fs = 48000  # Sampling rate (Hz)
N = 96000  # Number of samples
t = np.arange(N) / fs  # Time vector 

signal = np.sin(2 * np.pi * 200 * t)  # Desired signal (200 Hz sine wave)
noise = np.random.normal(0, 0.3, N)  # Gaussian noise
desired = signal + noise  # Noisy signal

# Adaptive filtering
y_hat, error_signal, weights = lms_adaptive_filter(noise, desired, M=8, mu=0.01)

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(t, desired, label="Noisy Signal (d)")
plt.plot(t, signal, label="Original Signal (Reference)", alpha=0.7)
plt.legend()
plt.title("Original & Noisy Signal")

plt.subplot(3, 1, 2)
plt.plot(t, y_hat, label="Estimated Noise (y)")
plt.legend()
plt.title("Estimated Noise by LMS")

plt.subplot(3, 1, 3)
plt.plot(t, error_signal, label="Recovered Signal (e)")
plt.legend()
plt.title("Recovered Signal After Adaptive Filtering")
plt.xlabel("Time (s)")

plt.tight_layout()
plt.show()

# Play audio signals
print("Playing Noisy Signal...")
sd.play(desired, samplerate=fs)
sd.wait()

print("Playing Estimated Noise...")
sd.play(y_hat, samplerate=fs)
sd.wait()

print("Playing Recovered Signal...")
sd.play(error_signal, samplerate=fs)
sd.wait()
