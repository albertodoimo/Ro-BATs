import pytta
import numpy as np

# List available devices
print(pytta.list_devices())

# Set default device
pytta.default()

# Sampling rate
fs = 44100

# Generate a sine wave
duration = 5  # duration in seconds
frequency = 1000  # frequency of the sine wave in Hz
amplitude = 0.5  # amplitude of the sine wave

t = np.linspace(0, duration, int(fs * duration), endpoint=False)
sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)

# make a sine sweep 
sweep = pytta.generate_sweep(20, 20000, duration, fs)
# Create a signal object
signal = pytta.SignalObj(sine_wave, samplingRate=fs)
signal = pytta.SignalObj(sweep, samplingRate=fs)

# Set up the measurement
ms = pytta.Measurement(
    samplingRate=fs,
    device=[1, 1],
    inChannel=[1, 2],
    outChannel=[1, 1],
    duration=duration,
    comment='Example sine wave measurement'
)

# Run the measurement
m1 = ms.run()

# Plot the recorded signal
m1.plot_time()
m1.plot_freq(smooth=True)

# make a sine sweep 
sweep = pytta.generate_sweep(20, 20000, duration, fs)