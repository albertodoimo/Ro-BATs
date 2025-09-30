import random
import numpy as np
import soundfile as sf
import csv
import os
import scipy.signal as signal
random.seed(78464)

def calculate_wait_period(P_min, P_max):
    """
    Calculate timer waiting period 
    
    Args:
        P_min: Minimum period
        P_max: Maximum period
    
    Returns:
        P_wait: Calculated waiting period
    """
    
    # 32-bit random value (0 to 2^32-1)
    r = random.randint(0, 2**32 - 1)
    P_wait = P_min + (r / (2**32)) * (P_max - P_min)
    return P_wait

def generate_pseudo_random_signal(P_min, P_max, num_values):
    """
    Generate pseudo-random signal using 32-bit PRNG values.
    
    Args:
        P_min: Minimum period
        P_max: Maximum period
        num_values: Number of values to generate
    
    Returns:
        List of calculated waiting periods
    """
    wait_periods = []
    for _ in range(num_values):
        

        P_wait = calculate_wait_period(P_min, P_max) 
        wait_periods.append(P_wait)
    
    return wait_periods

def generate_alternating_white_noise_audio(wait_periods, sample_rate):
    """
    Generate an audio file with alternating white noise (on/off) based on wait_periods.

    Args:
        wait_periods: List of durations (seconds) for each segment.
        sample_rate: Audio sample rate (Hz).
    """
    audio = []
    noise_on = True
    
    # 5 seconds of silence at the beginning
    silence = np.zeros(int(5 * sample_rate), dtype=np.float32)

    for duration in wait_periods:
        num_samples = int(duration * sample_rate)
        if noise_on:
            segment = np.random.uniform(-1, 1, num_samples).astype(np.float32)
        else:
            segment = np.zeros(num_samples, dtype=np.float32)
        audio.append(segment)
        noise_on = not noise_on  # Alternate on/off

    noise = np.concatenate(audio)
    audio_data = np.concatenate([silence] + audio)
    return audio_data, noise



if __name__ == "__main__":

    # Example parameters
    fps = 15  # minimum hardware sampling frequency
    P_min = 2*1/fps   # Minimum period in sec
    P_max = 4*P_min  # Maximum period in sec

    output_dir = os.path.join(os.path.dirname(__file__), "data/")
    os.makedirs(output_dir, exist_ok=True)

    # pseudo-random waiting periods
    periods = generate_pseudo_random_signal(P_min, P_max, 20)

    # Save wait periods to CSV
    with open(output_dir + "wait_periods.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["wait_period_seconds"])
        for period in periods:
            writer.writerow([period])

    # Audio settings
    total_duration = 30*60  # seconds 
    noise_interval = 60  # seconds interval between start of the noise segments
    fs = 48000 # audio sample rate
    freq = 15  # Hz square wave frequency

    # Generate audio file with alternating white noise based on periods
    _, noise = generate_alternating_white_noise_audio(periods, fs)

    # Apply low-pass filter to the noise to avoid interference with robots DOA
    sos = signal.butter(8, 2000, 'low', fs=fs, output='sos')
    filtered_noise = signal.sosfiltfilt(sos, noise)

    # Repeat the one-minute noise + silence pattern to fill the total duration
    one_minute_noise_signal = np.concatenate((filtered_noise, np.zeros((int(fs * noise_interval) - filtered_noise.shape[0]))))
    sync_signal = np.tile(one_minute_noise_signal, total_duration // noise_interval)

    # generate 15 Hz square wave
    t = np.linspace(0, total_duration, int(fs * total_duration), endpoint=False)
    square_wave = 0.8 * signal.square(2 * np.pi * freq * t)

    # Stack into stereo: L=square wave, R=sync signal
    stereo = np.stack([square_wave, sync_signal], axis=1)

    # Save to WAV file
    sf.write(output_dir + f"{freq}Hz_tracking_sync_signal.wav", stereo, fs)

