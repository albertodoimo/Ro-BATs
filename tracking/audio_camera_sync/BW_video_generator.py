import moviepy
import random
import numpy as np
import soundfile as sf
import csv
import os

def calculate_wait_period(P_min, P_max):
    """
    Calculate timer waiting period 
    
    Args:
        P_min: Minimum period
        P_max: Maximum period
    
    Returns:
        P_wait: Calculated waiting period
    """
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
        # Generate 32-bit random value (0 to 2^32-1)

        P_wait = calculate_wait_period(P_min, P_max)
        wait_periods.append(P_wait)
    
    return wait_periods

def generate_alternating_white_noise_audio(wait_periods, sample_rate, filename):
    """
    Generate an audio file with alternating white noise (on/off) based on wait_periods.

    Args:
        wait_periods: List of durations (seconds) for each segment.
        sample_rate: Audio sample rate (Hz).
        filename: Output WAV file name.
    """
    audio = []
    noise_on = True
    for duration in wait_periods:
        num_samples = int(duration * sample_rate)
        if noise_on:
            segment = np.random.uniform(-1, 1, num_samples).astype(np.float32)
        else:
            segment = np.zeros(num_samples, dtype=np.float32)
        audio.append(segment)
        noise_on = not noise_on  # Alternate on/off

    audio_data = np.concatenate(audio)
    sf.write(filename, 0.8*audio_data, sample_rate)


if __name__ == "__main__":
    # Example parameters
    fps = 20  # minimum hardware sampling frequency
    P_min = 2*1/fps   # Minimum period in sec
    P_max = 4*P_min  # Maximum period in sec

    output_dir = os.path.join(os.path.dirname(__file__), "data/")
    os.makedirs(output_dir, exist_ok=True)
    # Generate some pseudo-random waiting periods
    periods = generate_pseudo_random_signal(P_min, P_max, 100)

    with open(output_dir + "wait_periods.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["wait_period_seconds"])
        for period in periods:
            writer.writerow([period])

    # Generate alternating black and white clips based on waiting periods
    clips = []
    colors = [(0, 0, 0), (255, 255, 255)]  # Black, White
    color_index = 1
    total_duration = 0
    max_duration = 30  # seconds

    for P_wait in periods:
        # print(f"Creating clip with duration: {P_wait:.2f} seconds, color: {'Black' if color_index == 0 else 'White'}")
        clip = moviepy.ColorClip(size=(640, 480), color=colors[color_index], duration=P_wait)
        clips.append(clip)
        color_index = 1 - color_index  # Alternate color

    final_clip = moviepy.concatenate_videoclips(clips)

    # Generate audio and save to file
    generate_alternating_white_noise_audio(periods, 48000, output_dir + "alternating_white_noise.wav")

    # Load the generated audio
    audio_clip = moviepy.AudioFileClip(output_dir + "alternating_white_noise.wav", fps=48000)

    # Set audio to video
    final_clip = final_clip.with_audio(audio_clip)
    # final_clip.preview(fps=30)  # Preview the video

    # Export video with audio
    final_clip.write_videofile(output_dir + "alternating_black_white_with_audio.mp4", fps=30)
