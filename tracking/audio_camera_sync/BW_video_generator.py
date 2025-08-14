import moviepy
import random
import numpy as np
import soundfile as sf
import csv
import os
from moviepy import ColorClip, ImageClip

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

    audio_data = np.concatenate([silence] + audio)
    sf.write(filename, 0.8 * audio_data, sample_rate)


if __name__ == "__main__":

    # Example parameters
    fps = 20  # minimum hardware sampling frequency
    P_min = 2*1/fps   # Minimum period in sec
    P_max = 4*P_min  # Maximum period in sec

    output_dir = os.path.join(os.path.dirname(__file__), "data/")
    os.makedirs(output_dir, exist_ok=True)

    # pseudo-random waiting periods
    periods = generate_pseudo_random_signal(P_min, P_max, 100)

    # Save wait periods to CSV
    with open(output_dir + "wait_periods.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["wait_period_seconds"])
        for period in periods:
            writer.writerow([period])

    # Alternating clips based on waiting periods
    clips = []
    colors = [(0, 0, 0), (255, 255, 255)]  # Black, White
    color_index = 1
    total_duration = 0
    max_duration = 30  # seconds
    x = 640 
    y = 480
    fs = 48000 # audio sample rate
    image_path = os.path.join(os.path.dirname(__file__), "marker_200.jpg")

    # 5-second countdown at the beginning
    countdown_clips = []
    for i in range(5, 0, -1):
        txt = f"{i}"
        countdown_clip = moviepy.TextClip(
            text= f"{i}",
            font_size=120,
            color='white',
            size=(x, y),
            bg_color='black',
            duration=1
        ).with_position(("center", "center"))
        countdown_clips.append(countdown_clip)

    for P_wait in periods:
        if color_index == 0:
            # Black clip
            clip = ColorClip(size=(x, y), color=colors[0], duration=P_wait)
        else:
            # Image clip
            # Create a white background
            bg_clip = ColorClip(size=(x, y), color=colors[1], duration=P_wait)
            # Overlay the image on the white background, centered
            img_clip = ImageClip(image_path, duration=P_wait)
            img_clip = img_clip.resized(height=y//1.5)  # Resize image to fit
            img_clip = img_clip.with_position(("center", "center"))
            # Add text over the image
            txt_clip = moviepy.TextClip(text ="DICT_6X6_250\nID: 200", font_size = 20, color='black', size=(x, y//5), duration=P_wait)
            txt_clip = txt_clip.with_position(("center", "top"))

            clip = moviepy.CompositeVideoClip([bg_clip, img_clip, txt_clip])
        clips.append(clip)
        color_index = 1 - color_index  # Alternate between black and image

    # Concatenate countdown and main clips
    final_clip = moviepy.concatenate_videoclips(countdown_clips + clips)

    # Generate audio and save to file
    generate_alternating_white_noise_audio(periods, fs, output_dir + "alternating_white_noise.wav")

    # Load the generated audio
    audio_clip = moviepy.AudioFileClip(output_dir + "alternating_white_noise.wav", fps=fs)

    # Set audio to video
    final_clip = final_clip.with_audio(audio_clip)
    # final_clip.preview(fps=30)  # Preview the video

    # Export video with audio
    final_clip.write_videofile(output_dir + "alternating_black_white_with_audio.mp4", fps=30, audio_fps=fs )
