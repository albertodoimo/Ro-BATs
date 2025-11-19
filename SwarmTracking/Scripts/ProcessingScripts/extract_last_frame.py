import cv2
import os
from config import *
import sys
import pandas as pd

def extract_last_frame(video_path, output_path):
    """
    Extracts the last frame from a video file and saves it as an image.
    """
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # 1. Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    try:
        # 2. Get the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            print("Error: Could not determine frame count (video might be corrupted or stream format).")
            return

        print(f"Total frames detected: {total_frames}")

        # 3. Set the video position to the last frame
        # Frame indices are 0-based, so the last frame is total_frames - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

        # 4. Read the frame
        success, frame = cap.read()

        # RETRY LOGIC: 
        # Sometimes the reported frame count is slightly off by 1 or 2 frames 
        # depending on the video codec. If reading the exact last index fails,
        # we step back one frame and try again.
        if not success:
            print("Warning: Could not read exact last frame index. Trying previous frame...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 2)
            success, frame = cap.read()

        # 5. Save the result
        if success:
            cv2.imwrite(output_path, frame)
            print(f"Success! Last frame saved to: {output_path}")
        else:
            print("Error: Failed to extract the frame.")

    finally:
        # 6. Clean up resources
        cap.release()

# --- USAGE ---
if __name__ == "__main__":
    # Replace these with your actual file paths
    available_files = [f for f in os.listdir(interm_data_dir) if f.endswith('tracked.mp4')]

    if not available_files:
        print("No MP4 files found in the intermediate data directory.")
        sys.exit(1)

    # Extract date and time from filenames
    experiments = []
    for f in available_files:
        # Remove '.mp4' suffix
        name = f.replace('.mp4', '')
        # Split by underscore to get date and time
        parts = name.split('_')
        if len(parts) >= 2:
            experiments.append({'date': parts[0], 'time': parts[1], 'file': f})

    if not experiments:
        print("No valid experiment files found.")
        sys.exit(1)

    # Sort experiments by date and time
    experiments.sort(key=lambda x: (x['date'], x['time']))

    # Display options to user
    print("\nAvailable experiments:")
    for idx, exp in enumerate(experiments, 1):
        print(f"{idx}. {exp['date']}   {exp['time']}   {exp['file']}")

    # Get user selection
    while True:
        try:
            choice = int(input(f"\nSelect experiment (1-{len(experiments)}): "))
            if 1 <= choice <= len(experiments):
                selected_exp = experiments[choice - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(experiments)}")
        except ValueError:
            print("Please enter a valid number")

    experiment_date = selected_exp['date']
    experiment_time = selected_exp['time']
    input_video = os.path.join(interm_data_dir, selected_exp['file'])
    output_image = os.path.join(interm_data_dir, f"{experiment_date}_{experiment_time}_last_frame.png")

    print(f"Processing experiment: {experiment_date} at {experiment_time}")

    extract_last_frame(input_video, output_image) 