#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Author Name"
__copyright__ = "Copyright Info"
__credits__ = ["Author", "Author"]
__license__ = "License Name and Info"
__version__ = "0.0.0"
__maintainer__ = "Author"
__email__ = "contact@email.com"
__status__ = "status of the file"

"""
Video Creation Script

This script creates videos starting from raw frames from camera using ffmpeg.

"""
import os
import subprocess
import shutil

def create_video_from_frames(orig_file_dir, frames_dir, output_video, fps):
    """
    Creates a video from image frames using ffmpeg.
    Each frame is shown for exactly 1/fps seconds.
    Original files are copied into a new directory with sequential names.
    """

    frames = sorted([
        f for f in os.listdir(orig_file_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not frames:
        raise ValueError(f"No image frames found in {orig_file_dir}")

    # Ensure output frames directory exists
    os.makedirs(frames_dir, exist_ok=True)

    # Copy/rename sequentially
    ext = os.path.splitext(frames[0])[1].lower()  # keep original extension type
    for i, frame in enumerate(frames, start=1):
        new_name = f"frame_{i:06d}{ext}"  # e.g. frame_000001.jpg
        src = os.path.join(orig_file_dir, frame)
        dst = os.path.join(frames_dir, new_name)
        if not os.path.exists(dst):
            shutil.copy(src, dst)

    # ffmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-i", os.path.join(frames_dir, f"frame_%06d{ext}"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vf", f"fps={fps}",  # force exactly fps frames per second
        output_video,
    ]

    subprocess.run(cmd, check=True)


# Example usage
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    orig_frames_dir = "./Data/InputData/2025-09-24/2025-09-24_20-04-23/"
    frames_dir = "./Data/IntermediateData/Renamed_video_frames/2025-09-24_20-04-23/"
    video_dir = "./Data/IntermediateData/"

    # Check if an .mp4 file already exists in the video directory
    existing_mp4 = [f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')]

    if existing_mp4:
        print(f"\nExisting .mp4 file found in {video_dir}: \n{existing_mp4[0]}\n")
    else:
        create_video_from_frames(
            orig_frames_dir,
            frames_dir,
            "./Data/IntermediateData/2025-09-24_20-04-23.mp4",
        fps=15
        )

        print("Video created successfully.")