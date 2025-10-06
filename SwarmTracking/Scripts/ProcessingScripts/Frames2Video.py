#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Created: 2025-10-1
Author: Alberto Doimo
email: alberto.doimo@uni-konstanz.de

Description:

Video Creation Script

This script creates videos starting from raw frames from camera using ffmpeg.

"""
import os
from Utils_SwarmTracking import*

# Example usage
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    orig_frames_dir = "./Data/InputData/2025-10-02/2025-10-02_18-45-28/"
    frames_dir = "./Data/IntermediateData/Renamed_video_frames/2025-10-02_18-45-28/"
    video_dir = "./Data/IntermediateData/"

    # Check if an .mp4 file already exists in the video directory
    existing_mp4 = [f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')]
    if existing_mp4:
        print(f"\nExisting .mp4 file found in {video_dir}: \n{existing_mp4[0]}\n")
        user_input = input("A video file already exists. Do you want to continue? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Operation cancelled by user.")
            exit(0)
        else:
            create_video_from_frames(
                orig_frames_dir,
                frames_dir,
                "./Data/IntermediateData/2025-10-02_18-45-28.mp4",
                fps=15
            )

            print("Video created successfully.")