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
from config import *

# Example usage
if __name__ == "__main__":

    # Check if an .mp4 file already exists in the video directory
    existing_mp4 = sorted([f for f in os.listdir(interm_data_dir) if f.lower().endswith('.mp4')])
    if existing_mp4:
        print(f"\nExisting .mp4 file(s) in {interm_data_dir}:\n" + "\n".join(existing_mp4) + "\n")
        user_input = input(f"A video file already exists. Do you want to continue creating {output_video_path}? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Operation cancelled by user.")
            exit(0)
        else:
            create_video_from_frames(
                orig_dir,
                temporary_frames_dir,
                output_video_path,
                fps=15
            )

            print("Video created successfully.")