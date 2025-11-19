#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Created: 2025-10-1
Author: Alberto Doimo
email: alberto.doimo@uni-konstanz.de

Description:

Video Upsampling Script

This script upsamples videos to a specified frame rate using ffmpeg.

"""

import os
from Utils_SwarmTracking import*
from config import *

# Example usage
if __name__ == "__main__":
    fps = 60  # Desired upsampled frame rate

    # Check if an .mp4 file already exists in the video directory
    existing_mp4 = [f for f in os.listdir(interm_data_dir) if f.lower().endswith('.mp4')]

    if existing_mp4:
        print(f"\nExisting .mp4 file(s) in {interm_data_dir}:\n" + "\n".join(existing_mp4) + "\n")
        user_input = input("A video file already exists. Do you want to continue? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Operation cancelled by user.")
            exit(0)
        else:
            user_input = input(f"\nUpsampling rate: {fps}\n Confirm to continue? (y/n): ").strip().lower()
            if user_input != 'y':
                user_input = input(f"\nChange rate? (y/n)").strip().lower()
                if user_input != 'y':
                    print("Operation cancelled by user.")
                    exit(0)
                else:
                    new_fps = input(f"\nSet new rate: ").strip().lower()
                    upsample_video(
                        output_video_path,
                        output_video_path_upsampled,
                        fps=new_fps
                    )
            else:
                upsample_video(
                    output_video_path           ,
                    output_video_path_upsampled,
                    fps=fps
                )

            print("\n Video upsampled successfully \n")
