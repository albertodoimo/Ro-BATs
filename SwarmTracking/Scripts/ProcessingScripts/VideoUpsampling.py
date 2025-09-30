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
Video Upsampling Script

This script upsamples videos to a specified frame rate using ffmpeg.

"""

import os
import subprocess

def upsample_video(input_video, output_video, fps):
    """
    Upsamples a video to the specified fps using ffmpeg's minterpolate filter.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_video,
        "-filter:v",
        f"minterpolate='mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps={fps}'",

        output_video,
    ]
    subprocess.run(cmd, check=True)


# Example usage
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    orig_file_dir = "./Data/InputData/2025-09-24/2025-09-24_20-04-23/"
    frames_dir = "./Data/IntermediateData/Renamed_video_frames/2025-09-24_20-04-23/"
    video_dir = "./Data/IntermediateData/"

    # Check if an .mp4 file already exists in the video directory
    existing_mp4 = [f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')]

    if existing_mp4:
        print(f"\nExisting .mp4 file found in {video_dir}: \n{existing_mp4[0]}\n")
        # Skip frame-to-video creation and proceed to upsampling
    else:
        upsample_video(
            "./Data/IntermediateData/2025-09-24_20-04-23.mp4",
            "./Data/IntermediateData/2025-09-24_20-04-23_upsampled.mp4",
            fps=60
        )

        print("Video upsampled successfully.")


# ffmpeg -i /home/alberto/Documents/ActiveSensingCollectives_lab/Documents/IROS 2025/Ro-bat_presentation_video.mp4 -filter:v "minterpolate='mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps=60'" output.mp4
