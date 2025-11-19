#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Created: 2025-11-15
Author: Alberto Doimo
email: alberto.doimo@uni-konstanz.de

Description:
Master Script for Video and Audio Processing Pipeline
"""

import subprocess
import sys
import os

# List of scripts to run in order
scripts = [
    # PROCESSING SCRIPTS
    "config.py",
    "Frames2Video.py",
    # "VideoUpsampling.py",
    # "AudioSync.py",
    # "Call_identification.py",
    # "Audio_Analysis.py",
    "Audio_video_tracking.py"
    
    # ANALYSIS SCRIPTS
    # "calc_dist_from_obst.py"
    # "summary_plots.py"
]

# Run each script
for script in scripts:
    print(f"Running {script}...")
    script_dir = os.path.dirname(os.path.abspath(__file__))+"/ProcessingScripts"
    script_path = os.path.join(script_dir, script)
    # result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, cwd=script_dir, shell=True)
    # To see output in real-time, use:
    result = subprocess.run([sys.executable, script_path], cwd=script_dir)
    
    if result.returncode != 0:
        print(f"Error running {script}:")
        print(result.stderr)
        break
    else:
        print(f"{script} completed successfully")
        print(result.stdout)