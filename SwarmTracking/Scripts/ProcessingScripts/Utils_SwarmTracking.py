#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Created: 2025-10-6
Author: Alberto Doimo
email: alberto.doimo@uni-konstanz.de

Description:

Support functions used into the SwarmTracking processing scripts.

"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import subprocess
import random
random.seed(78464)
import subprocess
import shutil
import os
import csv
import cv2

def save_data_to_csv(matrix, filename, path):
    """
    Save matrix as csv file
    Parameters:
    - matrix: matrix to save 
    - filename: name of the csv file
    - path: path to save the csv file
    """
    full_path = os.path.join(path, filename)
    with open(full_path, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(matrix)
    print(f"Matrix has been saved as csv to {full_path}")


# Define the matched filter function
def matched_filter(recording, chirp_template):
    """
    Apply matched filtering to the input recording using the provided chirp template.
    """
    filtered_output = np.roll(signal.correlate(recording, chirp_template, 'same', method='direct'), -len(chirp_template)//2)
    # filtered_output *= signal.windows.tukey(filtered_output.size, 0.1) # apply a Tukey window to reduce edge effects
    filtered_envelope = np.abs(signal.hilbert(filtered_output)) # compute the envelope of the matched filter output
    return filtered_envelope


# Detect peaks in the matched filter output
def detect_peaks(filtered_output, sample_rate):
    """
    Detect peaks in the matched filter output using prominence thresholding.
    """
    peaks, properties = signal.find_peaks(filtered_output, prominence=0.8) # prominence is calculated on normalized signal
    return peaks


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


def window(vec, fs, show = True):
    """
    Apply a Tukey window to the input vector and optionally display its waveform and spectrogram.
    
    Args:
        vec: Input signal vector
        fs: Sampling frequency
        show: Boolean flag to display plots

    Returns: Windowed signal normalized to its maximum value
    """
    window = signal.windows.tukey(len(vec), alpha=0.2)
    windowed_vec = vec * window

    if show:
        dur = len(windowed_vec) / fs
        t = np.linspace(0, dur, len(windowed_vec))
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(t, windowed_vec)
        plt.subplot(2, 1, 2)
        plt.specgram(windowed_vec, NFFT=256, Fs=192e3)
        plt.show()

    return windowed_vec/max(windowed_vec)


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


def generate_sweeps(frequencies, duration, fs, silence_dur):

    """
    Generate linear frequency sweeps for given frequencies.

    Args:
        frequencies: List of tuples (start_freq, end_freq).
        duration: Duration of each sweep (seconds).
        fs: Audio sample rate (Hz).
        silence_dur: Duration of silence after each sweep (milliseconds).

    Returns:
        List of audio signals representing the frequency sweeps.
    """

    t = np.linspace(0, duration, int(fs * duration))

    chirp_up = signal.chirp(t, f0=frequencies[0], f1=frequencies[1], t1=t[-1])
    sig1 = window(chirp_up, fs,show=True)

    silence_samples = int(silence_dur * fs/1000)
    silence_vec = np.zeros((silence_samples, ))

    chirp_down = signal.chirp(t, f0=frequencies[1], f1=frequencies[0], t1=t[-1])
    sig2 = window(chirp_down, fs, show=True)

    full_sig = np.concatenate((sig1, silence_vec,sig2))
    print('len = ', len(full_sig))

    return full_sig, sig1, sig2

########################################################################################
# OPEN CV UTILS
########################################################################################


def get_marker_centers(corners, ids):
    marker_centers = []
    if ids is not None:
        for marker_corners in corners:
            pts = marker_corners[0]
            center_x = int(np.mean(pts[:, 0]))
            center_y = int(np.mean(pts[:, 1]))
            marker_centers.append((center_x, center_y))
    return marker_centers

def get_pair_centers(marker_pairs, centers_dict, corners, ids, reference_position, pixel_per_meters):
    pair_centers = {}
    id_list = ids.flatten().tolist() if ids is not None else []
    for a, b in marker_pairs:
        if a in centers_dict and b in centers_dict:
            x = int((centers_dict[a][0] + centers_dict[b][0]) / 2)
            y = int((centers_dict[a][1] + centers_dict[b][1]) / 2)
            center = (x, y)
        elif a in centers_dict and a in id_list and pixel_per_meters > 0:
            # Place center 5 cm to the right of marker a, parallel to corners 0-1
            a_idx = id_list.index(a)
            a_corners = corners[a_idx][0]
            vec = a_corners[1] - a_corners[0]
            vec = vec / np.linalg.norm(vec)
            shift = vec * (pixel_per_meters * 0.05)  # 5 cm to the right
            x = int(centers_dict[a][0] + shift[0])
            y = int(centers_dict[a][1] + shift[1])
            center = (x, y)
        elif b in centers_dict and b in id_list and pixel_per_meters > 0:
            # Place center 5 cm to the left of marker b, parallel to corners 0-1
            b_idx = id_list.index(b)
            b_corners = corners[b_idx][0]
            vec = b_corners[1] - b_corners[0]
            vec = vec / np.linalg.norm(vec)
            shift = -vec * (pixel_per_meters * 0.05)  # 5 cm to the left
            x = int(centers_dict[b][0] + shift[0])
            y = int(centers_dict[b][1] + shift[1])
            center = (x, y)
        else:
            continue
        if reference_position is not None and pixel_per_meters > 0:
            rel_center = ((center[0] - reference_position[0]) / pixel_per_meters, 
                          (center[1] - reference_position[1]) / pixel_per_meters)
            pair_centers[(a, b)] = rel_center
        else:
            pair_centers[(a, b)] = center
    return pair_centers

def draw_heading_arrows(frame, pair_centers, robot_names, corners, ids, reference_position, pixel_per_meters):
    heading_vectors = {}
    pixel_centers = {}
    id_list = ids.flatten().tolist() if ids is not None else []
    for (a, b), center in pair_centers.items():
        heading_vec = None
        arrow_start = None

        # Try to get heading from marker a
        if a in id_list:
            a_idx = id_list.index(a)
            a_corners = corners[a_idx][0]
            pt0 = a_corners[0]
            pt3 = a_corners[3]
            heading_vec = pt0 - pt3
        # If marker a not found, try marker b
        elif b in id_list:
            b_idx = id_list.index(b)
            b_corners = corners[b_idx][0]
            pt0 = b_corners[0]
            pt3 = b_corners[3]
            heading_vec = pt0 - pt3

        if heading_vec is not None:
            heading_vec = heading_vec / np.linalg.norm(heading_vec)
            heading_vectors[(a, b)] = heading_vec
            if reference_position is not None and pixel_per_meters > 0:
                arrow_start = np.array([
                    int(reference_position[0] + center[0] * pixel_per_meters),
                    int(reference_position[1] + center[1] * pixel_per_meters)
                ])
            else:
                arrow_start = np.array(center)
            arrow_length = 100
            arrow_end = arrow_start + heading_vec * arrow_length

            # Heading angle: 0 deg is vertical (facing top), increases clockwise (0-360)
            heading_angle_rad = np.arctan2(heading_vec[1], heading_vec[0])  # negative y for top
            heading_angle_deg = (np.degrees(heading_angle_rad) + 90) % 360  # 0 deg is top
            if 'heading_angle' not in locals():
                heading_angle = {}
            robot_name = robot_names.get((a, b), f"{a}-{b}") if 'robot_names' in locals() else f"{a}-{b}"
            heading_angle[robot_name] = heading_angle_deg

            # cv2.putText(frame, f"{angle_deg:.1f} deg", (arrow_start[0], arrow_start[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.arrowedLine(frame, tuple(arrow_start.astype(int)), tuple(arrow_end.astype(int)), (255, 255, 255), 4, tipLength=0.25)
            # cv2.putText(frame, str(heading_vec), (arrow_start[0], arrow_start[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1 )
            pixel_centers[(a, b)] = arrow_start
        else:
            if reference_position is not None and pixel_per_meters > 0:
                pixel_centers[(a, b)] = np.array([
                    int(reference_position[0] + center[0] * pixel_per_meters),
                    int(reference_position[1] + center[1] * pixel_per_meters)
                ])
            else:
                pixel_centers[(a, b)] = np.array(center)


    return heading_vectors, pixel_centers, heading_angle



def draw_pair_centers(frame, pair_centers, robot_names, reference_position, pixel_per_meters):
    for (a, b), center in pair_centers.items():
        if reference_position is not None and pixel_per_meters > 0:
            draw_center = (int(reference_position[0] + center[0] * pixel_per_meters),
                            int(reference_position[1] + center[1] * pixel_per_meters))
        else:
            draw_center = center
        cv2.circle(frame, draw_center, 8, (0, 0, 255), 3)
        cv2.circle(frame, draw_center, 100, (255, 255, 255), 2)
        if (a, b) in robot_names:
            if reference_position is not None and pixel_per_meters > 0:
                coord_text = f"({center[0]:.3f}m, {center[1]:.3f}m)"
            else:
                coord_text = f"({draw_center[0]}, {draw_center[1]})"
            cv2.putText(frame, robot_names[(a, b)], (draw_center[0]-20, draw_center[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, coord_text, (draw_center[0]-20, draw_center[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

def draw_heading_angles(frame, heading_vectors, pair_centers, robot_names):

    angle_results = {}
    for (a, b), heading_vec in heading_vectors.items():
        this_center = np.array(pair_centers[(a, b)])
        min_dist = float('inf')
        closest_center = None
        for (other_a, other_b), other_center in pair_centers.items():
            if (other_a, other_b) == (a, b):
                continue
            dist = np.linalg.norm(this_center - np.array(other_center))
            if dist < min_dist:
                min_dist = dist
                closest_center = np.array(other_center)
                closest_robot = robot_names.get((other_a, other_b), f"{other_a}-{other_b}")
                
        closest_robot_angle = None
        if closest_center is not None:
            to_closest = closest_center - this_center
            if np.linalg.norm(to_closest) > 0:
                to_closest_norm = to_closest / np.linalg.norm(to_closest)
                angle_rad = np.arctan2(
                    to_closest_norm[1], to_closest_norm[0]
                ) - np.arctan2(heading_vec[1], heading_vec[0])
                closest_robot_angle = np.degrees(angle_rad) % 360
                text_pos = (int(this_center[0] + 40), int(this_center[1] - 20))
                cv2.putText(frame, f"{closest_robot_angle:.0f} deg", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Draw arrow towards closest robot
                arrow_length = 100
                arrow_end = this_center + to_closest_norm * arrow_length
                cv2.arrowedLine(frame, tuple(this_center.astype(int)), tuple(arrow_end.astype(int)), (0, 0, 255), 3, tipLength=0.25)
        angle_results[robot_names.get((a, b), f"{a}-{b}")] = {
            'closest_robot': closest_robot,
            'angle_deg': closest_robot_angle,
        }
    return angle_results

def draw_closest_pair_line(frame, pair_centers, robot_names, reference_position, pixel_per_meters):
    intradistances = []
    if len(pair_centers) >= 2:
        centers_list = list(pair_centers.values())
        names_list = [robot_names.get(pair, f"{pair[0]}-{pair[1]}") for pair in pair_centers.keys()]
        min_dist = float('inf')
        closest_pair = (None, None)
        for i in range(len(centers_list)):
            for j in range(i+1, len(centers_list)):
                dist = np.linalg.norm(np.array(centers_list[i]) - np.array(centers_list[j]))
                name_pair = (names_list[i], names_list[j])
                intradistances.append({
                    'pair': name_pair,
                    'distance_m': dist
                })
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (centers_list[i], centers_list[j])
        if closest_pair[0] is not None and closest_pair[1] is not None and reference_position is not None and pixel_per_meters > 0:
            pt1 = np.array([
                int(reference_position[0] + closest_pair[0][0] * pixel_per_meters),
                int(reference_position[1] + closest_pair[0][1] * pixel_per_meters)
            ])
            pt2 = np.array([
                int(reference_position[0] + closest_pair[1][0] * pixel_per_meters),
                int(reference_position[1] + closest_pair[1][1] * pixel_per_meters)
            ])
            vec = pt2 - pt1
            dist_px = np.linalg.norm(vec)
            if dist_px != 0:
                direction = vec / dist_px
                radius = 100
                start = pt1 + direction * radius
                end = pt2 - direction * radius
            else:
                start = pt1
                end = pt2
            cv2.line(frame, tuple(start.astype(int)), tuple(end.astype(int)), (255, 255, 255), 2)
            dist_text = f"{min_dist * 1000:.0f}mm"
            midpoint = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            cv2.putText(frame, dist_text, (midpoint[0]-20, midpoint[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return intradistances
