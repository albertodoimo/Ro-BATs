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
from scipy.interpolate import interp1d
from functions import *
from functions.das_v2 import das_filter

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
    return filtered_output


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
    """
    Get the centers of detected ArUco markers.

    Args:
        corners: Detected marker corners from cv2.aruco.detectMarkers.
        ids: Detected marker IDs from cv2.aruco.detectMarkers.

    Returns:
        List of marker center coordinates (x, y).
    """
    marker_centers = []
    if ids is not None:
        for marker_corners in corners:
            pts = marker_corners[0]
            center_x = int(np.mean(pts[:, 0]))
            center_y = int(np.mean(pts[:, 1]))
            marker_centers.append((center_x, center_y))
    return marker_centers

def get_pair_centers(marker_pairs, centers_dict, corners, ids, reference_position, pixel_per_meters):
    """
    Calculates centers for each robot pair based on marker positions.
    If both markers are detected, uses their midpoint.
    If only one marker is detected, estimates the center 5 cm to the right (for marker a)
    or left (for marker b) along the line defined by corners 0-1.
    If neither marker is detected, skips that pair.

    Args:
        marker_pairs: List of tuples defining robot pairs by their marker IDs.
        centers_dict: Dictionary mapping marker IDs to their (x, y) pixel coordinates.
        corners: Detected marker corners from cv2.aruco.detectMarkers.
        ids: Detected marker IDs from cv2.aruco.detectMarkers.
        reference_position: (x, y) pixel coordinates of the reference position defined by carpet corners
        pixel_per_meters: Scaling factor from pixels to meters.

    Returns:
        pair_centers: Dictionary mapping robot pairs to their center coordinates (in pixels or meters).
    """
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

def draw_heading_arrows(frame, pair_centers, robot_names, corners, ids, reference_position, pixel_per_meters, robot_radius):

    """
    Draw heading direction arrows for each robot and calculates it relative to reference coordinates.
    
    Args:
        frame: Image frame to draw on.
        pair_centers: Dictionary mapping robot pairs to their center coordinates (in pixels or meters).
        robot_names: Dictionary mapping robot names to their marker ID pairs.
        corners: Detected marker corners from cv2.aruco.detectMarkers.
        ids: Detected marker IDs from cv2.aruco.detectMarkers.
        reference_position: (x, y) pixel coordinates of the reference position defined by carpet corners
        pixel_per_meters: Scaling factor from pixels to meters.
        robot_radius: Length of the circle around the robot (in pixels).
    
    Returns:
        heading_vectors: Dictionary mapping robot pairs to their heading direction vectors.
        heading_angle: Dictionary mapping robot names to their heading angles in degrees (0 deg is vertical, increases clockwise).
        
        """
    heading_vectors = {}
    id_list = ids.flatten().tolist() if ids is not None else []
    for (a, b), center in pair_centers.items():
        heading_vec = None
        arrow_start = None

        # Get heading direction from marker a or b
        # Heading vector is from corner 0 to corner 3 (top-left to bottom-left)

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
        
        # Draw direction arrow if heading vector is available
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
            arrow_length = robot_radius  # pixels
            arrow_end = arrow_start + heading_vec * arrow_length

            # Heading angle: 0 deg is vertical (facing top), increases clockwise (0-360)
            heading_angle_rad = np.arctan2(heading_vec[1], heading_vec[0])  # negative y for top
            heading_angle_deg = (np.degrees(heading_angle_rad) + 90) % 360  # 0 deg is top
            if 'heading_angle' not in locals():
                heading_angle = {}
                
            robot_name = [k for k, v in robot_names.items() if v == (a,b)]
            heading_angle[robot_name[0]] = heading_angle_deg

            # cv2.putText(frame, f"{heading_angle[robot_name[0]]:.1f} deg", (arrow_start[0] + robot_radius + 20, arrow_start[1] ), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.arrowedLine(frame, tuple(arrow_start.astype(int)), tuple(arrow_end.astype(int)), (255, 255, 255), 4, tipLength=0.25)

    return heading_vectors, heading_angle



def draw_and_label_pair_centers(frame, pair_centers, robot_names, reference_position, pixel_per_meters, robot_radius):
    """
    Draws circles at the centers of robot pairs and labels them with their names and coordinates.

    Args:
        frame: Image frame to draw on.
        pair_centers: Dictionary mapping robot pairs to their center coordinates (in pixels or meters).
        robot_names: Dictionary mapping robot names to their marker ID pairs.
        reference_position: (x, y) pixel coordinates of the reference position defined by carpet corners
        pixel_per_meters: Scaling factor from pixels to meters.
        robot_radius: Length of the circle around the robot (in pixels).
        
    """
    for (a, b), center in pair_centers.items():
        if reference_position is not None and pixel_per_meters > 0:
            draw_center = (int(reference_position[0] + center[0] * pixel_per_meters),
                            int(reference_position[1] + center[1] * pixel_per_meters))
        else:
            draw_center = center
        cv2.circle(frame, draw_center, 8, (0, 0, 255), 3)
        cv2.circle(frame, draw_center, robot_radius, (255, 255, 255), 2)

        robot_name = [k for k, v in robot_names.items() if v == (a,b)]
        cv2.putText(frame, f'IP: {robot_name[0]}', (draw_center[0] + robot_radius + 20, draw_center[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)


def draw_closest_robot_arrow(frame, reference_position, pixel_per_meters, heading_vectors, pair_centers, robot_names, robot_radius):
    """
    Draws arrows from each robot to its closest neighbor and calculates the angle between the robot's
    heading direction and the direction to the closest robot.
    
    Args:
        frame: Image frame to draw on.
        heading_vectors: Dictionary mapping robot pairs to their heading direction vectors.
        pair_centers: Dictionary mapping robot pairs to their center coordinates (in pixels or meters).
        robot_names: Dictionary mapping robot names to their marker ID pairs.
        robot_radius: Length of the circle around the robot (in pixels).

    """

    # Iterate over each robot's heading vector
    for (a, b), heading_vec in heading_vectors.items():
        this_center = np.array(pair_centers[(a, b)])
        min_dist = float('inf')
        closest_center = None
        closest_robot = None  # Ensure closest_robot is always defined
        # Find the closest robot by comparing distances to all other robots
        for (other_a, other_b), other_center in pair_centers.items():
            if (other_a, other_b) == (a, b):
                continue  # Skip self
            dist = np.linalg.norm(this_center - np.array(other_center))
            if dist < min_dist:
                min_dist = dist
                closest_center = np.array(other_center)
                closest_robot = [k for k, v in robot_names.items() if v == (other_a, other_b)]
                
        closest_robot_angle = None
        if closest_center is not None:
            # Compute vector pointing from this robot to the closest robot
            to_closest = closest_center - this_center
            if np.linalg.norm(to_closest) > 0:
                # Normalize the direction vector
                to_closest_norm = to_closest / np.linalg.norm(to_closest)
                # Calculate the angle between the robot's heading and the direction to the closest robot
                angle_rad = np.arctan2(
                    to_closest_norm[1], to_closest_norm[0]
                ) - np.arctan2(heading_vec[1], heading_vec[0])
                closest_robot_angle = np.degrees(angle_rad) % 360

                # Convert coordinates to pixel space if reference_position and pixel_per_meters are provided
                if reference_position is not None and pixel_per_meters > 0:
                    draw_center = (
                        int(reference_position[0] + this_center[0] * pixel_per_meters),
                        int(reference_position[1] + this_center[1] * pixel_per_meters)
                    )
                    arrow_end = np.array(draw_center) + to_closest_norm * robot_radius
                else:
                    draw_center = tuple(this_center.astype(int))
                    arrow_end = this_center + to_closest_norm * robot_radius


                # Draw the angle value on the frame
                text_pos = (draw_center[0] + robot_radius + 20, draw_center[1] + 50)
                cv2.putText(frame, f"{closest_robot_angle:.0f} deg", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Draw arrow towards closest robot
                cv2.arrowedLine(frame, draw_center, tuple(arrow_end.astype(int)), (0, 0, 255), 3, tipLength=0.25)
        else:
            print(f"No closest robot found for robot pair ({a}, {b})")


def draw_closest_pair_line(frame, pair_centers, robot_names, reference_position, pixel_per_meters, robot_radius):
    """
    Draws a line between the closest pair of robots and annotates the distance between them.
    Args:
        frame: Image frame to draw on.
        pair_centers: Dictionary mapping robot pairs to their center coordinates (in pixels or meters).
        robot_names: Dictionary mapping robot names to their marker ID pairs.
        reference_position: (x, y) pixel coordinates of the reference position defined by carpet corners
        pixel_per_meters: Scaling factor from pixels to meters.
        robot_radius: Length of the circle around the robot (in pixels).
    Returns:
        name_pair: Tuple of names of the closest robot pair.
        dist: Distance between the closest pair in meters.
    """

    if len(pair_centers) >= 2:
        centers_list = list(pair_centers.values())
        names_list = [robot_names.get(pair, f"{pair[0]}-{pair[1]}") for pair in pair_centers.keys()]
        min_dist = float('inf')
        closest_pair = (None, None)
        for i in range(len(centers_list)):
            for j in range(i+1, len(centers_list)):
                dist = np.linalg.norm(np.array(centers_list[i]) - np.array(centers_list[j]))
                name_pair = (names_list[i], names_list[j])
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
                radius = robot_radius 
                start = pt1 + direction * radius
                end = pt2 - direction * radius
            else:
                start = pt1
                end = pt2
            cv2.line(frame, tuple(start.astype(int)), tuple(end.astype(int)), (255, 255, 255), 2)
            dist_text = f"{min_dist * 1000:.0f}mm"
            midpoint = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            # cv2.putText(frame, dist_text, (midpoint[0]-20, midpoint[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return name_pair, dist

##########################################################################################
# AUDIO PROCESSING UTILS
##########################################################################################


def pascal_to_dbspl(X):
    '''
    Converts Pascals to dB SPL re 20 uPa
    '''
    return dB(X/20e-6)

def rms(X):
    return np.sqrt(np.mean(X**2))

dB = lambda X: 20*np.log10(abs(np.array(X).flatten()))

db_to_linear = lambda X: 10**(X/20)


# bin width clarification https://stackoverflow.com/questions/10754549/fft-bin-width-clarification
def get_rms_from_fft(freqs, spectrum, **kwargs):
    '''Use Parseval's theorem to get the RMS level of each frequency component
    This only works for RFFT spectrums!!!
    
    Parameters
    ----------
    freqs : (Nfreqs,) np.array >0 values
    spectrum : (Nfreqs,) np.array (complex)
    freq_range : (2,) array-like
        Min and max values
    
    Returns 
    -------
    root_mean_squared : float
        The RMS of the signal within the min-max frequency range
   
    '''
    minfreq, maxfreq = kwargs['freq_range']
    relevant_freqs = np.logical_and(freqs>=minfreq, freqs<=maxfreq)
    spectrum_copy = spectrum.copy()
    spectrum_copy[~relevant_freqs] = 0
    mean_sigsquared = np.sum(abs(spectrum_copy)**2)/spectrum.size
    root_mean_squared = np.sqrt(mean_sigsquared/(2*spectrum.size-1))
    return root_mean_squared


def calc_native_freqwise_rms(X, fs):
    '''
    Converts the FFT spectrum into a band-wise rms output. 
    The frequency-resolution of the spectrum/audio size decides
    the frequency resolution in general. 
    
    Parameters
    ----------
    X : np.array
        Audio
    fs : int
        Sampling rate in Hz
    
    Returns 
    -------
    fftfreqs, freqwise_rms : np.array
        fftfreqs holds the frequency bins from the RFFT
        freqwise_rms is the RMS value of each frequency bin. 
    '''
    rfft = np.fft.rfft(X)
    fftfreqs = np.fft.rfftfreq(X.size, 1/fs)
    # now calculate the rms per frequency-band
    # print('RFFT computation time:', time.time() - time1)
    freqwise_rms = []

    abs_rfft_squared = np.abs(rfft)**2
    mean_sq_freq = abs_rfft_squared / rfft.size
    rms_freq = np.sqrt(mean_sq_freq / (2*rfft.size-1))
    freqwise_rms = rms_freq.tolist()

    # freqwise_rms2 = []
    # for each in rfft:
    #     mean_sq_freq2 = np.sum(abs(each)**2)/rfft.size
    #     rms_freq2 = np.sqrt(mean_sq_freq2/(2*rfft.size-1))
    #     freqwise_rms2.append(rms_freq2)
    return fftfreqs, freqwise_rms

    

# Make an interpolation function 
def interpolate_freq_response(mic_freq_response, new_freqs):
    ''' 
    Parameters
    ----------
    mic_freq_response : tuple/list
        A tuple/list with two entries: (centrefreqs, centrefreq_RMS).
        
    new_freqs : list/array-like
        A set of new centre frequencies that need to be interpolated to. 

    Returns 
    -------
    tgtmicsens_interp : 
        
    Attention
    ---------
    Any frequencies outside of the calibration range will automatically be 
    assigned to the lowest sensitivity values measured in the input centrefreqs
    
    '''
    centrefreqs, mic_sensitivity = mic_freq_response 
    tgtmic_sens_interpfn = interp1d(centrefreqs, mic_sensitivity,
                                    kind='cubic', bounds_error=False,
                                    fill_value=np.min(mic_sensitivity))
    # interpolate the sensitivity of the mic to intermediate freqs
    tgtmicsens_interp = tgtmic_sens_interpfn(new_freqs)
    return tgtmicsens_interp



def update_das(buffer, fs, sos, ref, analyzed_buffer_time, tgtmic_relevant_freqs, interp_sensitivity, das_filter, channels, mic_spacing, highpass_freq, lowpass_freq, theta_das, critical_level, trigger_level, N_peaks):
    in_buffer = buffer
    # print('buffer queue time seconds=', start_time_1 - start_time)

    # Plot and save the raw input buffer for the reference channel
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 4))
    # plt.plot(in_buffer[:, self.ref])
    # plt.title('Raw Input Buffer - Reference Channel')
    # plt.xlabel('Sample')
    # plt.ylabel('Amplitude')
    # plt.tight_layout()
    # plt.savefig('raw_input_buffer_ref_channel.png')
    # plt.close()

    # Apply highpass filter to each channel using sosfiltfilt
    in_sig = signal.sosfiltfilt(sos, in_buffer, axis=0)

    # Apply matched filter to each channel separately

    # # Match filter the input with the output template to find similar sweeps
    # mf_signal = matched_filter(in_sig[:, self.ref], self.sweep)
    # peaks = detect_peaks(mf_signal, self.fs, prominence=0.5, distance=0.01)
    # if len(peaks) > 0:
    #     start_idx = peaks[0]
    #     end_idx = int(start_idx + self.sweep.shape[0])
    #     trimmed_input_peaks = in_sig[start_idx:end_idx, :]

    #     # plot trimmed input peaks 
    #     plt.figure(figsize=(10, 12))
    #     for ch in range(trimmed_input_peaks.shape[1]):
    #         ax = plt.subplot(trimmed_input_peaks.shape[1], 1, ch + 1, sharey=None if ch == 0 else plt.gca())
    #         plt.plot(trimmed_input_peaks[:, ch])
    #         plt.title(f'Trimmed Input Peaks - Channel {ch+1}')
    #         plt.xlabel('Sample')
    #         plt.grid(True)
    #         if ch == 0:
    #             plt.ylabel('Amplitude')
    #     plt.tight_layout()
    #     plt.savefig('trimmed_input_peaks_ref_channel.png')
    #     plt.close()

    # Filter the input with its envelope but without signal reference template
    filtered_envelope = np.abs(signal.hilbert(in_sig[:, ref], axis=0))
    # peaks = detect_peaks(in_sig[:, self.ref], self.fs, prominence=0.5, distance=0.01)

    max_envelope_idx = np.argmax(filtered_envelope)
    max_envelope_value = filtered_envelope[max_envelope_idx]
    # print('Max envelope value:', max_envelope_value)

    # Trim around the max
    trim_ms = analyzed_buffer_time # ms
    trim_samples = int(fs * trim_ms)
    half_trim = trim_samples // 2
    trimmed_signal = np.zeros((trim_samples, in_sig.shape[1]), dtype=in_sig.dtype)
    
    # Ensure trimmed_signal always has exactly trim_samples rows (matching trim_ms duration)
    if max_envelope_idx - half_trim < 0:
        start_idx = 0
        end_idx = trim_samples
    elif max_envelope_idx + half_trim > in_sig.shape[0]:
        end_idx = in_sig.shape[0]
        start_idx = end_idx - trim_samples
    else:
        start_idx = max_envelope_idx - half_trim
        end_idx = start_idx + trim_samples
    trimmed_signal = in_sig[start_idx:end_idx, :]

    # # plot trimmed input peaks 
    # plt.figure(figsize=(10, 12))
    # for ch in range(trimmed_signal.shape[1]):
    #     ax = plt.subplot(trimmed_signal.shape[1], 1, ch + 1, sharey=None if ch == 0 else plt.gca())
    #     plt.plot(trimmed_signal[:, ch])
    #     plt.title(f'Trimmed Input Peaks - Channel {ch+1}')
    #     plt.xlabel('Sample')
    #     plt.grid(True)
    #     if ch == 0:
    #         plt.ylabel('Amplitude')
    # plt.tight_layout()
    # plt.savefig('trimmed_input.png')
    # plt.close()


    # print('matched filter time seconds=', start_time_3 - start_time_2)

    # Plot matched filtered signal and detected peaks for the reference channel
    # plt.figure(figsize=(10, 4))
    # plt.plot(in_sig[:, self.ref], label='Matched Filtered Envelope')
    # plt.plot(peaks, mf_signal[peaks], "rx", label='Detected Peaks')
    # plt.title('Matched Filtered Envelope and Detected Peaks - Reference Channel')
    # plt.xlabel('Sample')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('matched_filtered_peaks_ref_channel.png')
    # plt.close()


    # Plot the filtered signal for the reference channel
    # plt.figure(figsize=(10, 4))
    # plt.plot(in_sig[:, self.ref])
    # plt.title('Filtered Signal - Reference Channel')
    # plt.xlabel('Sample')
    # plt.ylabel('Amplitude')
    # plt.tight_layout()
    # plt.savefig('filtered_signal_ref_channel.png')
    # plt.close()

    # centrefreqs_list = []
    # freqrms_list = []
    # for ch in range(in_sig.shape[1]):
    #     centrefreqs_ch, freqrms_ch = calc_native_freqwise_rms(in_sig[:, ch], self.fs)
    #     centrefreqs_list.append(centrefreqs_ch)
    #     freqrms_list.append(freqrms_ch)
    # centrefreqs = np.array(centrefreqs_list).T
    # freqrms = np.array(freqrms_list).T
    
    centrefreqs, freqrms = calc_native_freqwise_rms(trimmed_signal[:, ref], fs)
    freqwise_Parms = freqrms/interp_sensitivity
    # print('rms freqwise time =', start_time_4 - start_time_3)
    # # Calculate and save the average noise spectrum (ANS) figure

    # plt.figure(figsize=(10, 4))
    # plt.plot(centrefreqs, freqwise_Parms)
    # plt.title('freqwise_Parms')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Amplitude')
    # plt.tight_layout()
    # plt.savefig('filtered_spectrum.png')
    # plt.close()

    # Compute total RMS for each channel separately over the relevant frequency band
    # total_rms_freqwise_Parms = []
    # for ch in range(freqwise_Parms.shape[1]):
    #     relevant = self.tgtmic_relevant_freqs[:, ch]
    #     total_rms = np.sqrt(np.sum(freqwise_Parms[relevant, ch]**2))
    #     total_rms_freqwise_Parms.append(total_rms)
    # total_rms_freqwise_Parms = np.array(total_rms_freqwise_Parms)

    total_rms_freqwise_Parms = np.sqrt(np.sum(freqwise_Parms[tgtmic_relevant_freqs]**2))
    dB_SPL_level = pascal_to_dbspl(total_rms_freqwise_Parms) #dB SPL level for reference channel

    # print('time to calculate dB SPL =', time.time() - start_time_4)

    theta, spatial_resp, f_spec_axis, spectrum, bands = das_filter(trimmed_signal, fs, channels, mic_spacing, [highpass_freq, lowpass_freq], theta=theta_das)

    # print('freq axis', f_spec_axis.shape, 'bands shape', bands.shape, 'spectrum shape', spectrum.shape)

    # plt.figure(figsize=(10, 4))
    # plt.plot(f_spec_axis, np.abs(spectrum[:, self.ref, :]))
    # plt.title('DAS Spectrum')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Amplitude ')
    # plt.tight_layout()
    # plt.savefig('das_spectrum.png')
    # plt.close()


    #spatial_resp = gaussian_filter1d(spatial_resp, sigma=4)
    peaks, _ = signal.find_peaks(spatial_resp)  # Adjust height as needed
    # peak_angle = theta_das[np.argmax(spatial_resp)]
    peak_angles = theta[peaks]
    N = N_peaks # Number of peaks to keep

    # Sort peaks by their height and keep the N largest ones
    peak_heights = spatial_resp[peaks]
    top_n_peak_indices = np.argsort(peak_heights)[-N:]  # Indices of the N largest peaks # Indices of the N largest peaks
    top_n_peak_indices = top_n_peak_indices[::-1]
    peak_angles = theta[peaks[top_n_peak_indices]]  # Corresponding angles
    # print('peak angles', f"{peak_angles[0]}, dB SPL: {dB_SPL_level}, peak heights: {peak_heights[top_n_peak_indices]}\n")

    return peak_angles[0], dB_SPL_level

    # if dB_SPL_level > trigger_level or dB_SPL_level > critical_level:
    #     return peak_angles[0], dB_SPL_level
    # else:
    #     peak_angles = None
    #     return peak_angles, dB_SPL_level
