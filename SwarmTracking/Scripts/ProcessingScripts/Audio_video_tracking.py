#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Created: 2025-10-6
Author: Alberto Doimo
email: alberto.doimo@uni-konstanz.de

Description:

Video and audio analysis of the Ro-BATs experiments

"""
#%%
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import soundfile as sf
import os 
from Utils_SwarmTracking import *
import pandas as pd 
from pypylon import pylon
import cv2
import numpy as np
import os
import cv2
import sys

#%%
saving_bool = False # whether to save the output video with tracking

ips = [238, 240, 241]
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # project directory
input_dir = "Data/IntermediateData/"
date_dir = '2025-10-02/'
robot_rec_dir = '2025-10-02_18-45-28/'

file_name = '2025-09-24_20-04-23_upsampled'

output_video_path = os.path.join(project_dir, input_dir, file_name + '_tracked.mp4')

# # Look at every .mp4 file in the input_dir
# mp4_files = [f for f in os.listdir(os.path.join(project_dir,input_dir)) if f.lower().endswith('.mp4')]
# print("MP4 files in project_dir:", mp4_files)

video_path =  os.path.join(project_dir, input_dir, file_name) + '.mp4'
# audio_path = os.path.join(project_dir, input_dir, date_dir, robot_rec_dir, 'trimmed_audio/')

cap = cv2.VideoCapture(video_path)
video_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {video_fps}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
print(f"Video dimensions: {frame_width}x{frame_height}")

marker_pairs = [(4, 5), (6, 7), (10, 11)]

robot_names = {"241": (4, 5), "240": (6, 7), "238": (10, 11)}
color_map = {
    "241": (255, 0, 0),    # Blue
    "240": (0, 255, 0),    # Green
    "238": (0, 0, 255),    # Red
            }
arena_w = 1.47 # m
arena_l = 1.91 # m

# Load predefined dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
# Read CSV files containing robot positions for all detected IDs
robot_call_times = {}
csv_dir = os.path.join(project_dir, input_dir, date_dir, robot_rec_dir)
if os.path.exists(csv_dir):
    for robot_id in ips if 'ips' in locals() and ips is not None else []:
        csv_file = os.path.join(csv_dir, f"ip_{robot_id}_call_times.csv")
        if os.path.exists(csv_file):
            call_times = pd.read_csv(csv_file).to_numpy().flatten()
            robot_call_times[robot_id] = call_times
            print(f"Robot {robot_id} call times loaded:", call_times.shape)
        else:
            print(f"CSV file not found for robot {robot_id}:", csv_file)
else:
    print("CSV directory not found:", csv_dir)

#####################################################################################

# Audio Analysis

# robot_audio, fs = sf.read(audio_path)
# robot_audio = robot_audio[:, 0]
# print( 'Robot audio duration: %.1f [s]' % (len(robot_audio)/fs))

# fs = 48000
# dur = 20e-3
# hi_freq = 2e3
# low_freq = 20e3
# output_threshold = -50 # [dB]

# METHOD = 'das' # 'das', 'capon'
# if METHOD == 'das':
#     spatial_filter = das_filter

# t_tone = np.linspace(0, dur, int(fs*dur))
# chirp = signal.chirp(t_tone, hi_freq, t_tone[-1], low_freq)    

# C_AIR = 343
# min_distance = 8e-2 # [m]
# discarded_samples = int(np.floor(((min_distance + 2.5e-2)*2)/C_AIR*fs))
# max_distance = 1 # [m]
# max_index = int(np.floor(((max_distance + 2.5e-2)*2)/C_AIR*fs))

# def update(frame):
#     audio_data, _ = sf.read(audio_path, start=reading_points[frame], frames=offsets[frame])
#     dB_rms = 20*np.log10(np.mean(np.std(audio_data, axis=0)))    
#     if dB_rms > output_threshold:
#         filtered_signals = signal.correlate(audio_data, np.reshape(sig, (-1, 1)), 'same', method='fft')
#         roll_filt_sigs = np.roll(filtered_signals, -len(sig)//2, axis=0)            
#         try:
#             distance, direct_path, obst_echo = sonar(roll_filt_sigs, discarded_samples, max_index, fs)
#             distance = distance*100 # [m] to [cm]
#             theta, p = spatial_filter(
#                                         roll_filt_sigs[obst_echo - int(5e-4*fs):obst_echo + int(5e-4*fs)], 
#                                         fs=fs, nch=roll_filt_sigs.shape[1], d=2.70e-3, 
#                                         bw=(low_freq, hi_freq)
#                                     )
#             p_dB = 10*np.log10(p)
            
#             if direct_path != obst_echo:
#                 doa_index = np.argmax(p_dB)
#                 theta_hat = theta[doa_index]
#                 if distance > 0:            
#                     return distance, theta_hat
#                 else: return 0, 0
#             else: return 0, 0
#         except ValueError:
#             print('\nNo valid distance or DoA')
#             return 0, 0
#     else:
#             return 0, 0

##########################################################################################################
# First, determine the pixel to meter ratio using the ArUco markers
cap = cv2.VideoCapture(video_path)
try:

    pixel_per_meters = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect markers
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
        
        # Draw detected markers
        if ids is not None:
            corners_array = np.squeeze(np.array(corners))
            try:
                ind1 = np.where(ids == 1)[0]
                if len(ind1) == 0:
                    raise ValueError('Marker 0 not found')
                ind2 = np.where(ids == 2)[0]
                if len(ind2) == 0:
                    raise ValueError('Marker 1 not found')
                ind3 = np.where(ids == 3)[0]
                if len(ind3) == 0:
                    raise ValueError('Marker 2 not found')
                
                # bottom left of 1, top left of 2, top right of 3
                corners_1 = corners_array[ind1]
                corners_2 = corners_array[ind2]
                reference_position = corners_2[:, 2][0] # Use the bottom right corner of marker 2 as reference
                # print(f"Reference: {reference_position}")
                corners_3 = corners_array[ind3]
                pixel_per_meters = np.mean([np.linalg.norm(corners_1[:, 3] - corners_2[:, 0], axis=1)/arena_w, np.linalg.norm(corners_2[:, 0] - corners_3[:, 1], axis=1)/arena_l])
                print('Pixel per meters: %.2f' % pixel_per_meters)
            except ValueError as e:
                print('Error:', e)
        if pixel_per_meters > 0:
            break
        cap.release()
except Exception as e:
    print('Error reading video file:', e)
    sys.exit(1)
cap.release()
###########################################################################################################
#%%
     
def main():
    # Initialize screen recording variables
    # recorder = None
    # recording_started = False
    # screen_recording_enabled = False

    marker_pairs = [(4, 5), (6, 7), (10, 11)]
    robot_names = {"241": (4, 5), "240": (6, 7), "238": (10, 11)}
    cap = cv2.VideoCapture(video_path)

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (frame_width, frame_height))
    if not out.isOpened():
        print(f"Error: Could not open video writer at {output_video_path}")
        cap.release()
        return
    frame_count = 0

    try:
        while cap.isOpened():
            # Read the next frame from the video
            ret, frame = cap.read()
            
            # If no frame is returned, break the loop (end of video)
            if not ret:
                break
            # Create an ArUco marker detector with the specified dictionary and parameters
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            # Detect ArUco markers in the grayscale frame
            corners, ids, _ = detector.detectMarkers(frame)

            # If any markers are detected
            if ids is not None:
                marker_centers = get_marker_centers(corners, ids)
                id_list = ids.flatten().tolist() if ids is not None else []
                centers_dict = {id_: center for id_, center in zip(id_list, marker_centers)} # dictionary containing the center of each detected marker

                # Draw axes on the video to indicate increasing X (right) and Y (down) directions
                axis_length = 100  # pixels

                # X axis: right from reference_position
                x_axis_end = (int(reference_position[0] + axis_length), int(reference_position[1]))
                # Y axis: down from reference_position
                y_axis_end = (int(reference_position[0]), int(reference_position[1] + axis_length))

                # Draw X axis (red)
                cv2.arrowedLine(frame, tuple(reference_position.astype(int)), x_axis_end, (0, 0, 255), 4, tipLength=0.2)
                cv2.putText(frame, 'X', (x_axis_end[0] + 10, x_axis_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Draw Y axis (green)
                cv2.arrowedLine(frame, tuple(reference_position.astype(int)), y_axis_end, (0, 0, 255), 4, tipLength=0.2)
                cv2.putText(frame, 'Y', (y_axis_end[0], y_axis_end[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Draw tracking parameters on the frame
                robot_radius = 100 # pixels
                pair_centers = get_pair_centers(marker_pairs, centers_dict, corners, ids, reference_position, pixel_per_meters)
                draw_and_label_pair_centers(frame, pair_centers, robot_names, reference_position, pixel_per_meters, robot_radius)
                heading_vectors, heading_angle = draw_heading_arrows(frame, pair_centers, robot_names, corners, ids, reference_position, pixel_per_meters, robot_radius)
                draw_closest_robot_arrow(frame, reference_position , pixel_per_meters,heading_vectors,  pair_centers, robot_names, robot_radius)
                draw_closest_pair_line(frame, pair_centers, robot_names, reference_position, pixel_per_meters, robot_radius)


                # Check if frame_count matches closely any entry in robot_238_calls['call_time']
                for robot_id, call_times in robot_call_times.items():
                    if any(abs(frame_count / video_fps - t) < 0.01 for t in call_times):
                        # print(f"Robot {robot_id} call at frame {frame_count}, time {frame_count / video_fps}s")
                        pair = robot_names.get(str(robot_id))
                        if pair is not None:
                            center = pair_centers.get(pair)
                            if center is not None:
                                draw_center = (
                                    int(reference_position[0] + center[0] * pixel_per_meters),
                                    int(reference_position[1] + center[1] * pixel_per_meters)
                                )
                                # Use different colors for each robot
                                color = color_map.get(str(robot_id), (255, 255, 0))  
                                cv2.circle(frame, tuple(np.int32(draw_center)), robot_radius, color, -1)  # filled circle

                # pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
                # # pts = pts.reshape((-1,1,2))
                # cv2.polylines(frame,[pts],True,(0,255,255))

                # Save tracking data to a pandas DataFrame
                data = {
                    'origin_x': reference_position[0],
                    'origin_y': reference_position[1],
                }
                # Add robot positions (meters)
                for robot_name, pair in robot_names.items():
                    center = pair_centers.get(pair)
                    if center is not None:
                        data[f"{robot_name}_x_m"] = center[0]
                        data[f"{robot_name}_y_m"] = center[1]
                    else:
                        data[f"{robot_name}_x_m"] = None
                        data[f"{robot_name}_y_m"] = None

                # Add robot heading angles (degrees)
                for robot_name, angle in heading_angle.items():
                    data[f"{robot_name}_heading_deg"] = float(angle) if angle is not None else None

                # Append to CSV file
                df = pd.DataFrame([data])
                csv_path = os.path.join(project_dir, input_dir, file_name + '_data.csv')
                if not os.path.exists(csv_path):
                    df.to_csv(csv_path, index=False, mode='w')
                else:
                    df.to_csv(csv_path, index=False, mode='a', header=False)

                # Write frame to video file if recording is enabled
                if saving_bool and 'out' in globals():
                    out.write(frame)

                cv2.namedWindow(f"{file_name}", cv2.WINDOW_NORMAL)
                cv2.resizeWindow(f"{file_name}", 1600, 1200)
                # Show the camera window

                cv2.imshow(f"{file_name}", frame)

                frame_count += 1

                # Stop recording and save data when ESC is pressed or window is closed
                if (cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty(f"{file_name}", cv2.WND_PROP_VISIBLE) < 1):
                    print("Exiting...")
                    cap.release()
                    out.release()
                    cv2.destroyAllWindows()
                    break

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()    

if __name__ == "__main__":
    main()


# csv = pd.read_csv(os.path.join(output_dir, date_dir, robot_rec_dir, "ip_238_call_times.csv"), header=None)
# print(csv.shape)
# %%


# csv = pd.read_csv(os.path.join(output_dir, date_dir, robot_rec_dir, "ip_238_call_times.csv"), header=None)
# print(csv.shape)
# csv.T
 
# pd.DataFrame.to_csv(csv.T, 'test.csv', index=False, header=False)
