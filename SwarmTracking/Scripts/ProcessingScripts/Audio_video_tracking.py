#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Created: 2025-10-6
Author: Alberto Doimo
email: alberto.doimo@uni-konstanz.de

Description:

Video and audio tracking of the Ro-BATs experiments

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
saving_bool = True # whether to save the output video with tracking

###############################################################
# Experimental parameters
# Threshold dB SPL levels for DOA 
trigger_level =  70 # dB SPL attraction level
critical_level = 75 # dB SPL repulsion level

# Aruco markers pair corresponding to each robot
marker_pairs = [(4, 5), (6, 7), (10, 11)]
robot_names = {"241": (4, 5), "240": (6, 7), "238": (10, 11)}

# Color map for different robot IDs
color_map = {
    "241": (255, 255, 0),    # Cyan
    "240": (0, 255, 0),    # Green
    "238": (0, 0, 255),    # Red
            }

# Arena dimensions in meters
arena_w = 1.47 # m
arena_l = 1.91 # m

# List of robot IPs (IDs) to process
ips = [238, 240, 241]

# Set the project directory (three levels up from this script)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # project directory

###############################################################
# Directories
# Intermediate data directory
input_dir = "Data/IntermediateData/"

# Date directory
date_dir = '2025-10-02/'

# Directory for robot recordings (date and time)
robot_rec_dir = '2025-10-02_18-45-28/'

# Base file name for video 
file_name = '2025-10-02_18-45-28_upsampled'

# Output video path
output_video_path = os.path.join(project_dir, input_dir, file_name + '_tracked.mp4')

# Video path 
video_path = os.path.join(project_dir, input_dir, file_name) + '.mp4'

###############################################################
# Open the video file and get its properties
cap = cv2.VideoCapture(video_path)
video_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {video_fps}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
print(f"Video dimensions: {frame_width}x{frame_height}")

# Load predefined dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

#################################################################
# Read CSV files containing robot positions for all detected IDs
robot_call_times = {}
robot_DOA_angle = {}
robot_dB_SPL_level = {}
csv_dir = os.path.join(project_dir, input_dir, date_dir, robot_rec_dir)
if os.path.exists(csv_dir):
    for robot_id in ips if 'ips' in locals() and ips is not None else []:
        csv_file = os.path.join(csv_dir, f"ip_{robot_id}_audio_tracking.csv")
        if os.path.exists(csv_file):
            call_times = pd.read_csv(csv_file, usecols=[0]).to_numpy().flatten()
            print(call_times)
            DOA_angle = pd.read_csv(csv_file, usecols=[1]).to_numpy().flatten()
            dB_SPL_level = pd.read_csv(csv_file, usecols=[2]).to_numpy().flatten()

            robot_call_times[robot_id] = call_times
            robot_DOA_angle[robot_id] = DOA_angle
            robot_dB_SPL_level[robot_id] = dB_SPL_level

            print(f"Robot {robot_id} call times loaded:", call_times.shape)
            print(f"Robot {robot_id} DOA angle loaded:", DOA_angle.shape)
            print(f"Robot {robot_id} dB SPL level loaded:", dB_SPL_level.shape)
        else:
            print(f"CSV file not found for robot {robot_id}:", csv_file)
else:
    print("CSV directory not found:", csv_dir)

##########################################################################################################
# Determine the pixel to meter ratio using the ArUco markers
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
                print(f"Reference: {reference_position}")
                corners_3 = corners_array[ind3]
                pixel_per_meters = np.mean([np.linalg.norm(corners_1[:, 3] - corners_2[:, 0], axis=1)/arena_w, np.linalg.norm(corners_2[:, 0] - corners_3[:, 1], axis=1)/arena_l])
                print('Pixel per meters: %.2f' % pixel_per_meters)
            except ValueError as e:
                print('Error:', e ,'\n')
                pixel_per_meters = 1034.67
                reference_position = np.array([305, 76])
                print('Using default pixel_per_meters:', pixel_per_meters)
                print('Using default reference_position:', reference_position)

        if pixel_per_meters > 0:
            break
        cap.release()
except Exception as e:
    print('Error reading video file:', e)
    sys.exit(1)
cap.release()

#%%
###########################################################################################################  
# Main processing loop for video tracking
def main():
    # Open the video file for reading and create a VideoWriter for saving the tracked output
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (frame_width, frame_height))

    # Check if the VideoWriter was successfully opened
    if not out.isOpened():
        print(f"Error: Could not open video writer at {output_video_path}")
        cap.release()
        return

    frame_count = 0
    try:
        axis_length = 100  # pixels
        robot_radius = 100  # pixels
        DOA_arrow_length = 150  # pixels

        # Prepare CSV path and header
        audio_csv_path = os.path.join(project_dir, input_dir, file_name + '_tracking_data.csv')
        # If file exists, rename it to avoid overwriting
        if os.path.exists(audio_csv_path):
            backup_path = audio_csv_path.replace('.csv', '_backup.csv')
            os.rename(audio_csv_path, backup_path)
            print(f"Existing tracking data CSV renamed to: {backup_path}")
        write_header = True  # Always write header for the new file

        # Prepare columns for each robot: x_238, y_238, doa_238, spl_238, ...
        columns = ['frame', 'frame_second', 'origin_x', 'origin_y']
        for robot_id in ips:
            columns += [
                f'x_{robot_id}', f'y_{robot_id}', f'doa_{robot_id}', f'spl_{robot_id}', f'call_time_{robot_id}'
            ]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, _ = detector.detectMarkers(frame)
            pair_centers = {}
            heading_vectors = {}
            heading_angle = {}

            if ids is not None:
                marker_centers = get_marker_centers(corners, ids)
                id_list = ids.flatten().tolist() if ids is not None else []
                centers_dict = {id_: center for id_, center in zip(id_list, marker_centers)}

                x_axis_end = (int(reference_position[0] + axis_length), int(reference_position[1]))
                y_axis_end = (int(reference_position[0]), int(reference_position[1] + axis_length))
                cv2.arrowedLine(frame, tuple(reference_position.astype(int)), x_axis_end, (0, 0, 255), 4, tipLength=0.2)
                cv2.putText(frame, 'X', (x_axis_end[0] + 10, x_axis_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.arrowedLine(frame, tuple(reference_position.astype(int)), y_axis_end, (0, 0, 255), 4, tipLength=0.2)
                cv2.putText(frame, 'Y', (y_axis_end[0], y_axis_end[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                pair_centers = get_pair_centers(marker_pairs, centers_dict, corners, ids, reference_position, pixel_per_meters)
                draw_and_label_pair_centers(frame, pair_centers, robot_names, reference_position, pixel_per_meters, robot_radius)
                heading_vectors, heading_angle = draw_heading_arrows(frame, pair_centers, robot_names, corners, ids, reference_position, pixel_per_meters, robot_radius)
                # - Draw arrow to closest robot from reference
                # draw_closest_robot_arrow(frame, reference_position, pixel_per_meters, heading_vectors, pair_centers, robot_names, robot_radius)
                # - Draw line between closest robot pair
                # draw_closest_pair_line(frame, pair_centers, robot_names, reference_position, pixel_per_meters, robot_radius)

            # Prepare data for this frame
            frame_row = {
                'frame': frame_count,
                'frame_second': frame_count / video_fps,
                'origin_x': reference_position[0],
                'origin_y': reference_position[1]
            }

            for robot_id in ips:
                robot_id_str = str(robot_id)
                pair = robot_names.get(robot_id_str)
                center = pair_centers.get(pair) if pair is not None else None
                heading_deg = heading_angle.get(robot_id_str, None)
                heading_vec = heading_vectors.get(pair) if pair is not None else None

                # Defaults
                x_val = ''
                y_val = ''
                doa_val = ''
                spl_val = ''

                # If center is available, calculate positions
                if center is not None:
                    x_val = center[0]
                    y_val = center[1]

                # Find audio info for this frame
                call_times = robot_call_times.get(robot_id, [])
                DOA_angle = robot_DOA_angle.get(robot_id, [])
                dB_SPL_level = robot_dB_SPL_level.get(robot_id, [])
                frame_time = frame_count / video_fps + 0.13  # Adjust frame time for synchronization
                idx_candidates = [i for i, t in enumerate(call_times) if abs(frame_time - t) < 0.01]
                call_time_val = ''
                if idx_candidates:
                    idx = idx_candidates[0]
                    doa_val = DOA_angle[idx]
                    spl_val = dB_SPL_level[idx]
                    call_time_val = call_times[idx]

                    color = color_map.get(robot_id_str, (255, 255, 0))
                    if center is not None:
                        # Draw filled circle for detected call
                        draw_center_x = int(reference_position[0] + center[0] * pixel_per_meters)
                        draw_center_y = int(reference_position[1] + center[1] * pixel_per_meters)
                        cv2.circle(frame, (draw_center_x, draw_center_y), robot_radius, color, -1)
                        # Draw DOA arrow if SPL above threshold
                        if spl_val > trigger_level and heading_vec is not None:
                            theta = np.deg2rad(doa_val)
                            rot_matrix = np.array([
                                [np.cos(theta), -np.sin(theta)],
                                [np.sin(theta),  np.cos(theta)]
                            ])
                            doa_vec = rot_matrix @ heading_vec
                            arrow_dx = int(DOA_arrow_length * doa_vec[0])
                            arrow_dy = int(DOA_arrow_length * doa_vec[1])
                            arrow_end = (
                                draw_center_x + arrow_dx,
                                draw_center_y + arrow_dy
                            )
                            cv2.arrowedLine(frame, (draw_center_x, draw_center_y), arrow_end, color, 6, tipLength=0.25)
                            cv2.putText(frame, f"DOA: {doa_val:.1f}", (draw_center_x + robot_radius + 20, draw_center_y + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
                        # Always show dB SPL value near the robot
                        cv2.putText(frame, f"{spl_val:.1f} dB SPL", (draw_center_x + robot_radius + 20, draw_center_y + 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

                # Add to frame row
                frame_row[f'x_{robot_id}'] = x_val
                frame_row[f'y_{robot_id}'] = y_val
                frame_row[f'doa_{robot_id}'] = doa_val
                frame_row[f'spl_{robot_id}'] = spl_val
                frame_row[f'call_time_{robot_id}'] = call_time_val

            # Save info for this frame to CSV
            audio_df = pd.DataFrame([frame_row], columns=columns)
            audio_df.to_csv(audio_csv_path, index=False, mode='a', header=write_header)
            write_header = False  # Only write header for first frame

            if saving_bool:
                out.write(frame)
            cv2.namedWindow(f"{file_name}", cv2.WINDOW_NORMAL)
            cv2.imshow(f"{file_name}", frame)

            frame_count += 1

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


# %%
