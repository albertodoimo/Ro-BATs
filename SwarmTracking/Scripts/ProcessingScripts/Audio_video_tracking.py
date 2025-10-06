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

# arena dimensions


#%%

ips = [238, 240, 241]
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # project directory
input_dir = "Data/IntermediateData/"
date_dir = '2025-10-02/'
robot_rec_dir = '2025-10-02_18-45-28/'

file_name = '2025-09-24_20-04-23_upsampled'

# # Look at every .mp4 file in the input_dir
# mp4_files = [f for f in os.listdir(os.path.join(project_dir,input_dir)) if f.lower().endswith('.mp4')]
# print("MP4 files in project_dir:", mp4_files)

video_path =  os.path.join(project_dir, input_dir, file_name) + '.mp4'
# audio_path = os.path.join(project_dir, input_dir, date_dir, robot_rec_dir, 'trimmed_audio/')

cap = cv2.VideoCapture(video_path)
video_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {video_fps}")

screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
print(f"Video dimensions: {screen_width}x{screen_height}")

marker_pairs = [(4, 5), (6, 7), (10, 11)]

robot_names = {(4, 5): "241", (6, 7): "240", (10, 11): "238"}
arena_w = 1.47 # m
arena_l = 1.91 # m

# Load predefined dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# Loop through the video

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

def update(frame):
    audio_data, _ = sf.read(audio_path, start=reading_points[frame], frames=offsets[frame])
    dB_rms = 20*np.log10(np.mean(np.std(audio_data, axis=0)))    
    if dB_rms > output_threshold:
        filtered_signals = signal.correlate(audio_data, np.reshape(sig, (-1, 1)), 'same', method='fft')
        roll_filt_sigs = np.roll(filtered_signals, -len(sig)//2, axis=0)            
        try:
            distance, direct_path, obst_echo = sonar(roll_filt_sigs, discarded_samples, max_index, fs)
            distance = distance*100 # [m] to [cm]
            theta, p = spatial_filter(
                                        roll_filt_sigs[obst_echo - int(5e-4*fs):obst_echo + int(5e-4*fs)], 
                                        fs=fs, nch=roll_filt_sigs.shape[1], d=2.70e-3, 
                                        bw=(low_freq, hi_freq)
                                    )
            p_dB = 10*np.log10(p)
            
            if direct_path != obst_echo:
                doa_index = np.argmax(p_dB)
                theta_hat = theta[doa_index]
                if distance > 0:            
                    return distance, theta_hat
                else: return 0, 0
            else: return 0, 0
        except ValueError:
            print('\nNo valid distance or DoA')
            return 0, 0
    else:
            return 0, 0

# Define the arena pixel conversion in meters

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

#%%



# cap = cv2.VideoCapture(video_path)
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video_fps = 30
# output_dir = './blind_output/'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# out = cv2.VideoWriter(output_dir + file_name + '.mp4', fourcc, video_fps, (frame_width, frame_height))
# try:
#     frame_count = 0
#     trajectory = np.zeros((0, 2), dtype=np.float32)
#     counter = 0
#     true_counter = 0

#     obst_distances = []
#     dist_error = []
#     doas = []
#     doa_error = []
#     all_obstacles = False
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if frame_count == interp_video_frames[counter]:
#             counter += 1
#             if counter >= len(interp_video_frames):
#                 break
#             if frame_count in video_frames: 
#                 distance, doa = update(true_counter)
#                 true_counter += 1
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             # Detect markers
#             detector = aruco.ArucoDetector(aruco_dict, parameters)
#             corners, ids, _ = detector.detectMarkers(gray)
            
#             # Draw detected markers
#             if ids is not None:
#                 corners_array = np.squeeze(np.array(corners))
#                 # aruco.drawDetectedMarkers(frame, corners, ids)
#                 try:
#                     index = np.where(ids == robot_id)[0] # Find the index of the robot marker
#                     if len(index) == 0:
#                         raise ValueError('Robot marker not found')
#                     center = np.mean(corners_array[index], axis=1)[0]
#                     trajectory = np.append(trajectory, np.array([[center[0], center[1]]]), axis=0)
                    
#                     if distance != 0:
#                         mask = np.ones(len(ids), dtype=bool)
#                         mask[index] = False
#                         ind12 = np.where(ids == 12)[0]
#                         if len(ind12) > 0:
#                             mask[ind12] = False
#                         ind13 = np.where(ids == 13)[0]
#                         if len(ind13) > 0:
#                             mask[ind13] = False
#                         ind14 = np.where(ids == 14)[0]
#                         if len(ind14) > 0:
#                             mask[ind14] = False                    
#                         tl, tr, br, bl = np.squeeze(corners_array[index])
#                         mic_positions = np.astype(get_offset_point(center, tl, tr, offset=-pixel_per_meters*0.07), np.int32)                       

#                         obst_ids = ids[mask]
#                         indexes = np.argsort(obst_ids, axis=0)
#                         obst_corners = corners_array[mask]
#                         obst_centers = np.mean(obst_corners, axis=1)
#                         if (len(obst_centers) == 11 and not all_obstacles):
#                             all_obstacles = True
#                             print(obst_centers)
#                             obst_pos = np.squeeze(obst_centers.copy())[indexes]
#                             print(obst_pos)
#                             obst_list = np.squeeze(obst_pos).tolist()
#                             print('Position of all obstacles retrieved')
#                         obstacles, distances = shift_toward_point(obst_centers, mic_positions, 3.2, pixel_per_meters/100)
#                         if len(distances) > 0:
#                             D41 = tl - bl
#                             D14 = bl - tl
#                             D41_normalized = D41 / np.linalg.norm(D41)
#                             sorted_distances = np.sort(distances)
#                             found_nearest = False
#                             for sd in sorted_distances:
#                                 V_marker_space = np.squeeze(obstacles[np.where(distances == sd)]) - mic_positions
#                                 dot_product = np.dot(D41_normalized, V_marker_space)
#                                 cross_product = cross2d(V_marker_space, D14)
#                                 verse = -1 if cross_product > 0 else 1
#                                 angle = verse*np.arccos(dot_product / (np.linalg.norm(V_marker_space)))
#                                 # draw a line from the robot to the closest obstacle
#                                 if np.abs(angle) <= np.pi/2:               
#                                     closest_obstacle = np.squeeze(obstacles[np.where(distances == sd)])
#                                     found_nearest = True
#                                     break
#                             if found_nearest:
#                                 # print distance and angle                                
                                
#                                 # print('Time: %.1f [s]' % (frame_count/video_fps))
#                                 # print("Distance: %.1f [cm], Angle: %.1f [deg]" % (distance, doa))
#                                 # print("GT Distance: %.1f [cm], GT Angle: %.1f [deg]\n" % (sd, np.rad2deg(angle)))
#                                 obst_distances.append(sd)
#                                 dist_error.append(sd - distance)
#                                 doas.append(np.rad2deg(angle))
#                                 doa_error.append(np.rad2deg(angle) - doa)
#                                 cv2.arrowedLine(frame, mic_positions, closest_obstacle.astype(int), (255, 255, 0), 2)
#                                 # Draw the line
#                                 end_point = draw_line_with_angle(mic_positions, D41_normalized, distance, pixel_per_meters/100, doa)
#                                 cv2.arrowedLine(frame, mic_positions, end_point, (0, 255, 255), 2)
                
#                     if len(trajectory) > 2:
#                         # Draw trajectory
#                         for i in range(len(trajectory) - 1):
#                             cv2.line(frame, tuple(trajectory[i].astype(int)), tuple(trajectory[i + 1].astype(int)), (0, 255, 0), 2)
#                     # Add a legend for the two lines departing from the robot
#                     cv2.rectangle(frame, (50, 50), (620, 250), (0, 0, 0), -1)
#                     cv2.putText(frame, 'Ground truth', (int(80), int(100)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
#                     cv2.putText(frame, 'Estimated distance and direction', (int(80), int(200)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#                     out.write(frame)
#                     # Display result
#                     resized_frame = cv2.resize(frame, (screen_width, screen_height))
#                     cv2.imshow('ArUco Tracker', resized_frame)
#                 except Exception:
#                     traceback.print_exc()
#                     pass
#         frame_count += 1
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
# except Exception as e:
#     print(frame_count, e)
#     traceback.print_exc()
# # cv2.imwrite(file_name + '.jpg', resized_frame)
# cap.release()
# out.release()
# cv2.destroyAllWindows()
# data = {
#     'obstacle_distances': np.asarray(obst_distances).tolist(),
#     'distance_errors': np.asarray(dist_error).tolist(),
#     'obstacle_angles': np.asarray(doas).tolist(),
#     'angle_errors': np.asarray(doa_error).tolist(),
# }
# with open('./analysis/' + file_name + '.yaml', "w") as f:
#     yaml.dump(data, f)

# pos_dir = './obst_positions/'
# if not os.path.exists(pos_dir):
#     os.makedirs(pos_dir)
# pos_data = {
#     'obstacles_position': obst_list
# }
# with open(pos_dir + file_name + '.yaml', "w") as f:
#     yaml.dump(pos_data, f)



# csv = pd.read_csv(os.path.join(output_dir, date_dir, robot_rec_dir, "ip_238_call_times.csv"), header=None)
# print(csv.shape)