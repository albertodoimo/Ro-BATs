import cv2
from cv2 import aruco
import os
import traceback
import numpy as np
import pyautogui as pag
import sys
import yaml

os.chdir(os.path.dirname(os.path.abspath(__file__)))
save_dir = './trajectories_control/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

screen_width, screen_height = pag.size()
robot_id = 0
arena_w = 1.55  # [m]
arena_l = 2  # [m]
camera_path = './control/'
file_names = os.listdir(camera_path)
file_names = [f for f in file_names if f.endswith('.mp4')]
pixel_per_meters_list = []
for file_name in file_names:
    print(file_name[:-4])
    video_path = camera_path + file_name
    # cap = cv2.VideoCapture(video_path)
    # try:
    #     pixel_per_meters = 0
    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         # Detect markers
    #         detector = aruco.ArucoDetector(aruco_dict, parameters)
    #         corners, ids, _ = detector.detectMarkers(gray)
            
    #         # Draw detected markers
    #         if ids is not None:
    #             corners_array = np.squeeze(np.array(corners))
    #             try:
    #                 ind12 = np.where(ids == 12)[0]
    #                 if len(ind12) == 0:
    #                     raise ValueError('Marker 12 not found')
    #                 ind13 = np.where(ids == 13)[0]
    #                 if len(ind13) == 0:
    #                     raise ValueError('Marker 13 not found')
    #                 ind14 = np.where(ids == 14)[0]
    #                 if len(ind14) == 0:
    #                     raise ValueError('Marker 14 not found')
    #                 # bottom left of 12, top left of 13, top right of 14                
    #                 corners_12 = corners_array[ind12]
    #                 corners_13 = corners_array[ind13]
    #                 corners_14 = corners_array[ind14]
    #                 pixel_per_meters = np.mean([np.linalg.norm(corners_12[:, 3] - corners_13[:, 0], axis=1)/arena_w, np.linalg.norm(corners_13[:, 0] - corners_14[:, 1], axis=1)/arena_l])
    #                 print('Pixel per meters: %.2f' % pixel_per_meters)
    #             except ValueError:
    #                 print('Marker 12, 13 or 14 not found')
    #         if pixel_per_meters > 0:
    #             pixel_per_meters_list.append(pixel_per_meters)
    #             break
    # except:
    #     print('Error reading video file')
    #     sys.exit(1)
    # cap.release()
    cap = cv2.VideoCapture(video_path)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_fps = 30
    output_dir = './control_output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out = cv2.VideoWriter(output_dir + file_name + '.MP4', fourcc, video_fps, (frame_width, frame_height))
    frame_count = 0
    trajectory = np.zeros((0, 2), dtype=np.float32)
    try:
        while cap.isOpened():
            print(frame_count)
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 10 == 0:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
                    if ids is not None:
                        corners_array = np.squeeze(np.array(corners))
                        try:
                            index = np.where(ids == robot_id)[0] # Find the index of the robot marker
                            if len(index) == 0:
                                raise ValueError('Robot marker not found')
                            center = np.mean(corners_array[index], axis=1)[0]
                            trajectory = np.append(trajectory, np.array([[center[0], center[1]]]), axis=0)
                            if len(trajectory) > 2:
                                # Draw trajectory
                                for i in range(len(trajectory) - 1):
                                    cv2.line(frame, tuple(trajectory[i].astype(int)), tuple(trajectory[i + 1].astype(int)), (0, 255, 0), 2)
                            # Display result
                            out.write(frame)
                            resized_frame = cv2.resize(frame, (screen_width, screen_height))
                            cv2.imshow('ArUco Tracker', resized_frame)
                        except ValueError as e:
                            print(f"Error: {e}")
                            pass
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    traceback.print_exc()
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(frame_count, e)
        traceback.print_exc()
    # Save trajectory to YAML file
    # output_yaml = save_dir + file_name[:-4] + '_trajectory.yaml'
    # with open(output_yaml, 'a') as f:
    #     yaml.dump({'trajectory': trajectory.tolist()}, f)
    # print(f"Trajectory saved to {output_yaml}")
    cap.release()
    out.release()
cv2.destroyAllWindows()
# output_yaml = save_dir + 'conversion_factors.yaml'
# with open(output_yaml, 'w') as f:
#     yaml.dump({'pixel_to_meters': np.array(pixel_per_meters_list).tolist()}, f)
# print(f"Conversion factors saved to {output_yaml}")