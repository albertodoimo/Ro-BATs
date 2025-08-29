# This script tracks a robot's trajectory in videos using ArUco markers and saves the annotated videos.

"""
This script processes all .mp4 video files in a specified directory to track the trajectory of a robot identified by a specific ArUco marker ID.

Main functionalities:
- Iterates through all .mp4 files in the given camera_path directory.
- For each video, detects ArUco markers in every 10th frame using OpenCV starting from a 100 fps video from Basler camera.
- Identifies the marker corresponding to the specified robot_id.
- Tracks and draws the trajectory of the robot across frames.
- Saves the processed video with the drawn trajectory to an output directory.

"""

import cv2
from cv2 import aruco
import os
import traceback
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

robot_id = 6
camera_path = '/home/alberto/Videos/Basler Pyplon Viewer/'
file_names = os.listdir(camera_path)
file_names = [f for f in file_names if f.endswith('.mp4')]
pixel_per_meters_list = []
for file_name in file_names:
    print(file_name[:-4])
    video_path = camera_path + file_name
    cap = cv2.VideoCapture(video_path)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_fps = 10
    output_dir = './tracked_videos/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out = cv2.VideoWriter(output_dir + file_name, fourcc, video_fps, (frame_width, frame_height))
    frame_count = 0
    trajectory = np.zeros((0, 2), dtype=np.float32)
    try:
        while cap.isOpened():
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
                            cv2.namedWindow("Aruco tracking", cv2.WINDOW_NORMAL)
                            cv2.imshow("Aruco tracking", frame)
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

    cap.release()
    out.release()
cv2.destroyAllWindows()
