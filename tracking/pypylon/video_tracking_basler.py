from pypylon import pylon
import cv2
import numpy as np
import os
import yaml
import subprocess
from datetime import datetime
import queue
import csv

# Specify the path to your YAML file
yaml_file = "./tracking/camera_calibration/calibration_matrix_basler_2560-1600.yaml"
# Run a subprocess to record all microphones and save to a file
# Example using arecord (Linux ALSA utility), adjust as needed for your setup

# subprocess.Popen(
#     ["python3", "/home/alberto/Documents/ActiveSensingCollectives_lab/Ro-BATs/measurements/z723/record_all_mics.py", "-dir", "/home/alberto/Documents/ActiveSensingCollectives_lab/Ro-BATs/tracking/pypylon/"],
# )\

try:
    with open(yaml_file, 'r') as file:
        # Load the YAML content into a Python variable (usually a dict or list)
        data = yaml.safe_load(file)
        camera_matrix = np.array(data['camera_matrix'])
        dist_coeffs = np.array(data['dist_coeff'])
except FileNotFoundError:
    print(f"Error: The file '{yaml_file}' does not exist.")
except yaml.YAMLError as exc:
    print("Error parsing YAML file:", exc)


# Setup the camera
# camera model = ace2 R a2A4508-20umBAS
recording_bool = False
tl_factory = pylon.TlFactory.GetInstance()
devices = tl_factory.EnumerateDevices()
if not devices:
    print("No Basler camera found.")

camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
camera.Open()

# Set camera parameters
# Original image size
original_width = 4504
original_height = 4096
# Crop size
crop_w = 2560
crop_h = 1600

# Arena dimensions in meters from the marks on the carpet
arena_w = 1.47
arena_l = 1.91

camera.Width.SetValue(crop_w)
camera.Height.SetValue(crop_h)

# Center crop into the original image
camera.BslCenterX.Execute()
camera.BslCenterY.Execute()
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

# Convert Basler images to OpenCV format
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# Load ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
# aruco_dict = cv2.aruco.extendDictionary(40, 3)
aruco_params = cv2.aruco.DetectorParameters()
print("Press ESC to exit...")

# Create output directory 
video_fps = 10
print (f"Video FPS set to: {1/video_fps}")
camera_fps = camera.ResultingFrameRate.GetValue()
print(f"Hardware Camera FPS output: {camera_fps}")

output_dir = './tracking/pypylon/videos/'
video_file_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S:%f')[:-2] + '_basler_tracking'
file_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S:%f')[:-2] + '_basler_tracking'
if recording_bool == True:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out = cv2.VideoWriter(output_dir + video_file_name + '.MP4', fourcc, video_fps, (crop_w, crop_h))

# Set the upper limit of the camera's frame rate to 30 fps
camera.AcquisitionFrameRateEnable.Value = True
camera.AcquisitionFrameRate.Value = 20

data_queue = queue.Queue()

# arena dimensions
try:
    pixel_per_meters = 0
    camera.TimestampLatch.Execute()
    i = datetime.timestamp(datetime.now())
    while camera.IsGrabbing():
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grab_result.GrabSucceeded():
            image = converter.Convert(grab_result)
            frame = image.GetArray()
            h, w = frame.shape[:2]

            if 'mapx' not in locals():
                new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
                mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_16SC2)
                
            undistorted = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR)

            corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
            
            # Draw detected markers
            if ids is not None:
                corners_array = np.squeeze(np.array(corners))
                try:
                    ind0 = np.where(ids == 1)[0]
                    if len(ind0) == 0:
                        raise ValueError('Marker 0 not found')
                    ind1 = np.where(ids == 2)[0]
                    if len(ind1) == 0:
                        raise ValueError('Marker 1 not found')
                    ind2 = np.where(ids == 3)[0]
                    if len(ind2) == 0:
                        raise ValueError('Marker 2 not found')
                    # bottom left of 1, top left of 2, top right of 3
                    corners_0 = corners_array[ind0]
                    corners_1 = corners_array[ind1]
                    corners_2 = corners_array[ind2]
                    pixel_per_meters = np.mean([np.linalg.norm(corners_0[:, 3] - corners_1[:, 0], axis=1)/arena_w, np.linalg.norm(corners_1[:, 0] - corners_2[:, 1], axis=1)/arena_l])
                    print('Pixel per meters: %.2f' % pixel_per_meters)
                except ValueError:
                    print('Marker 0, 1 or 2 not found')
            if pixel_per_meters > 0:
                break
            grab_result.Release()
except Exception as e:
    print(f"Error calculating pixel per meters: {e}")


def main():
    try:
        # camera.TimestampLatch.Execute()
        i = datetime.timestamp(datetime.now())
        while camera.IsGrabbing():
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grab_result.GrabSucceeded():
                image = converter.Convert(grab_result)
                frame = image.GetArray()
                h, w = frame.shape[:2]

                # if 'mapx' not in locals():
                #     new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
                #     mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_16SC2)
                    
                # undistorted = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR)

                corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
                timestamp = str(datetime.timestamp(datetime.now()))
                marker_centers = []
                marker_pairs = [(4, 5), (6, 7), (10, 11)]
                pair_centers = {}

                if ids is not None:
                    for marker_corners in corners:
                        pts = marker_corners[0]
                        center_x = int(np.mean(pts[:, 0]))
                        center_y = int(np.mean(pts[:, 1]))
                        marker_centers.append((center_x, center_y))
                
                    id_list = ids.flatten().tolist()
                    centers_dict = {id_: center for id_, center in zip(id_list, marker_centers)}
                    for a, b in marker_pairs:
                        if a in centers_dict and b in centers_dict:
                            x = int((centers_dict[a][0] + centers_dict[b][0]) / 2)
                            y = int((centers_dict[a][1] + centers_dict[b][1]) / 2)
                            pair_centers[(a, b)] = (x, y)
                        elif a in centers_dict:
                            a_idx = id_list.index(a)
                            a_corners = corners[a_idx][0]
                            vec = a_corners[1] - a_corners[0]
                            vec = vec / np.linalg.norm(vec)
                            shift = vec * 60
                            x = int(centers_dict[a][0] + shift[0])
                            y = int(centers_dict[a][1] + shift[1])
                            pair_centers[(a, b)] = (x, y)
                        elif b in centers_dict:
                            b_idx = id_list.index(b)
                            b_corners = corners[b_idx][0]
                            vec = b_corners[0] - b_corners[1]
                            vec = vec / np.linalg.norm(vec)
                            shift = vec * 60
                            x = int(centers_dict[b][0] + shift[0])
                            y = int(centers_dict[b][1] + shift[1])
                            pair_centers[(a, b)] = (x, y)

                    robot_names = {(4, 5): "241", (6, 7): "240", (10, 11): "238"}
                    
                    for (a, b), center in pair_centers.items():
                        cv2.circle(frame, center, 8, (0, 0, 255), 3)
                        cv2.circle(frame, center, 100, (255, 255, 255), 2)
                        if (a, b) in robot_names:
                            cv2.putText(frame, robot_names[(a, b)], (center[0]-20, center[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                    # --- Draw a line between the two closest pair centers ---
                    if len(pair_centers) >= 2:
                        centers_list = list(pair_centers.values())
                        names_list = [robot_names.get(pair, f"{pair[0]}-{pair[1]}") for pair in pair_centers.keys()]
                        min_dist = float('inf')
                        closest_pair = (None, None)
                        intradistances = []
                        for i in range(len(centers_list)):
                            for j in range(i+1, len(centers_list)):
                                dist = np.linalg.norm(np.array(centers_list[i]) - np.array(centers_list[j]))
                                name_pair = (names_list[i], names_list[j])
                                intradistances.append({
                                    'pair': name_pair,
                                    'distance_m': dist / pixel_per_meters if pixel_per_meters > 0 else None
                                })
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_pair = (centers_list[i], centers_list[j])
                        if closest_pair[0] is not None and closest_pair[1] is not None:
                            # Draw only the middle fifth of the line between the two closest pair centers
                            pt1 = np.array(closest_pair[0])
                            pt2 = np.array(closest_pair[1])
                            # Draw the line from the edge of the first circle to the edge of the second circle (outer circles)
                            vec = pt2 - pt1
                            dist = np.linalg.norm(vec)
                            if dist != 0:
                                direction = vec / dist
                                radius = 100  # same as the circle radius used above
                                start = pt1 + direction * radius
                                end = pt2 - direction * radius
                            else:
                                start = pt1
                                end = pt2
                            cv2.line(frame, tuple(start.astype(int)), tuple(end.astype(int)), (255, 255, 255), 2)
                            min_dist_m = min_dist / pixel_per_meters if pixel_per_meters > 0 else 0
                            dist_text = f"{min_dist_m*1000:.0f}mm"
                            midpoint = ((closest_pair[0][0] + closest_pair[1][0]) // 2, (closest_pair[0][1] + closest_pair[1][1]) // 2)
                            cv2.putText(frame, dist_text, (midpoint[0]-20, midpoint[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    data_queue.put({
                        'timestamp': timestamp,
                        'ids': ids.tolist() if ids is not None else [],
                        'pair_centers': {
                            robot_names.get((a, b), f"{a}-{b}"): center
                            for (a, b), center in pair_centers.items()
                        },
                        'distances': intradistances if len(pair_centers) >= 2 else []
                    })

                text_size, _ = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = 40
                # cv2.putText(undistorted, timestamp, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                #             1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, timestamp, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                            1.3, (0, 0, 0), 3, cv2.LINE_AA)
                
                cv2.namedWindow("Basler Camera original", cv2.WINDOW_NORMAL)
                # Set window size to half of your screen (example: 1280x800)
                cv2.resizeWindow("Basler Camera original", 1280, 800)
                cv2.imshow("Basler Camera original", frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    csv_file = output_dir + file_name + '_markers.csv'
                    with open(csv_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['timestamp', 'ids', 'pair_centers', 'intradistances'])
                        while not data_queue.empty():
                            item = data_queue.get()
                            writer.writerow([
                                item['timestamp'],
                                item['ids'],
                                item['pair_centers'],
                                item['distances']
                            ])
                    break

            grab_result.Release()
    finally:
        camera.StopGrabbing()
        camera.Close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
