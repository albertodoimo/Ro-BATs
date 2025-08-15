from pypylon import pylon
import cv2
import numpy as np
import os
import yaml
import subprocess
from datetime import datetime
import queue
import csv
import pyscreenrec

# Specify the path to your YAML file
yaml_file = "./tracking/camera_calibration/calibration_matrix_basler_2560-1600.yaml"
# Run a subprocess to record all microphones and save to a file
# Example using arecord (Linux ALSA utility), adjust as needed for your setup

# subprocess.Popen(
#     ["python3", "/home/alberto/Documents/ActiveSensingCollectives_lab/Ro-BATs/measurements/z723/record_all_mics.py", "-dir", "/home/alberto/Documents/ActiveSensingCollectives_lab/Ro-BATs/tracking/pypylon/"],
# )\

# # Run the check_pi_sync.sh script and wait for it to finish before continuing
# subprocess.run(
#     ["bash", "/home/alberto/Documents/ActiveSensingCollectives_lab/Ro-BATs/tracking/check_pi_sync.sh"])

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
enable_screen_recording = True  # Set to True to enable screen recording (may cause issues on some systems)
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
arena_w = 1.47 # m
arena_l = 1.91 # m

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
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
# aruco_dict = cv2.aruco.extendDictionary(40, 3)
aruco_params = cv2.aruco.DetectorParameters()
print("Press ESC to exit...")

# Create output directory 
# video_fps = 10
# print (f"Video FPS set to: {1/video_fps}")
camera_fps = camera.ResultingFrameRate.GetValue()
print(f"Hardware Camera FPS output: {camera_fps}")

output_dir = './tracking/pypylon/data/'
# video_file_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S:%f')[:-2] + '_basler_tracking'
file_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S:%f')[:-2] + '_basler_tracking'
# if recording_bool == True:
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     out = cv2.VideoWriter(output_dir + video_file_name + '.MP4', fourcc, video_fps, (crop_w, crop_h))

# Set the upper limit of the camera's frame rate to 30 fps
camera.AcquisitionFrameRateEnable.Value = True
camera.AcquisitionFrameRate.Value = 20

data_queue = queue.Queue()

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
                except ValueError:
                    print('Marker 0, 1 or 2 not found')
            if pixel_per_meters > 0:
                break
            grab_result.Release()
except Exception as e:
    print(f"Error calculating pixel per meters: {e}")

def main():
    # Initialize screen recording variables
    # recorder = None
    # recording_started = False
    # screen_recording_enabled = False
    
    try:
        i = datetime.timestamp(datetime.now())
        while camera.IsGrabbing():
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grab_result.GrabSucceeded():
                image = converter.Convert(grab_result)
                frame = image.GetArray()
                original_frame = frame.copy()
                h, w = frame.shape[:2]

                corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
                timestamp = str(datetime.timestamp(datetime.now()))
                marker_pairs = [(4, 5), (6, 7), (10, 11)]
                robot_names = {(4, 5): "241", (6, 7): "240", (10, 11): "238"}

                marker_centers = get_marker_centers(corners, ids)
                id_list = ids.flatten().tolist() if ids is not None else []
                centers_dict = {id_: center for id_, center in zip(id_list, marker_centers)}


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
                
                pair_centers = get_pair_centers(marker_pairs, centers_dict, corners, ids, reference_position, pixel_per_meters)
                heading_vectors, pixel_centers, heading_angle = draw_heading_arrows(frame, pair_centers, robot_names, corners, ids, reference_position, pixel_per_meters)
                closest_robot_angle = draw_heading_angles(frame, heading_vectors, pixel_centers, robot_names)
                draw_pair_centers(frame, pair_centers, robot_names, reference_position, pixel_per_meters)
                intradistances = draw_closest_pair_line(frame, pair_centers, robot_names, reference_position, pixel_per_meters)
            

                # Save to npy file as dictionary (append mode)
                npy_file = output_dir + file_name + '_markers.npy'
                data_dict = {
                    'timestamp': timestamp,
                    'origin': (reference_position[0], reference_position[1]),
                    'robot_positions[m]': {
                        robot_names.get((a, b), f"{a}-{b}"): center
                        for (a, b), center in pair_centers.items()
                    },
                    'robot_heading_angles[deg]': {
                        robot_name: float(heading_angle) if heading_angle is not None else None
                        for robot_name, heading_angle in heading_angle.items()
                    },
                    'noise_on': True if (ids is not None and 200 in ids) else False
                }
                
                if not os.path.exists(npy_file):
                    np.save(npy_file, [data_dict])
                else:
                    existing_data = list(np.load(npy_file, allow_pickle=True))
                    existing_data.append(data_dict)
                    np.save(npy_file, existing_data)

                text_size, _ = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = 40
                cv2.putText(frame, timestamp, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                            1.3, (0, 0, 0), 3, cv2.LINE_AA)

                cv2.namedWindow("Basler Camera tracking", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Basler Camera tracking", 1600, 1200)
                # Show the camera window
                cv2.imshow("Basler Camera tracking", frame)

                # # Show the original camera feed
                # cv2.namedWindow("Basler Camera original", cv2.WINDOW_NORMAL)
                # cv2.resizeWindow("Basler Camera original", 1600, 1200)
                # cv2.imshow("Basler Camera original", original_frame)

                # # Write frame to video file if recording is enabled
                # if recording_bool and 'out' in globals():
                #     out.write(frame)

                # Stop recording and save data when ESC is pressed or window is closed
                if (cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty("Basler Camera tracking", cv2.WND_PROP_VISIBLE) < 1):
                    # csv_file = output_dir + file_name + '_markers.csv'
                    # with open(csv_file, 'w', newline='') as f:
                    #     writer = csv.writer(f)
                    #     writer.writerow(['timestamp[POSIX]', 'pair_centers[m]', 'distances[mm]', 'angle_results[deg]'])
                    #     while not data_queue.empty():
                    #         item = data_queue.get()
                    #         writer.writerow([
                    #             item['timestamp[POSIX]'],
                    #             item['pair_centers[m]'],
                    #             item['distances[mm]'],
                    #             item['angle_results[deg]']
                    #         ])
                    break

            grab_result.Release()
    finally:
        camera.StopGrabbing()
        camera.Close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
