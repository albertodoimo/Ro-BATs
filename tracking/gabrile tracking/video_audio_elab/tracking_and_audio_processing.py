import cv2
from cv2 import aruco
import os
import traceback
import yaml
import numpy as np
import sys
import io
import ffmpeg
import soundfile as sf
import librosa
from scipy import signal
from das_v2 import das_filter
from capon import capon_method
from matplotlib import pyplot as plt

import ffmpeg

# Cross product in 2D
def cross2d(x, y):
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

def get_offset_point(center, top_left, top_right, offset=100):
    """
    Compute a point 100 pixels in the normal direction from the top edge
    of the ArUco marker.
    
    Parameters:
        center (tuple): (cx, cy), the center of the marker.
        top_left (tuple): (x, y) of the top-left corner.
        top_right (tuple): (x, y) of the top-right corner.
        offset (float): Distance in pixels to move from center (default 100).
    
    Returns:
        (x, y): The new 2D coordinates offset from the center.
    """
    # Convert to numpy arrays
    pt1 = np.array(top_left, dtype=np.float32)
    pt2 = np.array(top_right, dtype=np.float32)
    center = np.array(center, dtype=np.float32)

    # Direction vector of the top edge
    edge_vec = pt2 - pt1
    edge_vec /= np.linalg.norm(edge_vec)  # Normalize

    # Rotate 90 degrees counter-clockwise to get outward normal
    normal_vec = np.array([-edge_vec[1], edge_vec[0]])

    # Compute the new point
    new_point = center + offset * normal_vec

    return new_point

def shift_toward_point(points, p1, shift_cm, px_per_cm):
    """
    Shift p2 toward p1 by shift_cm (in cm), and return new point and distance.
    
    Parameters:
        p1 (tuple): First point (x1, y1)
        p2 (tuple): Second point (x2, y2)
        shift_cm (float): Distance to shift p2 toward p1 (in cm)
        px_per_cm (float): Conversion factor from cm to pixels

    Returns:
        shifted_p2 (tuple): New coordinates of p2 after shifting
        new_distance (float): Distance from p1 to shifted_p2 (in cm)
    """
    coordinates = []
    distances = []
    p1 = np.array(p1, dtype=np.float32)
    for p2 in points:
        # Convert to numpy arrays       
        p2 = np.array(p2, dtype=np.float32)

        # Compute the direction vector from p2 to p1
        direction = p1 - p2
        distance_px = np.linalg.norm(direction)

        if distance_px == 0:
            raise ValueError("Points are identical; cannot compute direction.")

        # Normalize direction and compute shift in pixels
        direction_normalized = direction / distance_px
        shift_px = shift_cm * px_per_cm

        # Shift p2 toward p1
        shifted_p2 = p2 + direction_normalized * shift_px

        # Compute new distance (in pixels), then convert to cm
        new_distance_px = np.linalg.norm(p1 - shifted_p2)
        new_distance_cm = new_distance_px / px_per_cm
        coordinates.append(shifted_p2)
        distances.append(new_distance_cm)

    return np.array(coordinates), np.array(distances)

def draw_line_with_angle(start_point, direction_vector, distance_cm, pix_per_cm, angle_deg):
    # Normalize the direction vector
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    # Convert angle to radians
    angle_rad = -np.deg2rad(angle_deg)

    # Rotation matrix (2D)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])

    # Rotate the direction vector
    rotated_vector = rotation_matrix @ direction_vector

    # Scale the rotated vector by the distance
    displacement = rotated_vector * distance_cm * pix_per_cm

    # Calculate end point
    end_point = (int(start_point[0] + displacement[0]), int(start_point[1] + displacement[1]))

    return end_point

def insert_between_large_diffs(arr):
    result = []

    for i in range(len(arr) - 1):
        result.append(arr[i])
        diff = arr[i+1] - arr[i]

        if abs(diff) > 10:
            # Insert two evenly spaced values between arr[i] and arr[i+1]
            step = diff / 3
            result.append(arr[i] + step)
            result.append(arr[i] + 2 * step)

    result.append(arr[-1])  # Add the last element
    return np.array(result)

def pow_two_pad_and_window(vec, show=False):
    window = signal.windows.tukey(len(vec), alpha=0.3)
    windowed_vec = vec * window
    padded_windowed_vec = np.pad(windowed_vec, (0, 2**int(np.ceil(np.log2(len(windowed_vec)))) - len(windowed_vec)))
    if show:
        dur = len(padded_windowed_vec) / fs
        t = np.linspace(0, dur, len(padded_windowed_vec))
        plt.figure()
        plt.plot(t, padded_windowed_vec)
        plt.show()
    return padded_windowed_vec/max(padded_windowed_vec)*0.8

os.chdir(os.path.dirname(os.path.abspath(__file__)))

file_name = 'GX010584'

camera_path = '/home/alberto/Videos/GOPRO/2025-07-21/' + file_name + '.mp4'
robot_path = './audio/' + 'MULTIWAV_134.34.226.238_2025-07-21__15-53-15' + '.wav'
# offsets_path = './offsets/' + file_name + '.yaml'

# with open(offsets_path, "r") as file:
#     try:
#         data = yaml.safe_load(file)  # Use safe_load to avoid potential security issues
#     except yaml.YAMLError as error:
#         print(f"Error loading YAML file: {error}")
# reading_points = data['reading_points']
# reading_points = np.array(reading_points)
# offsets = data['offsets']
# offsets = np.array(offsets)
offsets = np.array([1, 1, 0, 0])  # Insert 0 at the beginning

gopro_fps = 60
screen_width, screen_height = 4000, 3000
robot_id = 8
arena_w = 1.55
arena_l = 2
# Load video file
video_path = camera_path


# Load predefined dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

# Loop through the video
try:
    robot_audio, fs = sf.read(robot_path)
    robot_audio = robot_audio[:, 0]
    print( 'Robot audio duration: %.1f [s]' % (len(robot_audio)/fs))
    # Run ffmpeg to extract audio and pipe as WAV
    out, _ = (
        ffmpeg
        .input(camera_path)
        .output('pipe:', format='wav', acodec='pcm_s16le')
        .run(capture_stdout=True, capture_stderr=True)
    )

    # Load audio from bytes using soundfile
    camera_audio, sr = librosa.load(io.BytesIO(out), sr=fs, mono=True)
    print( 'Camera audio duration: %.1f [s]' % (len(camera_audio)/fs))

    xcorr = np.roll(signal.correlate(camera_audio, robot_audio, mode='same'), -len(robot_audio) // 2)
    index = np.argmax(np.abs(xcorr))
    start_frame = int(index / sr * gopro_fps)
    print('Start frame: %d' % start_frame)
    video_frames = np.astype((reading_points) / fs * gopro_fps + start_frame, np.int32)
    interp_video_frames = np.astype(insert_between_large_diffs(video_frames), np.int32)

    fs = 48000
    dur = 20e-3
    hi_freq = 2e3
    low_freq = 20e3
    output_threshold = -50 # [dB]

    METHOD = 'das' # 'das', 'capon'
    if METHOD == 'das':
        spatial_filter = das_filter

    t_tone = np.linspace(0, dur, int(fs*dur))
    chirp = signal.chirp(t_tone, hi_freq, t_tone[-1], low_freq)    
    sig = pow_two_pad_and_window(chirp)

    C_AIR = 343
    min_distance = 8e-2 # [m]
    discarded_samples = int(np.floor(((min_distance + 2.5e-2)*2)/C_AIR*fs))
    max_distance = 1 # [m]
    max_index = int(np.floor(((max_distance + 2.5e-2)*2)/C_AIR*fs))

    def update(frame):
        audio_data, _ = sf.read(robot_path, start=reading_points[frame], frames=offsets[frame])
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

except ffmpeg.Error as e:
    print('ffmpeg error:', e.stderr.decode(), file=sys.stderr)
    sys.exit(1)
cap = cv2.VideoCapture(video_path)

try:
    pixel_per_meters = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect markers
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
        
        # Draw detected markers
        if ids is not None:
            corners_array = np.squeeze(np.array(corners))
            try:
                ind12 = np.where(ids == 12)[0]
                if len(ind12) == 0:
                    raise ValueError('Marker 12 not found')
                ind13 = np.where(ids == 13)[0]
                if len(ind13) == 0:
                    raise ValueError('Marker 13 not found')
                ind14 = np.where(ids == 14)[0]
                if len(ind14) == 0:
                    raise ValueError('Marker 14 not found')
                # bottom left of 12, top left of 13, top right of 14                
                corners_12 = corners_array[ind12]
                corners_13 = corners_array[ind13]
                corners_14 = corners_array[ind14]
                pixel_per_meters = np.mean([np.linalg.norm(corners_12[:, 3] - corners_13[:, 0], axis=1)/arena_w, np.linalg.norm(corners_13[:, 0] - corners_14[:, 1], axis=1)/arena_l])
                print('Pixel per meters: %.2f' % pixel_per_meters)
            except ValueError:
                print('Marker 12, 13 or 14 not found')
        if pixel_per_meters > 0:
            break
except:
    print('Error reading video file:', e)
    sys.exit(1)
cap.release()

cap = cv2.VideoCapture(video_path)
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_fps = 30
output_dir = './blind_output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
out = cv2.VideoWriter(output_dir + file_name + '.mp4', fourcc, video_fps, (frame_width, frame_height))
try:
    frame_count = 0
    trajectory = np.zeros((0, 2), dtype=np.float32)
    counter = 0
    true_counter = 0

    obst_distances = []
    dist_error = []
    doas = []
    doa_error = []
    all_obstacles = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count == interp_video_frames[counter]:
            counter += 1
            if counter >= len(interp_video_frames):
                break
            if frame_count in video_frames: 
                distance, doa = update(true_counter)
                true_counter += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect markers
            detector = aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, _ = detector.detectMarkers(gray)
            
            # Draw detected markers
            if ids is not None:
                corners_array = np.squeeze(np.array(corners))
                # aruco.drawDetectedMarkers(frame, corners, ids)
                try:
                    index = np.where(ids == robot_id)[0] # Find the index of the robot marker
                    if len(index) == 0:
                        raise ValueError('Robot marker not found')
                    center = np.mean(corners_array[index], axis=1)[0]
                    trajectory = np.append(trajectory, np.array([[center[0], center[1]]]), axis=0)
                    
                    if distance != 0:
                        mask = np.ones(len(ids), dtype=bool)
                        mask[index] = False
                        ind12 = np.where(ids == 12)[0]
                        if len(ind12) > 0:
                            mask[ind12] = False
                        ind13 = np.where(ids == 13)[0]
                        if len(ind13) > 0:
                            mask[ind13] = False
                        ind14 = np.where(ids == 14)[0]
                        if len(ind14) > 0:
                            mask[ind14] = False                    
                        tl, tr, br, bl = np.squeeze(corners_array[index])
                        mic_positions = np.astype(get_offset_point(center, tl, tr, offset=-pixel_per_meters*0.07), np.int32)                       

                        obst_ids = ids[mask]
                        indexes = np.argsort(obst_ids, axis=0)
                        obst_corners = corners_array[mask]
                        obst_centers = np.mean(obst_corners, axis=1)
                        if (len(obst_centers) == 11 and not all_obstacles):
                            all_obstacles = True
                            print(obst_centers)
                            obst_pos = np.squeeze(obst_centers.copy())[indexes]
                            print(obst_pos)
                            obst_list = np.squeeze(obst_pos).tolist()
                            print('Position of all obstacles retrieved')
                        obstacles, distances = shift_toward_point(obst_centers, mic_positions, 3.2, pixel_per_meters/100)
                        if len(distances) > 0:
                            D41 = tl - bl
                            D14 = bl - tl
                            D41_normalized = D41 / np.linalg.norm(D41)
                            sorted_distances = np.sort(distances)
                            found_nearest = False
                            for sd in sorted_distances:
                                V_marker_space = np.squeeze(obstacles[np.where(distances == sd)]) - mic_positions
                                dot_product = np.dot(D41_normalized, V_marker_space)
                                cross_product = cross2d(V_marker_space, D14)
                                verse = -1 if cross_product > 0 else 1
                                angle = verse*np.arccos(dot_product / (np.linalg.norm(V_marker_space)))
                                # draw a line from the robot to the closest obstacle
                                if np.abs(angle) <= np.pi/2:               
                                    closest_obstacle = np.squeeze(obstacles[np.where(distances == sd)])
                                    found_nearest = True
                                    break
                            if found_nearest:
                                # print distance and angle                                
                                
                                # print('Time: %.1f [s]' % (frame_count/video_fps))
                                # print("Distance: %.1f [cm], Angle: %.1f [deg]" % (distance, doa))
                                # print("GT Distance: %.1f [cm], GT Angle: %.1f [deg]\n" % (sd, np.rad2deg(angle)))
                                obst_distances.append(sd)
                                dist_error.append(sd - distance)
                                doas.append(np.rad2deg(angle))
                                doa_error.append(np.rad2deg(angle) - doa)
                                cv2.arrowedLine(frame, mic_positions, closest_obstacle.astype(int), (255, 255, 0), 2)
                                # Draw the line
                                end_point = draw_line_with_angle(mic_positions, D41_normalized, distance, pixel_per_meters/100, doa)
                                cv2.arrowedLine(frame, mic_positions, end_point, (0, 255, 255), 2)
                
                    if len(trajectory) > 2:
                        # Draw trajectory
                        for i in range(len(trajectory) - 1):
                            cv2.line(frame, tuple(trajectory[i].astype(int)), tuple(trajectory[i + 1].astype(int)), (0, 255, 0), 2)
                    # Add a legend for the two lines departing from the robot
                    cv2.rectangle(frame, (50, 50), (620, 250), (0, 0, 0), -1)
                    cv2.putText(frame, 'Ground truth', (int(80), int(100)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    cv2.putText(frame, 'Estimated distance and direction', (int(80), int(200)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    out.write(frame)
                    # Display result
                    resized_frame = cv2.resize(frame, (screen_width, screen_height))
                    cv2.imshow('ArUco Tracker', resized_frame)
                except Exception:
                    traceback.print_exc()
                    pass
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(frame_count, e)
    traceback.print_exc()
# cv2.imwrite(file_name + '.jpg', resized_frame)
cap.release()
out.release()
cv2.destroyAllWindows()
data = {
    'obstacle_distances': np.asarray(obst_distances).tolist(),
    'distance_errors': np.asarray(dist_error).tolist(),
    'obstacle_angles': np.asarray(doas).tolist(),
    'angle_errors': np.asarray(doa_error).tolist(),
}
with open('./analysis/' + file_name + '.yaml', "w") as f:
    yaml.dump(data, f)

pos_dir = './obst_positions/'
if not os.path.exists(pos_dir):
    os.makedirs(pos_dir)
pos_data = {
    'obstacles_position': obst_list
}
with open(pos_dir + file_name + '.yaml', "w") as f:
    yaml.dump(pos_data, f)