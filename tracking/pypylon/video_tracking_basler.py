from pypylon import pylon
import cv2
import numpy as np
import os
import yaml
import subprocess
from datetime import datetime

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
camera_fps = camera.ResultingFrameRate.GetValue()
print(f"Hardware Camera FPS output: {camera_fps}")

output_dir = './tracking/videos/'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
file_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S:%f')[:-2] + '_basler_tracking'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
out = cv2.VideoWriter(output_dir + file_name + '.MP4', fourcc, video_fps, (crop_w, crop_h))
# Set the upper limit of the camera's frame rate to 30 fps
camera.AcquisitionFrameRateEnable.Value = True
camera.AcquisitionFrameRate.Value = video_fps

camera.TimestampLatch.Execute()
# Get the timestamp value
i = camera.TimestampLatchValue.Value
print(f"Timestamp Latch Value: {i}")

# Enable data chunks
# camera.ChunkModeActive.Value = True
# # Select a chunk
# camera.ChunkSelector.Value = "ExposureTime"
# # Enable the selected chunk
# camera.ChunkEnable.Value = True
# # Now, you must implement chunk retrieval in your application.
# # For C++, C, and .NET sample implementations, see the "Grab_ChunkImage" and
# # "Chunks" code samples in the pylon API Documentation

def main():
    try:
        while camera.IsGrabbing():
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            # Check if the image was successfully grabbed from the camera
            if grab_result.GrabSucceeded():
                # Convert the grabbed image to OpenCV format
                image = converter.Convert(grab_result)
                frame = image.GetArray()

                # Get the image dimensions
                h, w = frame.shape[:2]

                # Compute undistortion map only once for performance
                # This creates a new camera matrix and undistortion maps based on calibration parameters
                if 'mapx' not in locals():
                    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
                    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_16SC2)
                    
                # Apply undistortion to the frame using the computed maps
                undistorted = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR)

                # Detect ArUco markers
                # corners, ids, _ = cv2.aruco.detectMarkers(undistorted, aruco_dict, parameters=aruco_params)
                corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

                # Draw detected markers
                if ids is not None:
                    # cv2.aruco.drawDetectedMarkers(undistorted, corners, ids)
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')[:-3]  # Format timestamp to milliseconds
                timestamp = str(datetime.timestamp(datetime.now())) # POSIX timestamp

                # Put timestamp text on the frame
                cv2.putText(undistorted, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)

                out.write(frame)
                camera.TimestampLatch.Execute()
                i = camera.TimestampLatchValue.Value
                print(f"Timestamp Latch Value: {i}")

                # Show the frame
                # cv2.namedWindow("Basler Camera stream", cv2.WINDOW_NORMAL)
                # cv2.imshow("Basler Camera stream", undistorted)
                cv2.namedWindow("Basler Camera original", cv2.WINDOW_NORMAL)
                cv2.imshow("Basler Camera original", frame)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    break

            grab_result.Release()
    finally:
        camera.StopGrabbing()
        camera.Close()
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
