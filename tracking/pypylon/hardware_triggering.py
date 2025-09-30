# Various test on Pypylon setup for Hardware triggering the camera to record and store images
# using an external signal

# Documentation at:
# https://github.com/basler/pypylon-samples/blob/main/notebooks/basic-examples/USB_hardware_trigger_and_chunks.ipynb

#%%
import pypylon.pylon as pylon
import pypylon.genicam as geni
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import datetime
import pandas as pd
import platform
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

# Arena dimensions in meters from the marks on the carpet
arena_w = 1.47 # m
arena_l = 1.91 # m

# enable the chunk that
# samples all IO lines on every FrameStart
camera.ChunkModeActive = True
camera.ChunkSelector = "LineStatusAll"
camera.ChunkEnable = True
print(camera.ChunkSelector.Symbolics)

#%%

camera.Width.SetValue(crop_w)
camera.Height.SetValue(crop_h)

# Center crop into the original image
camera.BslCenterX.Execute()
camera.BslCenterY.Execute()

# Set the upper limit of the camera's frame rate
camera.AcquisitionFrameRateEnable.Value = True
camera.AcquisitionFrameRate.Value = 1000

print('Resulting Frame Rate', camera.ResultingFrameRate.Value)

# Hardware Trigger 

# get clean powerup state
camera.UserSetSelector = "Default"
camera.UserSetLoad.Execute()
# setup the io section

camera.TriggerSelector = "FrameStart"
camera.TriggerSource = "Line1"
camera.TriggerMode = "On"
camera.TriggerActivation.Value = "RisingEdge"

img = pylon.PylonImage()
camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
try:
    while camera.IsGrabbing():
        with camera.RetrieveResult(10000, pylon.TimeoutHandling_ThrowException) as result:

            # Calling AttachGrabResultBuffer creates another reference to the
            # grab result buffer. This prevents the buffer's reuse for grabbing.
            img.AttachGrabResultBuffer(result)

            if platform.system() == 'Linux':
                # The JPEG format that is used here supports adjusting the image
                # quality (100 -> best quality, 0 -> poor quality).
                ipo = pylon.ImagePersistenceOptions()

                quality = 50
                ipo.SetQuality(quality)

                filename = f"data/frames/saved_pypylon_img_{datetime.datetime.now(datetime.timezone.utc)}_{quality}.jpeg"
                img.Save(pylon.ImageFileFormat_Jpeg, filename, ipo)
            else:
                filename = "saved_pypylon_img_%d.png" % (time.time() * 1000)
                img.Save(pylon.ImageFileFormat_Png, filename)

            # In order to make it possible to reuse the grab result for grabbing
            # again, we have to release the image (effectively emptying the
            # image object).

            # Stop recording and save data when ESC is pressed or window is closed

except KeyboardInterrupt:
    print("Interrupted by user (Ctrl+C).")
    
    img.Release()

    camera.StopGrabbing()
    camera.Close()

#%%

camera.Width.SetValue(crop_w)
camera.Height.SetValue(crop_h)

# Center crop into the original image
camera.BslCenterX.Execute()
camera.BslCenterY.Execute()

# Set the upper limit of the camera's frame rate
camera.AcquisitionFrameRateEnable.Value = True
camera.AcquisitionFrameRate.Value = 1000

print('Resulting Frame Rate', camera.ResultingFrameRate.Value)

###########################################
# Check if the camera is doing what it's supposed to do
camera.StartGrabbingMax(100)

io_res = []
while camera.IsGrabbing():
    with camera.RetrieveResult(1000) as res:
        time_stamp = res.TimeStamp
        print(f"Timestamp: {time_stamp}")
        io_res.append((time_stamp, res.ChunkLineStatusAll.Value))

camera.StopGrabbing()

print(io_res[:10])

# simple logic analyzercamera.Close()

# convert to numpy array
io_array = np.array(io_res)
# extract first column timestamps
x_vals = io_array[:,0]
#  start with first timestamp as '0'
x_vals -= x_vals[0]

# extract second column io values
y_vals = io_array[:,1]
# Plot the status of each IO line as a logic analyzer
plt.figure(figsize=(12, 12))
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
for bit in range(3):
    logic_level = ((y_vals & (1 << bit)) != 0)
    axs[bit].step(x_vals / 1e9, logic_level, where='post')
    axs[bit].set_ylabel(f"Line {bit+1}")
    axs[bit].set_yticks([0, 1])
    axs[bit].set_ylim(-0.2, 1.2)
    axs[bit].grid(True, axis='x')
axs[-1].set_xlabel("Time [s]")
plt.suptitle("Logic Analyzer - IO Line Status Over Time")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

camera.Close()

#%%
########################################################3
# Hardware Trigger 

# get clean powerup state
camera.UserSetSelector = "Default"
camera.UserSetLoad.Execute()
# setup the io section

camera.TriggerSelector = "FrameStart"
camera.TriggerSource = "Line1"
camera.TriggerMode = "On"
camera.TriggerActivation.Value = "RisingEdge"


images = {}
grabs = []
img = pylon.PylonImage()

camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
countOfImagesToGrab = 100 * 3

for _ in range(countOfImagesToGrab):
    if not camera.IsGrabbing():
        print("not grabbing!")
        break

    with camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException) as grabResult:
        cam_idx = grabResult.GetCameraContext()
        if grabResult.GrabSucceeded():
            images.setdefault(cam_idx, []).append(grabResult.GetArray())
            grabs.append(cam_idx)
        else:
            print(f"Error: Grab failed for camera {cam_idx}. Error code: {grabResult.GetErrorCode()}, Description: {grabResult.GetErrorDescription()}")

        img.AttachGrabResultBuffer(grabResult)

        if platform.system() == 'Linux':
            # The JPEG format that is used here supports adjusting the image
            # quality (100 -> best quality, 0 -> poor quality).
            ipo = pylon.ImagePersistenceOptions()
            quality = 50
            ipo.SetQuality(quality)

            filename = "saved_pypylon_img_%d.jpeg" % quality
            img.Save(pylon.ImageFileFormat_Jpeg, filename, ipo)
        else:
            filename = "saved_pypylon_img_%d.png" % (time.time() * 1000)
            img.Save(pylon.ImageFileFormat_Png, filename)

camera.StopGrabbing()
grabs = np.array(grabs)
print("done!")
camera.Close()

#%%
#######################################

# TRIGGERING SETTINGS

# Trigger Activation Modes

# You can set the activation mode using the TriggerActivation parameter.
# This setting only applies to hardware triggering. It defines which signal 
# transition activates the selected trigger. For example, you can specify 
# that a trigger becomes active when the trigger signal falls.
# Depending on your camera model, the following trigger activation modes are available:

# RisingEdge: The trigger becomes active when the trigger signal rises, i.e., 
    #  when the signal status changes from low to high.
# FallingEdge: The trigger becomes active when the trigger signal falls, i.e., 
    #  when the signal status changes from high to low.
# AnyEdge: The trigger becomes active when the trigger signal falls or rises.
# LevelHigh: The trigger is active as long as the trigger signal is high.
# LevelLow: The trigger is active as long as the trigger signal is low.

# Enable triggered image acquisition for the Frame Start trigger
camera.TriggerMode.Value = "On"
# Set the trigger source to Line 1
camera.TriggerSource.Value = "Line1"
# Set the trigger activation mode to level high
camera.TriggerActivation.Value = "RisingEdge"

# Triggering frames

ExposureMode = TriggerControlled

TriggerSelector = ExposureStart
TriggerMode = On
TriggerActivation = RisingEdge
TriggerSource = Line1

TriggerSelector = ExposureEnd
TriggerMode = On
TriggerActivation = FallingEdge
TriggerSource = Line2

#Triggering a Series of Images
TriggerSelector = FrameBurstStart
TriggerMode = On
AcquisitionBurstFrameCount = 3 # 3 images triggered 
TriggerActivation = RisingEdge
TriggerSource = Line1

# ACQUISITION SETTINGS

# Configure continuous image acquisition on the cameras
camera.AcquisitionMode.Value = "Continuous"
# Switch on image acquisition
camera.AcquisitionStart.Execute()
# The camera waits for trigger signals
# (...)
# Switch off image acquisition
camera.AcquisitionStop.Execute()
# Switch image acquisition back on
camera.AcquisitionStart.Execute()
# The camera waits for trigger signals
# (...)
# Abort image acquisition
camera.AcquisitionAbort.Execute()

##########################################

# def main():
#     # Initialize screen recording variables
#     # recorder = None
#     # recording_started = False
#     # screen_recording_enabled = False
    
#     try:
#         while camera.IsGrabbing():
#             grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

#             if grab_result.GrabSucceeded():
#                 timestamp = str(datetime.datetime.timestamp(datetime.datetime.now(datetime.timezone.utc))-readout_time)

#                 image = converter.Convert(grab_result)
#                 frame = image.GetArray()
#                 original_frame = frame.copy()
#                 h, w = frame.shape[:2]

#                 corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
#                 if ids is not None:
#                     noise_on = 200 in ids
#                     print(f"Noise on: {noise_on}")
#                 start_time = time.time()
#                 marker_pairs = [(4, 5), (6, 7), (10, 11)]
#                 robot_names = {(4, 5): "241", (6, 7): "240", (10, 11): "238"}
           
#             grab_result.Release()
#     finally:
#         camera.StopGrabbing()
#         camera.Close()
#         cv2.destroyAllWindows()

#         # subprocess.run(
#         #     ["bash", "/home/alberto/Documents/ActiveSensingCollectives_lab/Ro-BATs/tracking/stop_all_robots.sh"],
#         #     check=True
#         # )       

# if __name__ == "__main__":
#     main()

# # %%
