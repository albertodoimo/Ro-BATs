# Pypylon setup for Hardware triggering the camera to record and store images
# using an external signal from soundcard on left channel of stereo signal

# Documentation at:
# https://github.com/basler/pypylon/issues/842
#%%
import pypylon.pylon as pylon
import time
import datetime
import platform
import os 

###################################################################

# Setup the camera (model = ace2 R a2A4508-20umBAS; Max frame rate = 19.4 fps)

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

# Crop the image to the desired size
camera.Width.SetValue(crop_w)
camera.Height.SetValue(crop_h)

# Center crop into the original image
camera.BslCenterX.Execute()
camera.BslCenterY.Execute()

# Set the upper limit of the camera's frame rate
camera.AcquisitionFrameRateEnable.Value = True
camera.AcquisitionFrameRate.Value = 1000

# Hardware Trigger Configuration
camera.TriggerSelector.Value = "FrameStart"
camera.TriggerSource.Value = "Line1" #opto-coupled line
camera.TriggerMode.Value = "On"
camera.TriggerActivation.Value = "RisingEdge"

#######################################################################

# setup for saving images
img = pylon.PylonImage()
camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
image_quality = 100
platform_name = 'Linux'  

# Create folder for saving recordings
timenow = datetime.datetime.now()
time = timenow.strftime('%Y-%m-%d')
time1 = timenow.strftime('%Y-%m-%d_%H-%M-%S')
#%%
file_dir = os.path.dirname(os.path.abspath(__file__)) # project directory
save_path = 'data/'
date_folder_name = str(time)
hour_folder_name = str(time1)
folder_path = os.path.join(file_dir, save_path, date_folder_name, hour_folder_name)
os.makedirs(folder_path, exist_ok=True)

print('\nCamera is ready and waiting for a trigger signal on Line1...\n')

def main():

    try:
        while camera.IsGrabbing():
            print('Recording ... (???)')
            with camera.RetrieveResult(20000, pylon.TimeoutHandling_ThrowException) as result:
                
                # Calling AttachGrabResultBuffer creates another reference to the
                # grab result buffer. This prevents the buffer's reuse for grabbing.
                img.AttachGrabResultBuffer(result)

                if platform.system() == 'Linux':
                    # The JPEG format that is used here supports adjusting the image
                    # quality (100 -> best quality, 0 -> poor quality).
                    ipo = pylon.ImagePersistenceOptions()
                    ipo.SetQuality(image_quality)

                    filename = f"{folder_path}/{datetime.datetime.now(datetime.timezone.utc).isoformat()}_{image_quality}.jpeg"
                    img.Save(pylon.ImageFileFormat_Jpeg, filename, ipo)
                else:
                    filename = f"{folder_path}/{datetime.datetime.now(datetime.timezone.utc).isoformat()}_{image_quality}.png"
                    img.Save(pylon.ImageFileFormat_Png, filename, ipo)

    except KeyboardInterrupt:
        print("Interrupted by user (Ctrl+C).")
    finally:
        img.Release()
        if camera.IsGrabbing():
            camera.StopGrabbing()
            print("Camera stopped.")
        camera.Close()

if __name__ == "__main__":
    main()
