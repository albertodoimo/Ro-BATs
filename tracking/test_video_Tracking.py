import cv2
import numpy as np
from datetime import datetime
from mss import mss
import math
import zmq
import csv
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import soundfile as sf

from scipy.fftpack import fft, ifft
from scipy import signal


input_video_path = '/Users/alberto/Documents/UNIVERSITA/MAGISTRALE/tesi/robat video-foto/pdm 7 mic array/inverted_loop_pdm array_7mic_fast.mp4'  # replace with your input video path
#input_video_path = '/Users/alberto/Desktop/test_swarmlab.mp4'
#input_video_path = '/Users/alberto/Desktop/2024-08-20__16-39-58_CC 2.MP4'
output_video_path = '/Users/alberto/Desktop/test.MP4'  # replace with your desired output video path
overlay_img_path = '/Users/alberto/Documents/UNIVERSITA/MAGISTRALE/tesi/github/Ro-BATs/tracking/ROBAT LOGO.png'  # replace with your overlay image path

video = cv2.VideoCapture(input_video_path)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

print(width)
print(height)
print(fps)

video.release()

class Aruco_tracker:
    def __init__(self, cam_id=-1, monitor_id=0, debug=True, debug_stream=True, frame_width=width, frame_height=height, crop_img=False, num_tags=15, decision_margin=20, record_stream=False, publish_pos=False, print_pos=False, detect_arena=False):
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self._parameters = cv2.aruco.DetectorParameters()
        self._parameters.adaptiveThreshConstant = 10
        self._debug = debug
        self._debug_stream = debug_stream
        self._record_stream = record_stream
        self._publish_pos = publish_pos
        self._cam_id = cam_id
        self._num_tags = num_tags
        self._decision_margin = decision_margin
        self._print_pos = print_pos
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._crop_img = crop_img

        if cam_id == -1:
            self._sct = mss()
            self._sct.compression_level = 2
            self._monitor = self._sct.monitors[monitor_id]
        else:
            self._cam = cv2.VideoCapture(cam_id)
            self._cam.set(cv2.CAP_PROP_FOURCC, 0x47504A4D)
            self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
            self._cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cam.set(cv2.CAP_PROP_AUTOFOCUS, False)
            self._cam.set(cv2.CAP_PROP_FOCUS, 0)

        if self._debug_stream:
            cv2.namedWindow("tracking", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("tracking", 700, 700)

        if record_stream:
            self._sub_socket = zmq.Context().socket(zmq.SUB)
            self._sub_socket.connect('tcp://localhost:5556')
            self._sub_socket.setsockopt_string(zmq.SUBSCRIBE, "43")
            self._writer = cv2.VideoWriter('init.mp4', cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), fps, (frame_width, frame_height))
            self._record = False

        if self._publish_pos:
            self._pub_socket = zmq.Context().socket(zmq.PUB)
            self._pub_socket.bind("tcp://*:5556")
            if detect_arena:
                self._arena_corners = np.float32([[0, 0], [0, 0], [0, 0], [0, 0]])
                self.detect_arena()

    def detect_arena(self):
        cap_img = np.array(self._sct.grab(self._monitor))
        gray = cv2.cvtColor(cap_img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, self._aruco_dict, parameters=self._parameters)

        while 100 not in ids or 101 not in ids or 102 not in ids or 103 not in ids:
            cap_img = np.array(self._sct.grab(self._monitor))
            gray = cv2.cvtColor(cap_img, cv2.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, self._aruco_dict, parameters=self._parameters)
            sys.stdout.write("\033[K")
            print("\r", "Arena not detected", end="")
            pass

        for i in range(len(ids)):
            (topLeft, topRight, bottomRight, bottomLeft) = corners[i][0]
            if ids[i] == 100:
                self._arena_corners[0] = topLeft
            elif ids[i] == 101:
                self._arena_corners[1][:] = bottomLeft
            elif ids[i] == 102:
                self._arena_corners[2][:] = topRight
            elif ids[i] == 103:
                self._arena_corners[3][:] = bottomRight

        print(self._arena_corners)
        self._pub_socket.send_pyobj([42, self._arena_corners])

    def get_positions(self):
        if self._cam_id == -1:
            cap_img = np.array(self._sct.grab(self._monitor))
        else:
            ret, cap_img = self._cam.read()
            record_img = cap_img
            if not ret:
                console.error("Failed to grab frame", severe=True)
                return np.zeros(self._num_tags * 2)

            if self._crop_img:
                cap_img = cap_img[0:1080, 420:1500]

        if self._crop_img:
            cap_img = cv2.resize(cap_img, (self._frame_width, self._frame_height), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(cap_img, cv2.COLOR_BGR2GRAY)
        displ_img = cap_img
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, self._aruco_dict, parameters=self._parameters)
        print('id = ',ids)
        marker_size = np.linalg.norm(corners[0][0] - corners[0][1])  # Diagonal length of marker
        overlay_size = (int(marker_size * 2), int(marker_size * 2))  # Larger than marker size

        x_positions = [0] * self._num_tags
        y_positions = [0] * self._num_tags
        omega_angles = [0] * self._num_tags
        positions = np.zeros(self._num_tags * 2)
        printpositions = []

        if len(corners) > 0:
            ids = ids.flatten()

            for (markerCorner, markerID) in zip(corners, ids):
                if markerID < self._num_tags:
                    corners = markerCorner.reshape((4, 2))
                    (topRight, bottomRight, bottomLeft, topLeft) = corners.astype(np.int)

                    cX = int((bottomLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((bottomLeft[1] + bottomRight[1]) / 2.0)

                    fX = int((topRight[0] + topLeft[0]) / 2.0)
                    fY = int((topRight[1] + topLeft[1]) / 2.0)

                    v = (cX - fX, cY - fY)
                    v_len = math.sqrt(v[0]**2 + v[1]**2)
                    omega = math.atan2(v[1], -v[0])

                    crX = int(cX + math.sin(omega + math.pi / 2) * v_len / 4)
                    crY = int(cY + math.cos(omega + math.pi / 2) * v_len / 4)

                    positions[markerID] = crX
                    positions[markerID + self._num_tags] = crY
                    x_positions[markerID] = crX
                    y_positions[markerID] = crY
                    printpositions.append([markerID, x_positions[markerID], y_positions[markerID], omega, v])

                    if self._publish_pos:
                        self._pub_socket.send_string("%d %d %d %d %d %d" % (markerID, x_positions[markerID], y_positions[markerID], omega, v[0], v[1]))

                    if self._debug_stream:
                        cv2.line(cap_img, topLeft, topRight, (0, 255, 0), 2)
                        cv2.line(cap_img, topRight, bottomRight, (0, 255, 0), 2)
                        cv2.line(cap_img, bottomRight, bottomLeft, (0, 255, 0), 2)
                        cv2.line(cap_img, bottomLeft, topLeft, (0, 255, 0), 2)
                        cv2.circle(cap_img, (crX, crY), 4, (0, 0, 255), -1)
                        cv2.circle(cap_img, (fX, fY), 4, (0, 255, 255), -1)
                        cv2.putText(cap_img, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if self._record_stream:
            if self._record:
                self._writer.write(record_img)
            try:
                string = self._sub_socket.recv(flags=zmq.NOBLOCK).decode('utf-8')
                topic, data = string.split()
                if data == "record":
                    self._record = True
                    self._writer.release()
                    date_time = datetime.now()
                    timestamp = date_time.strftime("%d%m%Y_%H%M%S") + '.mp4'
                    print(timestamp)
                    self._writer = cv2.VideoWriter(timestamp, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (self._frame_width, self._frame_height))
                    print("record")
                elif data == 'save':
                    self._record = False
                    print("save")
            except zmq.Again as error:
                pass

        if self._debug_stream:
            cv2.imshow("tracking", displ_img)

        if cv2.waitKey(1) & 0xFF == 27:
            return

        if self._print_pos:
            sys.stdout.write("\033[K")
            print("\r", "\t", printpositions, end="")

        return positions, overlay_size

#    def close(self):
#        print("close")
#        self._writer.release()

#%%

data, samplerate = sf.read('/Users/alberto/Desktop/2024-08-16__17-11-42_MULTIWAV/1.wav')
print('ch=', np.shape(data)[1])

method = 'CC'
doa_name = 'SRP'

c = 343   # speed of sound
fs = samplerate
block_size = samplerate//fps
print('bs=', block_size)
channels = np.shape(data)[1]
mic_spacing = 0.015 #m
ref = channels//2 #central mic in odd array as ref
#ref= 0 #left most mic as reference
nfft = 32  # FFT size

auto_hipas_freq = int(343/(2*(mic_spacing*(channels-1))))
print('HP frequency:', auto_hipas_freq)
auto_lowpas_freq = int(343/(2*mic_spacing))
print('LP frequency:', auto_lowpas_freq)

highpass_freq, lowpass_freq = [auto_hipas_freq ,auto_lowpas_freq]
freq_range = [highpass_freq, lowpass_freq]

nyq_freq = fs/2.0
b, a = signal.butter(4, [highpass_freq/nyq_freq,lowpass_freq/nyq_freq],btype='bandpass') # to be 'allowed' in Hz.


echo = pra.linear_2D_array(center=[(channels-1)*mic_spacing//2,0], M=channels, phi=0, d=mic_spacing)

for block in sf.blocks('/Users/alberto/Desktop/2024-08-16__17-11-42_MULTIWAV/1.wav', blocksize=block_size, overlap=0):
    buffer = block
print('buff=',np.shape(buffer))
#print(buffer)


theta_values = []

def bandpass_sound(rec_buffer,a,b):
    """
    """
    rec_buffer_bp = np.apply_along_axis(lambda X : signal.lfilter(b, a, X),0, rec_buffer)
    return(rec_buffer_bp)

def calc_delay(two_ch,fs):
    
    '''
    Parameters
    ----------
    two_ch : (Nsamples, 2) np.array
        Input audio buffer
    ba_filt : (2,) tuple
        The coefficients of the low/high/band-pass filter
    fs : int, optional
        Frequency of sampling in Hz. Defaults to 44.1 kHz

    Returns
    -------
    delay : float
        The time-delay in seconds between the arriving audio across the 
        channels. 
    '''
    for each_column in range(2):
        two_ch[:,each_column] = two_ch[:,each_column]

    cc = np.correlate(two_ch[:,0],two_ch[:,1],'same')
    midpoint = cc.size/2.0
    delay = np.argmax(cc) - midpoint
    # convert delay to seconds
    delay *= 1/float(fs)
    return delay

def gcc_phat(sig,refsig, fs):
    # Compute the cross-correlation between the two signals
    #sig = sig[:,1]
    
    n = sig.shape[0] + refsig.shape[0]
    SIG = fft(sig, n=n)
    REFSIG = fft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    cc = np.fft.ifft(R / np.abs(R))
    max_shift = int(np.floor(n / 2))
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    #plt.plot(cc)
    #plt.show()
    #plt.title('gcc-phat')
    shift = np.argmax(np.abs(cc)) - max_shift
    return -shift / float(fs)

def calc_multich_delays(multich_audio,ref_sig,fs):
    '''s
    Calculates peak delay based with reference of 
    channel 1. 
    '''
    nchannels = multich_audio.shape[1]
    delay_set = []
    delay_set_gcc = []
    i=0
    while i < nchannels:
        if i != ref:
            #print(i)
            
            #delay_set.append(calc_delay(multich_audio[:,[ref, i]],fs)) #cc without phat norm
            delay_set.append(gcc_phat(multich_audio[:,i],ref_sig,fs)) #gcc phat correlation
            i+=1
        else:
            #print('else',i)
            i+=1
            pass

    #print('delay=',delay_set)
    #print('delay gcc=',delay_set_gcc)
    return np.array(delay_set)

def avar_angle(delay_set,nchannels,mic_spacing):
    '''
    calculates the mean angle of arrival to the array
    with channel 1 as reference
    '''
    theta = []
    #print(delay_set)
    if ref!=0: #centered reference that works with odd mics
        for each in range(0, nchannels//2):
            #print('\n1',each)
            #print('1',nchannels//2-each)
            theta.append(-np.arcsin((delay_set[each]*343)/((nchannels//2-each)*mic_spacing))) # rad
            i=nchannels//2-each
            #print('i=',i)
        for each in range(nchannels//2, nchannels-1):
            #print('\n2',each)
            #print('2',i)
            theta.append(np.arcsin((delay_set[each]*343)/((i)*mic_spacing))) # rad
            i+=1
    else:   
        for each in range(0, nchannels-1):
            theta.append(np.arcsin((delay_set[each]*343)/((each+1)*mic_spacing))) # rad

    avar_theta = np.mean(theta)
    return avar_theta

def update():

    in_sig = buffer

    ref_channels = in_sig
    #print(np.shape(in_sig))
    #print('ref_channels=', np.shape(ref_channels))
    ref_channels_bp = bandpass_sound(ref_channels,a,b)
    #print('ref_channels_bp=', np.shape(ref_channels_bp))
    ref_sig = in_sig[:,ref]
    delay_crossch= calc_multich_delays(in_sig,ref_sig, fs)

    # calculate avarage angle
    avar_theta = avar_angle(delay_crossch,channels,mic_spacing)

    print('avarage theta',avar_theta)

    print('avarage theta deg = ', np.rad2deg(avar_theta))
    return np.rad2deg(avar_theta)


def update_polar():
    # Your streaming data source logic goes here

    in_sig = buffer

    X = pra.transform.stft.analysis(in_sig, nfft, nfft // 2)
    X = X.transpose([2, 1, 0])

    doa = pra.doa.algorithms[doa_name](echo, fs, nfft, c=c, num_src=2, max_four=4)
    doa.locate_sources(X, freq_range=freq_range)
    #print('azimuth_recon=',doa.azimuth_recon) # rad value of detected angles
    theta_pra_deg = (doa.azimuth_recon * 180 / np.pi) 
    #print('theta=',theta_pra_deg) #degrees value of detected angles

    spatial_resp = doa.grid.values # 360 values for plot
    #print('spat_resp',spatial_resp) 

    # normalize   
    min_val = spatial_resp.min()
    max_val = spatial_resp.max()
    spatial_resp = (spatial_resp - min_val) / (max_val - min_val)


#%%
def draw_trajectories_on_video(input_video_path, output_video_path, aruco_tracker, overlay_img_path):
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 24, (frame_width, frame_height))

    trajectories = {}
    colors = {}

    # Camera calibration parameters (you might need to calibrate your camera)
    camera_matrix = np.array([[1406.08415449821, 0, 0],
                              [2.20679787308599, 1417.99930662800, 0],
                              [1014.13643417416, 566.347754321696, -1]]).reshape(3, 3)

    dist_coeffs = np.array([-0.2380769334, 0.0931325835, 0, 0, 0])
    
    overlay_img = cv2.imread(overlay_img_path)
    overlay_img = cv2.resize(overlay_img, (100, 100))  # Adjust size as needed of the overlay image
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
              
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_tracker._aruco_dict, parameters=aruco_tracker._parameters)

        if ids is not None:
            ids = ids.flatten()
            for corner, markerID in zip(corners, ids):
                if markerID not in trajectories:
                    trajectories[markerID] = []
                    colors[markerID] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                    colors[70] = [0,255,0]
                
                # Estimate pose of each marker
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corner, 0.08, camera_matrix, dist_coeffs)

                # Draw 3D axis on the marker
                #cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs, tvecs, 0.1)

                center = np.mean(corner[0], axis=0)
                trajectories[markerID].append(center)

                for i in range(1, len(trajectories[markerID])):
                    #print('i1',i)
                    cv2.line(frame, tuple(trajectories[markerID][i-1].astype(int)), tuple(trajectories[markerID][i].astype(int)), colors[markerID], 3)

                    # Overlay image on marker
                    # Compute the homography to warp the overlay image
                    pts_dst = corner[0].astype(int)
                    #print(pts_dst)
                    pts_src = np.array([[0, 0], [overlay_img.shape[1], 0], [overlay_img.shape[1], overlay_img.shape[0]], [0, overlay_img.shape[0]]])
                    h, _ = cv2.findHomography(pts_src, pts_dst)

                    # Warp the overlay image onto the marker
                    overlay_warped = cv2.warpPerspective(overlay_img, h, (frame.shape[1], frame.shape[0]))

                    # Create a mask of the overlay image
                    overlay_mask = cv2.cvtColor(overlay_warped, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(overlay_mask, 1, 255, cv2.THRESH_BINARY)

                    # Invert the mask for the overlay
                    mask_inv = cv2.bitwise_not(mask)

                    # Black-out the area of the overlay in the frame
                    img_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)

                    # Take only region of overlay from overlay image
                    img_fg = cv2.bitwise_and(overlay_warped, overlay_warped, mask=mask)

                    # Put overlay on top of the current frame
                    frame = cv2.add(img_bg, img_fg)

                # Compute rotation angle around the z-axis (in-plane rotation)
                rotation_matrix, _ = cv2.Rodrigues(rvecs[0])
                angle = -np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))

                # Read and process spatial response data
                spatial_resp = []
                #with open('/Users/alberto/Documents/UNIVERSITA/MAGISTRALE/tesi/github/Ro-BATs/tracking/spat_resp.csv', "r", newline='') as file:
                #with open('/Users/alberto/Desktop/2024-08-20__16-39-58_rec_data/2024-08-20__16-39-58_spat_resp_CC.csv', "r", newline='') as file:
                #    reader = csv.reader(file)
                #    for row in reader:
                #        spatial_resp.append(list(map(float, row)))
                #        #print(np.shape(spatial_resp))


                # Create a polar plot

                if method == 'CC':
                    spatial_resp = update()
                    spatial_resp = np.array(spatial_resp)
                    print('spat resp', np.shape(spatial_resp))

                
#                    x = float(spatial_resp[i]) 
#                    #print('x',x)
#                    if math.isnan(x):
#                        spatial_resp[i][0] = 0
#                    print('2',spatial_resp[i][0]) 
#
#                    values = np.zeros(360)
#
#                    for ii in range(len(values)):
#                        if round(90-np.rad2deg(spatial_resp[i][0])) == ii:
#                            print('not 90',np.rad2deg(spatial_resp[i][0]))
#                            print('round',round(90-np.rad2deg(spatial_resp[i][0])))
#                            values[ii] = 1
#                        else:
#                            values[ii] = 0
#                    print('val',values)
#
                    x = float(spatial_resp) 
                    #print('x',x)
                    if math.isnan(x):
                        spatial_resp = 0
                    print('2',spatial_resp) 

                    values = np.zeros(360)

                    for ii in range(len(values)):
                        if round(90-np.rad2deg(spatial_resp)) == ii:
                            print('not 90',np.rad2deg(spatial_resp))
                            print('round',round(90-np.rad2deg(spatial_resp)))
                            values[ii] = 1
                        else:
                            values[ii] = 0
                    print('val',values)

                    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},figsize=(6, 6))
                    theta = np.linspace(0, 2*np.pi, 360)
                    line, = ax.plot(theta, values)
                    line.set_ydata(values)


                    ax.set_theta_direction(1)
                    ax.set_title("Polar Plot")
                    #ax.set_theta_offset(np.pi/2)
                    ax.set_rticks([])
                    ax.set_thetagrids(range(0, 360, 30))
                    ax.grid(True)
                    plt.savefig('polar_plot.png')
                    plt.close(fig)
                    #plt.show()

                elif method == 'PRA':
                    #print(np.shape(spatial_resp))
                    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},figsize=(6, 6))
                    theta = np.linspace(0, 2*np.pi, 360)
                    #print('i3',i)
                    ax.plot(theta, spatial_resp[i])
                    ax.set_theta_direction(1)
                    ax.set_title("Polar Plot")
                    #ax.set_theta_offset(np.pi)
                    ax.set_rticks([])
                    ax.set_thetagrids(range(0, 360, 30))
                    ax.grid(True)
                    plt.savefig('polar_plot.png')
                    plt.close(fig)
                
                # Load and resize the polar plot image
                overlay_img_polar = cv2.imread('/Users/alberto/Documents/UNIVERSITA/MAGISTRALE/tesi/github/Ro-BATs/tracking/polar_plot.png')
                overlay_img_polar = cv2.resize(overlay_img_polar, (400, 400))

                # Calculate rotation matrix for in-plane rotation
                M = cv2.getRotationMatrix2D((overlay_img_polar.shape[1] // 2, overlay_img_polar.shape[0] // 2), angle, 1.0)

                # Rotate overlay image
                rotated_overlay = cv2.warpAffine(overlay_img_polar, M, (overlay_img_polar.shape[1], overlay_img_polar.shape[0]), 
                                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                # Position the overlay on the frame
                #x, y = 50, 50  # You can change this to position the overlay in a different location
                x,y  = (width-overlay_img_polar.shape[0], height-overlay_img_polar.shape[1]) # position if the image on the video

                # Create a mask of the overlay image
                overlay_gray = cv2.cvtColor(rotated_overlay, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(overlay_gray, 1, 255, cv2.THRESH_BINARY)

                # Invert the mask
                mask_inv = cv2.bitwise_not(mask)

                # Black-out the area of the overlay in the frame
                img_bg = cv2.bitwise_and(frame[y:y+rotated_overlay.shape[0], x:x+rotated_overlay.shape[1]], 
                                         frame[y:y+rotated_overlay.shape[0], x:x+rotated_overlay.shape[1]], 
                                         mask=mask_inv)

                # Take only the region of the overlay from the overlay image
                img_fg = cv2.bitwise_and(rotated_overlay, rotated_overlay, mask=mask)

                # Put the overlay on top of the current frame
                frame[y:y+rotated_overlay.shape[0], x:x+rotated_overlay.shape[1]] = cv2.add(img_bg, img_fg)

        out.write(frame)
        cv2.imshow('Trajectories', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
# Usage
# aruco_tracker should be an initialized ArucoTracker object
# draw_trajectories_on_video('input_video.mp4', 'output_video.mp4', aruco_tracker, 'overlay_image.png')

# Example usage 

aruco_tracker = Aruco_tracker(cam_id=-1, monitor_id=0, debug=False, debug_stream=False, frame_width=width, frame_height=height, crop_img=False, num_tags=15, decision_margin=20, record_stream=False, publish_pos=False, print_pos=False, detect_arena=False)

#overlay_img_robat_path = '/Users/alberto/'  # replace with your overlay image path

draw_trajectories_on_video(input_video_path, output_video_path, aruco_tracker, overlay_img_path)
