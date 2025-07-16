from py_console import console

import cv2
import numpy as np
import sys  
import zmq
from datetime import datetime
from mss import mss

import math

# This code defines a class called `Aruco_tracker` that is used for tracking ArUco markers in a video stream or screen capture.
#  Here's a breakdown of what the code does:
# 
# 1. The `__init__` method initializes various parameters and settings, such as the camera ID, frame size, debug mode, record mode, and more.
#   It also sets up the ArUco dictionary and detector parameters.
# 
# 2. If the camera ID is set to -1, it captures the screen using the `mss` library. Otherwise, it opens a video capture device using OpenCV's `cv2.VideoCapture`.
# 
# 3. If debug mode is enabled, it creates a window to display the video feed with detected markers.
# 
# 4. If record mode is enabled, it sets up a ZMQ socket to receive commands for starting and stopping the video recording.
# 
# 5. If position publishing is enabled, it sets up a ZMQ socket to publish the detected marker positions.
# 
# 6. The `detect_arena` method is used to detect four specific ArUco markers (100, 101, 102, 103) that define the corners of the arena.
#   It publishes the detected arena corners over the ZMQ socket.
# 
# 7. The `get_positions` method is the main method that performs the ArUco marker detection and tracking.
#   It captures a frame from the camera or screen, converts it to grayscale, and detects the ArUco markers using OpenCV's `cv2.aruco.detectMarkers` function.
# 
# 8. For each detected marker, it calculates the center position, orientation, and other relevant information.
#   If position publishing is enabled, it publishes the marker positions over the ZMQ socket.
# 
# 9. If debug mode is enabled, it draws bounding boxes, marker IDs, and other information on the video frame and displays it in the window.
# 
# 10. If record mode is enabled, it checks for commands from the ZMQ socket to start or stop recording the video stream.
# 
# 11. If print mode is enabled, it prints the detected marker positions to the console.
# 
# 12. The `close` method is used to release the video writer when the program is closing.
# 
# In summary, this code provides a way to track ArUco markers in a video stream or screen capture, with various options for debugging, recording,
#   and publishing the marker positions over a ZMQ socket.
# 


class Aruco_tracker:
    
    def __init__(self, 
                    cam_id=-1,                   # Camera ID -1 for screen capture
                    monitor_id=0,               # Monitor ID for screen capture
                    debug=True,                # Debug mode
                    debug_stream=False,         # Debug stream
                    frame_width=1920,           # Frame width for input frame
                    frame_height=1080,          # Frame height for input frame
                    crop_img=False,              # Crop image to focus on arena
                    num_tags=15,                # Number of tags to track
                    decision_margin=20,         # Margin of error for tag detection
                    record_stream=False,        # Record stream to file
                    publish_pos=False,          # Publish positions to zmq
                    print_pos=False,            # Print positions to console    
                    detect_arena = False):      # Detect arena and publish to zmq

        """Initializes the positions dictionary with the given active robots' IDs and creates a preview window if in debug mode
        """        
        
        # set dictionary size depending on the aruco marker selected
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

        # detector parameters can be set here (List of detection parameters[3])
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
            # Get desired monitor
            self._sct = mss()
            self._sct.compression_level = 2
            self._monitor = self._sct.monitors[monitor_id]
        else:
            # Get desired video capture device
            self._cam = cv2.VideoCapture(cam_id)
            # self._cam = cv2.VideoCapture('/Volumes/Extreme SSD/università/tesi/robat V0 video/overhead camera/RobatV0 static collision avoidance 1.mp4')

            # Set codec to MJPEG
            self._cam.set(cv2.CAP_PROP_FOURCC, 0x47504A4D)

            # Set image size
            self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

            # Set buffer size
            self._cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Set camera focus manually
            self._cam.set(cv2.CAP_PROP_AUTOFOCUS, False)
            self._cam.set(cv2.CAP_PROP_FOCUS, 0)

        if self._debug_stream:
            # Create window for showing video feed
            cv2.namedWindow("tracking", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("tracking", 700, 700)


        if record_stream:
            self._sub_socket = zmq.Context().socket(zmq.SUB)
            self._sub_socket.connect('tcp://localhost:5556')
            self._sub_socket.setsockopt_string(zmq.SUBSCRIBE, "43")
            self._writer= cv2.VideoWriter('init.mp4', cv2.VideoWriter_fourcc('M','P','4','V'), 20, (frame_width,frame_height))
            self._record = False


        if self._publish_pos:
            self._pub_socket = zmq.Context().socket(zmq.PUB)
            self._pub_socket.bind("tcp://*:5556")               

            if detect_arena:
                self._arena_corners = np.float32([[0,0],[0,0],[0,0],[0,0]])                                      
                self._detect_arena()
     

    def detect_arena(self):
        cap_img = np.array(self._sct.grab(self._monitor))
        gray = cv2.cvtColor(cap_img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, self._aruco_dict, parameters=self._parameters)


        while 100 not in ids or 101 not in ids or 102 not in ids or 103 not in ids:
            cap_img = np.array(self._sct.grab(self._monitor))
            gray = cv2.cvtColor(cap_img, cv2.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, self._aruco_dict, parameters=self._parameters)
            sys.stdout.write("\033[K")
            print("\r", 
            "Arena not detected", end="")
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
        """Returns the global dictionary that maps robots' IDs to their positions"""

        if self._cam_id == -1:
            cap_img = np.array(self._sct.grab(self._monitor))
        else:
            ret, cap_img = self._cam.read()
            record_img = cap_img
            if not ret:
                console.error("Failed to grab frame", severe=True)
                return np.zeros(self._num_tags*2)

            if self._crop_img:
                cap_img = cap_img[0:1080, 420:1500]

        if self._crop_img:
            cap_img = cv2.resize(cap_img, (self._frame_width,self._frame_height), interpolation =cv2.INTER_AREA)


        gray = cv2.cvtColor(cap_img, cv2.COLOR_BGR2GRAY)

        # Set images to use for detecting Markers and for being shown onscreen
        displ_img = cap_img

        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, self._aruco_dict, parameters=self._parameters)

        x_positions = [0] * self._num_tags
        y_positions = [0] * self._num_tags
        omega_angles = [0] * self._num_tags

        positions = np.zeros(self._num_tags*2)

        printpositions = []

        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()

            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):

                # Filter results so only the ones with reasonable certainty are shown
                if markerID < self._num_tags:

                    # extract the marker corners (which are always returned in
                    # top-left, top-right, bottom-right, and bottom-left order)
                    corners = markerCorner.reshape((4, 2))
                    # (topLeft, topRight, bottomRight, bottomLeft) = corners
                    (topRight, bottomRight, bottomLeft, topLeft) = corners.astype(np.int)


                    cX = int((bottomLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((bottomLeft[1] + bottomRight[1]) / 2.0)

                    fX = int((topRight[0] + topLeft[0]) / 2.0)
                    fY = int((topRight[1] + topLeft[1]) / 2.0)


                    v = (cX - fX, cY - fY)
                    v_len = math.sqrt(v[0]**2+v[1]**2)
                    omega = math.atan2(v[1], -v[0])


                    crX = int(cX + math.sin(omega + math.pi/2) * v_len/4)
                    crY = int(cY + math.cos(omega + math.pi/2) * v_len/4)

                    positions[markerID] = crX
                    positions[markerID + self._num_tags] = crY

                    x_positions[markerID]  = crX
                    y_positions[markerID]  = crY

                    printpositions.append([markerID, x_positions[markerID], y_positions[markerID], omega, v])

                    if self._publish_pos:
                        self._pub_socket.send_string("%d %d %d %d %d %d" % (markerID, x_positions[markerID], y_positions[markerID], omega, v[0], v[1]))

                    if self._debug_stream:

                        # draw the bounding box of the ArUCo detection
                        cv2.line(cap_img, topLeft, topRight, (0, 255, 0), 2)
                        cv2.line(cap_img, topRight, bottomRight, (0, 255, 0), 2)
                        cv2.line(cap_img, bottomRight, bottomLeft, (0, 255, 0), 2)
                        cv2.line(cap_img, bottomLeft, topLeft, (0, 255, 0), 2)
                        # compute and draw the center (x, y)-coordinates of the ArUco

                        cv2.circle(cap_img, (crX, crY), 4, (0, 0, 255), -1)
                        cv2.circle(cap_img, (fX, fY), 4, (0, 255, 255), -1)

                        # draw the ArUco marker ID on the image
                        cv2.putText(cap_img, str(markerID),
                            (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)        
        
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
                    timestamp = date_time.strftime("%d%m%Y_%H%M%S")+'.mp4'
                    print(timestamp)
                    self._writer= cv2.VideoWriter(timestamp, cv2.VideoWriter_fourcc('m','p','4','v'), 20, (self._frame_width,self._frame_height))
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
            print("\r", 
            "\t",printpositions, end="")

        
        return positions

    def close(self):
        print("close")
        self._writer.release()

cap = cv2.VideoCapture('/Volumes/Extreme SSD/università/tesi/robat V0 video/overhead camera/RobatV0 static collision avoidance 1.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Process the frame here
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
