import cv2
import numpy as np
from datetime import datetime
from mss import mss
import math
import zmq

class Aruco_tracker:
    def __init__(self, cam_id=-1, monitor_id=0, debug=True, debug_stream=True, frame_width=1920, frame_height=1080, crop_img=False, num_tags=15, decision_margin=20, record_stream=False, publish_pos=False, print_pos=False, detect_arena=False):
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self._parameters = cv2.aruco.DetectorParameters_create()
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
            self._writer = cv2.VideoWriter('init.mp4', cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 20, (frame_width, frame_height))
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
                    self._writer = cv2.VideoWriter(timestamp, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 20, (self._frame_width, self._frame_height))
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

        return positions

    def close(self):
        print("close")
        self._writer.release()

def draw_trajectories_on_video(input_video_path, output_video_path, aruco_tracker):
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 20, (frame_width, frame_height))

    trajectories = {}
    colors = {}

    # Camera calibration parameters (you might need to calibrate your camera)
    camera_matrix = np.array([[1406.08415449821, 0, 0],
                              [2.20679787308599, 1417.99930662800, 0],
                              [1014.13643417416, 566.347754321696, 1]]).reshape(3, 3)

    dist_coeffs = np.array([-0.2380769334, 0.0931325835, 0, 0, 0])
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
              
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_tracker._aruco_dict, parameters=aruco_tracker._parameters)
        #print('ids',ids) 
        if ids is not None:
            ids = ids.flatten()
            for corner, markerID in zip(corners, ids):
                if markerID not in trajectories:
                    trajectories[markerID] = []
                    colors[70] = (0,255,0) #robat
                    colors[markerID] = ()
                    # colors[markerID] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                    


                # Draw 3D axis
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corner, 0.05, camera_matrix, dist_coeffs)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs, tvecs, 0.1)

                center = np.mean(corner[0], axis=0)
                trajectories[markerID].append(center)
                for i in range(1, len(trajectories[markerID])):
                    #print(len(trajectories[markerID]))
                    cv2.line(frame, tuple(trajectories[markerID][i-1].astype(int)), tuple(trajectories[markerID][i].astype(int)), colors[markerID], 3)

        out.write(frame)
        cv2.imshow('Trajectories', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
aruco_tracker = Aruco_tracker(cam_id=-1, monitor_id=0, debug=False, debug_stream=False, frame_width=1920, frame_height=1080, crop_img=False, num_tags=15, decision_margin=20, record_stream=False, publish_pos=False, print_pos=False, detect_arena=False)

linux_path = '/home/adoimo/Desktop/'
#mac_path =  '/Extreme SSD/università/tesi/robat V0 video'
#ssd_path = '/media/adoimo/Extreme SSD/università/tesi/robat V0 video'
video_name = 'Basler_acA1920-40uc__24531279__20240621_173542632.mp4'
input_video_path = linux_path + video_name  # replace with your input video path
#input_video_path = '/Volumes/Extreme SSD/università/tesi/robat V0 video/overhead camera/Basler_acA1920-40uc__24531279__20240621_173535657.mp4'  # replace with your input video path
output_video_path =  linux_path + '/video_out/' + video_name  # replace with your desired output video path
draw_trajectories_on_video(input_video_path, output_video_path, aruco_tracker)
