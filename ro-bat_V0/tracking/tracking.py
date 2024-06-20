
import cv2

# Load the video
cap = cv2.VideoCapture('/Volumes/Extreme SSD/università/tesi/robat V0 video/overhead camera/RobatV0 static collision avoidance 1.mp4')

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()



# Initialize the background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply background subtraction
    fgmask = fgbg.apply(frame)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        # Filter out small contours to remove noise
        if cv2.contourArea(cnt) < 500:
            continue
        
        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture('/Volumes/Extreme SSD/università/tesi/robat V0 video/overhead camera/RobatV0 static collision avoidance 1.mp4')

# Get the video frame width, height, and frames per second (fps)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

# Initialize the background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Initialize a list to hold all trackers
trackers  = cv2.legacy.MultiTracker_create()

# List to hold the trajectories of the objects
trajectories = []

# Flag to check if tracking is initialized
tracking_initialized = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if not tracking_initialized:
        # Apply background subtraction
        fgmask = fgbg.apply(frame)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            # Filter out small contours to remove noise
            if cv2.contourArea(cnt) < 500:
                continue
            
            # Get bounding box of the contour
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Initialize tracker for each bounding box
            tracker = cv2.TrackerCSRT_create()
            trackers.add(tracker, frame, (x, y, w, h))
            
            # Initialize trajectory list for this object
            trajectories.append([(x + w // 2, y + h // 2)])
        
        tracking_initialized = True
    else:
        # Update trackers
        success, boxes = trackers.update(frame)
        
        for i, newbox in enumerate(boxes):
            x, y, w, h = [int(v) for v in newbox]
            center = (x + w // 2, y + h // 2)
            trajectories[i].append(center)
            
            # Draw the bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
            
            # Draw the trajectory
            for j in range(1, len(trajectories[i])):
                if trajectories[i][j - 1] is None or trajectories[i][j] is None:
                    continue
                cv2.line(frame, trajectories[i][j - 1], trajectories[i][j], (0, 0, 255), 2)
    
    # Write the frame to the output video
    out.write(frame)
    
    # Display the resulting frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
