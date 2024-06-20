# import the necessary packages
import numpy as np
import cv2


num_markers = 15
marker_size = 370
x_offset = 85
y_offset = 1024 - marker_size - 44 -15

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# allocate memory for the output ArUCo tag and then draw the ArUCo
# tag on the output image
for id in range(num_markers):

    tag = np.zeros((marker_size, marker_size, 1), dtype="uint8")
    cv2.aruco.generateImageMarker(arucoDict, id, marker_size, tag, 1)

    cv2.imwrite('plainDICT_6X6_250-'+str(id)+'.png', tag)
