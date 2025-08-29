import os
import numpy as np
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

marker_image = aruco.generateImageMarker(aruco_dict, 200, 220, 1)
cv2.imwrite('marker_200.png', marker_image)
plt.imshow(marker_image, cmap='gray')
plt.show()