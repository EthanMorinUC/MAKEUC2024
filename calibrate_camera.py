import cv2
import numpy as np
import pickle

# Load the image containing multiple ArUco markers
image = cv2.imread('arucos.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters_create()

# Detect the markers in the image
corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

# Assuming the size of the markers in real world units (e.g., 1 unit x 1 unit)
marker_length = 1.0

# If markers are detected
if ids is not None:
    # Calibrate the camera using the detected markers
    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(
        corners, ids, np.array([marker_length]*len(ids)), gray.shape[::-1], None, None
    )

    # Save the calibration results
    with open('my_camera_calibration.p', 'wb') as f:
        pickle.dump((ret, mtx, dist, rvecs, tvecs), f)

    print("Calibration successful. Calibration file saved as my_camera_calibration.p")
else:
    print("No markers detected in the image.")
