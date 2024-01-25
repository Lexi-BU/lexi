import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle


# Define the row number for each row
row_num = np.array([4, 3, 2, 1, 0, -1, -2, -3, -4])

# Define the number of holes in each of the nine rows
num_holes = np.array([1, 5, 7, 7, 9, 7, 7, 5, 1])

# Define the distance between adjacent holes in each of the nine rows
dist_h_inch = 0.394
dist_h_cm = dist_h_inch * 2.54

# Define the distance from x-axis to the center of the first hole in each of the nine rows
dist_x_cm = row_num * dist_h_cm

# Define the diameter of the holes
d_h_inch = 0.020
d_h_cm = d_h_inch * 2.54

# Define the x and y coordinates of each of the holes
x_holes = np.array([])
y_holes = np.array([])
for i in range(0, len(row_num)):
    hole_number = np.arange(-(num_holes[i] - 1) / 2, (num_holes[i] - 1) / 2 + 1, 1)
    y_holes = np.append(y_holes, dist_x_cm[i] * np.ones(len(hole_number)))
    x_holes = np.append(x_holes, hole_number * dist_h_cm)

# Add the location of four more holes separately
x_holes = np.append(x_holes, np.array([-0.197, 0.197, 0.984, -0.197]) * 2.54)
y_holes = np.append(y_holes, np.array([-0.591, -0.197, 0.197, 1.378]) * 2.54)
xy_holes = np.array([x_holes, y_holes])

# Define a new coordinate system where the previous coordinate system is rotated by 45 degrees
theta = np.radians(-44.5)
theta_deg = np.degrees(theta)
c, s = np.cos(theta), np.sin(theta)
R = np.array(((c, -s), (s, c)))
xy_new_holes = np.dot(R, xy_holes)

# Define the coordinates of a rectangle in the distorted image
pts1 = xy_new_holes.T.astype(np.float32)

# Define the pickle file name
pickle_file = (
    "../data/2022_04_28_1313_LEXI_Sci_unit1_mcp_unit1_eBox_1900V_counts.pickle"
)

# Open the pickle file
with open(pickle_file, "rb") as f:
    counts = pickle.load(f)
    xedges = pickle.load(f)
    yedges = pickle.load(f)

# Close the pickle file
f.close()

counts = counts.T

# Apply gaussian smoothing to the image
counts = cv2.GaussianBlur(counts, (5, 5), 0)


# Define the x and y coordinates of the center of each pixel
x = np.array([])
y = np.array([])
for i in range(0, len(xedges) - 1):
    x = np.append(x, (xedges[i] + xedges[i + 1]) / 2)
    y = np.append(y, (yedges[i] + yedges[i + 1]) / 2)


# At each xy_new coordinate, find the location of point with maximum count value within a circle of
# radius 0.5 cm


# Create a meshgrid from xedges and yedges
X, Y = np.meshgrid(x, y)

# Combine x, y coordinates into a 2D array
xy_coordinates = np.column_stack((X.ravel(), Y.ravel()))

# Define max_count_coordinates as an array of NaN values and same shape as xy_new_holes
max_count_coordinates = np.full_like(xy_new_holes, np.nan)

# Radius for searching
search_radius = 0.35  # You can adjust this based on your requirement
search_radius_array = np.full_like(xy_new_holes[0], search_radius)

# Make the radius corresponding to following indices smaller:
# Indices: 26
# Radius: 0.25
search_radius_array[26] = 0.25

# Make the radius corresponding to following indices bigger:
# Indices: 5, 35, 45, 46
# Radius: 0.45
search_radius_array[[5, 35, 45, 46]] = 0.45

# Ignore the following indices:
# Indices: 0, 20, 27, 28, 42, 47, 48
# Radius: NaN
search_radius_array[[0, 20, 27, 28, 42, 47, 48]] = np.nan

# Iterate through each point in xy_new_holes
for idx, (xpoint, ypoint) in enumerate(zip(xy_new_holes[0][0:], xy_new_holes[1][0:])):
    try:
        # Calculate the Euclidean distance
        distances = np.linalg.norm(xy_coordinates - np.array([xpoint, ypoint]), axis=1)

        # Find the indices within the specified search radius
        indices_within_radius = np.where(distances <= search_radius_array[idx])[0]

        # Find the index where the count is maximum within the specified search radius
        max_count_index = np.nanargmax(counts.ravel()[indices_within_radius])

        # Convert the 1D index to 2D indices
        max_count_index_2d = np.unravel_index(
            indices_within_radius[max_count_index], counts.shape
        )

        # Add the coordinates of the point with maximum count value to max_count_coordinates
        max_count_coordinates[0][np.where(xy_new_holes[0] == xpoint)] = x[
            max_count_index_2d[1]
        ]
        max_count_coordinates[1][np.where(xy_new_holes[1] == ypoint)] = y[
            max_count_index_2d[0]
        ]
    except Exception:
        # print(f"For {xpoint, ypoint}, {e}")
        continue


# Define the corresponding coordinates in the undistorted image
pts2 = max_count_coordinates.T.astype(np.float32)

# Convert points to the format required by OpenCV
undistorted_points = pts1.reshape((-1, 1, 2))
distorted_points = pts2.reshape((-1, 1, 2))

# Read the distorted image
distorted_img = cv2.imread(
    "../figures/non_lin/original.jpg"
)  # Replace with your actual image path

gray = cv2.cvtColor(distorted_img, cv2.COLOR_BGR2GRAY)

# Get the width and height of the image
width = distorted_img.shape[1]
height = distorted_img.shape[0]

# Define the 3D coordinates of the calibration pattern (dummy values since we don't have a calibration pattern)
object_points = np.zeros((len(pts1), 1, 3), np.float32)

# Calibrate the camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    object_points, distorted_points, gray.shape[::-1], None, None
)

# Calculate the camera matrix
camera_matrix = cv2.initUndistortRectifyMap(
    undistorted_points,
    distorted_points,
    np.eye(3),
    cameraMatrix=np.eye(3),
    size=(width, height),
)[0]

# Apply the camera matrix to the distorted image
undistorted_img = cv2.remap(
    distorted_img, camera_matrix, None, interpolation=cv2.INTER_LINEAR
)

# Save the undistorted image
cv2.imwrite("../figures/non_lin/undistorted.jpg", undistorted_img)
