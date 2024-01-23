import numpy as np
import matplotlib.pyplot as plt
import pickle

# Turn on latex rendering for matplotlib
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

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

"""
# Plot the coordinates of the holes in the new coordinate system
plt.figure()
plt.scatter(xy_new_holes[0], xy_new_holes[1], s=5, color="r")

# Draw a dashed line to show the x-axis and y-axis
# plt.plot([-4, 4], [0, 0], "--", color="k")
# plt.plot([0, 0], [-4, 4], "--", color="k")

plt.xlabel("x (cm)")
plt.ylabel("y (cm)")
plt.title(
    f"Coordinates of the holes at $\\theta = {theta_deg}^{{\\circ}}$", fontsize=14
)
# Add a circle to the plot of radius 4.5 cm
circle = plt.Circle((0, 0), 4, color="b", fill=False, linestyle="--", alpha=0.2)
plt.gca().add_artist(circle)
plt.axis("equal")

plt.xlim(-4, 4)
plt.ylim(-4, 4)
# Save the image as png with transparent background
plt.savefig("../figures/coordinates.png", dpi=300, transparent=True)
"""

# Define the pickle file name
pickle_file = (
    "../data/2022_04_28_1313_LEXI_Sci_unit1_mcp_unit1_eBox_1900V_counts.pickle"
)

# Open the pickle file
with open(pickle_file, "rb") as f:
    counts = pickle.load(f)
    xedges = pickle.load(f)
    yedges = pickle.load(f)

counts = counts.T
# Close the pickle file
f.close()

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

# Initialize an array to store the result
max_count_coordinates = []

# Radius for searching
search_radius = 0.25  # You can adjust this based on your requirement

# Iterate through each point in xy_new_holes
for xpoint, ypoint in zip(xy_new_holes[0], xy_new_holes[1]):
    # Calculate the Euclidean distance
    distances = np.linalg.norm(xy_coordinates - np.array([xpoint, ypoint]), axis=1)

    # Find the indices within the specified search radius
    indices_within_radius = np.where(distances <= search_radius)[0]

    # Find the index with the maximum count
    max_count_index = np.argmax(counts.ravel()[indices_within_radius])

    # Convert the 1D index to 2D indices
    max_count_index_2d = np.unravel_index(
        indices_within_radius[max_count_index], counts.shape
    )

    # Append the coordinates where counts are maximum to the result array
    max_count_coordinates.append((X[max_count_index_2d], Y[max_count_index_2d]))

# Print the result
for idx, coordinates in enumerate(max_count_coordinates):
    print(f"Hole {idx + 1}: Coordinates with max count - {coordinates}")

# Plot the coordinates on a scatter plot
plt.figure()
plt.scatter(xy_new_holes[0], xy_new_holes[1], s=5, color="b")
# At each xy_new_holes coordinate, plot a circle of radius search_radius
for xpoint, ypoint in zip(xy_new_holes[0], xy_new_holes[1]):
    circle = plt.Circle((xpoint, ypoint), search_radius, color="r", fill=False)
    plt.gca().add_artist(circle)

plt.scatter(
    np.array(max_count_coordinates)[:, 0],
    np.array(max_count_coordinates)[:, 1],
    s=5,
    color="r",
)

# Scatter plot the counts on the image using pcolormesh function with counts as the color map and
# xedges and yedges as the x and y coordinates
plt.pcolormesh(xedges, yedges, counts, cmap="gray", shading="auto", alpha=0.5)


# Save the figure
plt.xlabel("x (cm)")
plt.ylabel("y (cm)")

circle = plt.Circle((0, 0), 4, color="k", fill=False)
plt.gca().add_artist(circle)
plt.axis("equal")

plt.xlim(-4, 4)
plt.ylim(-4, 4)

plt.title("Mask and Data Overlay")
plt.savefig("../figures/coordinates_max_count_v2.png", dpi=300)
