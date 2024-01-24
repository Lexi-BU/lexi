import numpy as np
import matplotlib.pyplot as plt
import pickle
import tabulate
from wand.image import Image
from wand.color import Color


# Turn on latex rendering for matplotlib
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

plot_figures = False

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

# Find the distance between the center of each hole and the point with maximum count value
distances = np.linalg.norm(
    max_count_coordinates - xy_new_holes, axis=0
)  # Euclidean distance

# Print the statistics in a table
table = tabulate.tabulate(
    [
        ["Mean", np.nanmean(distances)],
        ["Standard deviation", np.nanstd(distances)],
        ["Minimum", np.nanmin(distances)],
        ["Maximum", np.nanmax(distances)],
    ],
    headers=["Statistics", "Values"],
    tablefmt="fancy_grid",
)

print(table)


def barrel_correction(r, k1, k2, k3, p1, p2):
    """
    Barrel distortion correction function

    Parameters
    ----------
    r : numpy.ndarray
        The radial distance from the center of the image
    k1 : float
        The first radial distortion coefficient
    k2 : float
        The second radial distortion coefficient
    k3 : float
        The third radial distortion coefficient
    p1 : float
        The first tangential distortion coefficient
    p2 : float
        The second tangential distortion coefficient

    Returns
    -------
    numpy.ndarray
        The corrected radial distance from the center of the image
    """
    r2 = r**2
    r4 = r2**2
    r6 = r4 * r2
    r8 = r6 * r2
    return r * (1 + k1 * r2 + k2 * r4 + k3 * r6) + 2 * p1 * r * r2 + p2 * (r4 + 2 * r2)


# Set image backgrounf to white
import matplotlib.pyplot as plt

# Set image to style to default
plt.style.use("default")

img_i = np.rot90(counts.T)
with Image.from_array(img_i) as img:
    img.background_color = Color("white")
    # Set the color of the foreground to black
    # img.colorize("black", "white")

    img.save(filename="../figures/non_lin/original.jpg")


# Apply the barrel distortion correction to the image

with Image.from_array(img_i) as img:
    img.background_color = Color("white")
    # img.background_color = img[0, 0]
    # If the background color is not set, then the image will be black
    # img.alpha_channel = "remove"

    img.virtual_pixel = "background"

    args = (1.0, 2.1, -0.1, 1.1)
    img.distort("barrel", args)

    # Save img as an array
    img_array = np.array(img)

    # Save the image to a pdf file and set its resolution to 300 dpi
    img.save(filename="../figures/non_lin/barrel.png")


# Define the max_count_coordinates_barrel as an array of NaN values and same shape as xy_new_holes
max_count_coordinates_barrel = np.full_like(xy_new_holes, np.nan)

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
        max_count_coordinates_barrel[0][np.where(xy_new_holes[0] == xpoint)] = x[
            max_count_index_2d[1]
        ]
        max_count_coordinates_barrel[1][np.where(xy_new_holes[1] == ypoint)] = y[
            max_count_index_2d[0]
        ]
    except Exception:
        # print(f"For {xpoint, ypoint}, {e}")
        continue
