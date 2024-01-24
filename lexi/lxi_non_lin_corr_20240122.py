import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import minimize

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

if plot_figures:
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

    except Exception as e:
        print(f"For {xpoint, ypoint}, {e}")
        continue

# Print the result
# for idx, coordinates in enumerate(max_count_coordinates):
#     print(f"Hole {idx + 1}: Coordinates with max count - {coordinates}")

if plot_figures:
    # Plot the coordinates on a scatter plot
    plt.figure()
    plt.scatter(
        xy_new_holes[0], xy_new_holes[1], s=5, color="b", label="Location of holes"
    )

    # Add a text label outside each circle to show the hole number
    for idx, (xcoordinates, ycoordinates) in enumerate(
        zip(xy_new_holes[0], xy_new_holes[1])
    ):
        # print(
        #     f"Hole {hole_count + 1}: Coordinates with max count - {xcoordinates}, {ycoordinates}"
        # )
        if np.isnan(search_radius_array[idx]):
            plt.text(
                xcoordinates + 0.1,
                ycoordinates + 0.1,
                f"{idx + 1}",
                fontsize=8,
                color="b",
            )
        else:
            plt.text(
                xcoordinates + search_radius_array[idx] / np.sqrt(2),
                ycoordinates + search_radius_array[idx] / np.sqrt(2),
                f"{idx + 1}",
                fontsize=8,
                color="b",
            )
    # At each xy_new_holes coordinate, plot a circle of radius search_radius
    for idx, (xpoint, ypoint) in enumerate(zip(xy_new_holes[0], xy_new_holes[1])):
        circle = plt.Circle(
            (xpoint, ypoint),
            search_radius_array[idx],
            ls="--",
            lw=0.5,
            color="r",
            fill=False,
        )
        plt.gca().add_artist(circle)

    plt.scatter(
        np.array(max_count_coordinates)[0],
        np.array(max_count_coordinates)[1],
        s=5,
        color="r",
        label="Location of max count",
    )

    # Scatter plot the counts on the image using pcolormesh function with counts as the color map and
    # xedges and yedges as the x and y coordinates
    plt.pcolormesh(xedges, yedges, counts, cmap="gray", shading="auto", alpha=0.5)

    # Save the figure
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")

    circle = plt.Circle((0, 0), 4, ls="--", lw=0.5, color="k", fill=False)
    plt.gca().add_artist(circle)
    plt.axis("equal")

    plt.xlim(-4, 4)
    plt.ylim(-4, 4)

    # Add a legend to the plot right outside the plot and align it to center
    plt.legend(bbox_to_anchor=(-0.05, 1.1), loc="upper left", borderaxespad=0.0)

    # plt.tight_layout()

    plt.title("Mask and Data Overlay")
    plt.savefig("../figures/coordinates_max_count_v3.png", dpi=300)


# Find the distance between the center of each hole and the point with maximum count value
distances = np.linalg.norm(
    max_count_coordinates - xy_new_holes, axis=0
)  # Euclidean distance

# Print summary statistics
print(f"Mean distance: {np.nanmean(distances)}")
print(f"Median distance: {np.nanmedian(distances)}")
print(f"Standard deviation: {np.nanstd(distances)}")
print(f"Minimum distance: {np.nanmin(distances)}")
print(f"Maximum distance: {np.nanmax(distances)}")
print(f"Sum of distances: {np.nansum(distances)}")


# Write a function to dewarp the max_count_coordinates in a way that the distnnace between the center
# of each hole and the point with maximum count value is minimized
def dewarp_coordinates(xy_new_holes, max_count_coordinates):
    """
    Dewarp the coordinates of the holes in a way that the distance between the center of each hole
    and the point with maximum count value is minimized

    Parameters
    ----------
    xy_new_holes : numpy.ndarray
        The x and y coordinates of the holes in the new coordinate system
    max_count_coordinates : numpy.ndarray
        The x and y coordinates of the point with maximum count value

    Returns
    -------
    dewarped_coordinates : numpy.ndarray
        The dewarped x and y coordinates of the holes in the new coordinate system
    """

    # Define the function to minimize
    def func_to_minimize(x, xy_new_holes, max_count_coordinates):
        """
        Function to minimize

        Parameters
        ----------
        x : numpy.ndarray
            The x and y coordinates of the holes in the new coordinate system
        xy_new_holes : numpy.ndarray
            The x and y coordinates of the holes in the new coordinate system
        max_count_coordinates : numpy.ndarray
            The x and y coordinates of the point with maximum count value

        Returns
        -------
        distance : float
            The distance between the center of each hole and the point with maximum count value
        """
        # Reshape x to a 2D array
        x = x.reshape(2, -1)

        # Calculate the distance between the center of each hole and the point with maximum count
        # value
        distance = np.linalg.norm(x - max_count_coordinates, axis=0)

        # print(f"Distance: {np.nansum(distance)}")
        return distance

    # Define the initial guess
    x0 = xy_new_holes.ravel()

    # Define the bounds
    bounds = np.vstack(
        (
            np.full_like(xy_new_holes.ravel(), -np.inf),
            np.full_like(xy_new_holes.ravel(), np.inf),
        )
    ).T

    # Define the constraints
    constraints = {"type": "eq", "fun": lambda x: x.reshape(2, -1).sum(axis=1)}

    # Minimize the function
    res = minimize(
        func_to_minimize,
        x0,
        args=(xy_new_holes, max_count_coordinates),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    # Reshape the result to a 2D array
    dewarped_coordinates = res.x.reshape(2, -1)

    return dewarped_coordinates


# Dewarp the coordinates of the holes in a way that the distance between the center of each hole
# and the point with maximum count value is minimized
dewarped_coordinates = dewarp_coordinates(xy_new_holes, max_count_coordinates)

# Plot the dewarped coordinates on a scatter plot
plot_figures = True
if plot_figures:
    plt.figure()

    plt.scatter(
        xy_new_holes[0],
        xy_new_holes[1],
        s=5,
        color="b",
        label="Location of holes",
        alpha=1,
    )
    plt.scatter(
        dewarped_coordinates[0],
        dewarped_coordinates[1],
        s=5,
        color="r",
        alpha=0.5,
        label="Dewarped coordinates",
    )

    # Add a text label outside each circle to show the hole number
    for idx, (xcoordinates, ycoordinates) in enumerate(
        zip(xy_new_holes[0], xy_new_holes[1])
    ):
        # print(
        #     f"Hole {hole_count + 1}: Coordinates with max count - {xcoordinates}, {ycoordinates}"
        # )
        if np.isnan(search_radius_array[idx]):
            plt.text(
                xcoordinates + 0.1,
                ycoordinates + 0.1,
                f"{idx + 1}",
                fontsize=8,
                color="b",
            )
        else:
            plt.text(
                xcoordinates + search_radius_array[idx] / np.sqrt(2),
                ycoordinates + search_radius_array[idx] / np.sqrt(2),
                f"{idx + 1}",
                fontsize=8,
                color="b",
            )
    # At each xy_new_holes coordinate, plot a circle of radius search_radius
    for idx, (xpoint, ypoint) in enumerate(zip(xy_new_holes[0], xy_new_holes[1])):
        circle = plt.Circle(
            (xpoint, ypoint),
            search_radius_array[idx],
            ls="--",
            lw=0.5,
            color="r",
            fill=False,
        )
        plt.gca().add_artist(circle)

    # Scatter plot the counts on the image using pcolormesh function with counts as the color map and
    # xedges and yedges as the x and y coordinates
    plt.pcolormesh(xedges, yedges, counts, cmap="gray", shading="auto", alpha=0.5)

    # Save the figure
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")

    circle = plt.Circle((0, 0), 4, ls="--", lw=0.5, color="k", fill=False)
    plt.gca().add_artist(circle)
    plt.axis("equal")

    plt.xlim(-4, 4)
    plt.ylim(-4, 4)

    # Add a legend to the plot right outside the plot and align it to center
    plt.legend(bbox_to_anchor=(-0.05, 1.1), loc="upper left", borderaxespad=0.0)

    # plt.tight_layout()

    plt.title("Mask and Data Overlay")
    plt.savefig("../figures/dewarped_coordinates.png", dpi=300)


new_dist = np.linalg.norm(dewarped_coordinates - xy_new_holes, axis=0)
print(f"New mean distance: {np.nanmean(new_dist)}")
print(f"New median distance: {np.nanmedian(new_dist)}")
print(f"New standard deviation: {np.nanstd(new_dist)}")
print(f"New minimum distance: {np.nanmin(new_dist)}")
print(f"New maximum distance: {np.nanmax(new_dist)}")
print(f"New sum of distances: {np.nansum(new_dist)}")
