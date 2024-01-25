import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.gaussian_process.kernels import (
    RBF,
    WhiteKernel,
    Matern,
    RationalQuadratic,
    ExpSineSquared,
    DotProduct,
)
from sklearn.gaussian_process import GaussianProcessRegressor

# Turn on latex rendering for matplotlib
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

plot_figures = True

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
apply_smoothing = False
if apply_smoothing:
    # Apply gaussian smoothing to the image
    counts_smoothed = cv2.GaussianBlur(counts, (3, 3), 0)
else:
    counts_smoothed = counts

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
        max_count_index = np.nanargmax(counts_smoothed.ravel()[indices_within_radius])

        # Convert the 1D index to 2D indices
        max_count_index_2d = np.unravel_index(
            indices_within_radius[max_count_index], counts_smoothed.shape
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

# Take the transpose of max_count_coordinates and xy_new_holes
actual_position = xy_new_holes.T
measured_position = max_count_coordinates.T

# Get rid of the NaN values in actual_position and measured_position
nan_indices = np.where(np.isnan(actual_position) | np.isnan(measured_position))
actual_position = np.delete(actual_position, nan_indices, 0)
measured_position = np.delete(measured_position, nan_indices, 0)

# Get the values of xy_new_holes at points where max_count_coordinates is NaN
xy_new_holes_nan = np.array(
    [
        xy_new_holes[0][np.isnan(max_count_coordinates[0])],
        xy_new_holes[1][np.isnan(max_count_coordinates[1])],
    ]
)

# Define a list of length scales
length_scale_list = np.arange(1, 10, 1)
# Define a list of n_restarts_optimizer
n_restarts_optimizer = 10
# Define a list of kernel names using a combination of single or multiple kernels
kernel_name_list = [
    RBF(length_scale=5, length_scale_bounds=(1e-2, 1e2)),
    RBF(length_scale=2, length_scale_bounds=(1e-2, 1e2)),
    RBF(length_scale=3, length_scale_bounds=(1e-2, 1e2)),
    RBF(length_scale=4, length_scale_bounds=(1e-2, 1e2))
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1)),
    RBF(length_scale=5, length_scale_bounds=(1e-2, 1e2))
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1)),
    Matern(length_scale=5, length_scale_bounds=(1e-2, 1e2), nu=1.5),
    Matern(length_scale=5, length_scale_bounds=(1e-2, 1e2), nu=2.5),
    Matern(length_scale=5, length_scale_bounds=(1e-2, 1e2), nu=0.5),
    RationalQuadratic(length_scale=5, length_scale_bounds=(1e-2, 1e2), alpha=0.1),
    RationalQuadratic(length_scale=5, length_scale_bounds=(1e-2, 1e2), alpha=0.5),
    RationalQuadratic(length_scale=5, length_scale_bounds=(1e-2, 1e2), alpha=1),
    RBF(length_scale=5, length_scale_bounds=(1e-2, 1e2))
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1))
    + DotProduct(sigma_0=1, sigma_0_bounds=(1e-2, 1e2)),
    RBF(length_scale=5, length_scale_bounds=(1e-2, 1e2))
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1))
    + DotProduct(sigma_0=2, sigma_0_bounds=(1e-2, 1e2)),
    RBF(length_scale=5, length_scale_bounds=(1e-2, 1e2))
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1))
    + DotProduct(sigma_0=3, sigma_0_bounds=(1e-2, 1e2)),
    Matern(length_scale=5, length_scale_bounds=(1e-2, 1e2), nu=1.5)
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1)),
    Matern(length_scale=5, length_scale_bounds=(1e-2, 1e2), nu=2.5)
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1)),
    Matern(length_scale=5, length_scale_bounds=(1e-2, 1e2), nu=0.5)
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1)),
    RationalQuadratic(length_scale=5, length_scale_bounds=(1e-2, 1e2), alpha=0.1)
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1)),
    RationalQuadratic(length_scale=5, length_scale_bounds=(1e-2, 1e2), alpha=0.5)
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1)),
    RationalQuadratic(length_scale=5, length_scale_bounds=(1e-2, 1e2), alpha=1)
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1)),
    ExpSineSquared(
        length_scale=5,
        length_scale_bounds=(1e-2, 1e2),
        periodicity=3,
        periodicity_bounds=(1e-2, 1e2),
    )
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1)),
    ExpSineSquared(
        length_scale=5,
        length_scale_bounds=(1e-2, 1e2),
        periodicity=5,
        periodicity_bounds=(1e-2, 1e2),
    )
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1)),
    ExpSineSquared(
        length_scale=5,
        length_scale_bounds=(1e-2, 1e2),
        periodicity=7,
        periodicity_bounds=(1e-2, 1e2),
    )
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1)),
    DotProduct(sigma_0=1, sigma_0_bounds=(1e-2, 1e2))
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1)),
]

# Compute the shift in the x and y coordinates of the holes between the actual and measured
# positions
delta_x = measured_position[:, 0] - actual_position[:, 0]
delta_y = measured_position[:, 1] - actual_position[:, 1]
delta_xy = np.array([delta_x, delta_y])

for idx, kernel in enumerate(kernel_name_list[:]):
    # Define the kernel
    length_scale = 3.0
    # kernel = RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2))
    kernel = kernel
    # # Define the the number of restarts for the optimizer
    # n_restarts_optimizer = 10
    alpha = 0.0

    # Define the Gaussian process regressor
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=n_restarts_optimizer,
    )
    # Print the names of all the kernels in each kernel
    kernel_name = str(kernel)

    # Get the training and test data
    rng_state_value = 4
    rng = np.random.RandomState(rng_state_value)
    training_fraction = 0.8
    n_train = int(training_fraction * len(delta_x))

    # Define the training and test data chosen randomly
    train_idx = rng.permutation(len(delta_x))[:n_train]
    test_idx = rng.permutation(len(delta_x))[n_train:]

    # Train the Gaussian process regressor on the actual_position as input and delta_xy as output
    gp.fit(measured_position[train_idx], delta_xy[:, train_idx].T)

    # Predict the delta_xy for the test data
    y_pred, sigma = gp.predict(measured_position[test_idx], return_std=True)

    # Calculate the mean absolute error
    mae = np.mean(np.abs(y_pred - delta_xy[:, test_idx].T), axis=0)

    # Calculate the mean absolute percentage error
    mape = np.mean(
        np.abs((y_pred - delta_xy[:, test_idx].T) / delta_xy[:, test_idx].T) * 100,
        axis=0,
    )

    # Calculate the root mean square error
    rmse = np.sqrt(np.mean((y_pred - delta_xy[:, test_idx].T) ** 2, axis=0))

    # Predict the delta_xy for the entire data
    y_pred_all, sigma_all = gp.predict(measured_position, return_std=True)

    modified_measured_position = measured_position - y_pred_all

    # Data file name
    # kernel_name = str(kernel).split("(")[0]
    gp_data_file = f"../data/gp_data_{length_scale}_{n_restarts_optimizer}_{alpha}_{training_fraction}_{rng_state_value}_{kernel_name}.pickle"
    # Save the trained Gaussian process regressor
    with open(gp_data_file, "wb") as f:
        pickle.dump(gp, f)

    # Close the pickle file
    f.close()

    run_gp_hist = True
    if run_gp_hist:
        # Convert the counts to a 1D array
        counts_1d = counts.ravel()

        # Drop all the locations from xy_coordinates where the counts are zero or NaN
        xy_coordinates_non_zero = xy_coordinates[
            (counts_1d != 0) & (~np.isnan(counts_1d))
        ]

        # Get counts at the locations where the counts are non-zero
        counts_non_zero = counts_1d[(counts_1d != 0) & (~np.isnan(counts_1d))]

        # Predict the delta_xy for all the points in xy_coordinates
        y_pred_all_counts, sigma_all_counts = gp.predict(
            xy_coordinates_non_zero, return_std=True
        )

        # Calculate the modified measured position for all the points in xy_coordinates
        modified_measured_position_all_counts = (
            xy_coordinates_non_zero - y_pred_all_counts
        )

    # Plot the actual_position, measured_position and modified_measured_position on a scatter plot
    plt.figure(figsize=(8, 8))
    modified = True

    if modified:
        hist_x = modified_measured_position_all_counts[:, 0]
        hist_y = modified_measured_position_all_counts[:, 1]
    else:
        hist_x = xy_coordinates_non_zero[:, 0]
        hist_y = xy_coordinates_non_zero[:, 1]

    plt.scatter(
        hist_x,
        hist_y,
        s=2,
        color="gray",
        marker=".",
        label="Data points",
        alpha=0.2,
    )

    plt.scatter(
        actual_position[:, 0],
        actual_position[:, 1],
        s=20,
        color="b",
        marker="o",
        facecolors="none",
        edgecolors="b",
        label="Hole Locations",
        alpha=1,
    )

    plt.scatter(
        xy_new_holes_nan[0],
        xy_new_holes_nan[1],
        s=10,
        color="k",
        marker="o",
        facecolors="none",
        edgecolors="k",
        alpha=0.8,
    )

    plt.scatter(
        measured_position[:, 0],
        measured_position[:, 1],
        s=5,
        color="r",
        marker="d",
        label="Measured position",
        alpha=1,
    )

    plt.scatter(
        modified_measured_position[:, 0],
        modified_measured_position[:, 1],
        s=5,
        color="k",
        marker="8",
        label="Modified measured position",
        alpha=0.5,
    )

    # Add a circle of radius 4 cm
    circle = plt.Circle((0, 0), 4, ls="--", lw=0.5, color="k", fill=False)
    plt.gca().add_artist(circle)

    # plt.pcolormesh(xedges, yedges, counts, cmap="gray", shading="auto", alpha=0.5)

    # Add a legend to the plot right outside the plot and align it to center
    plt.legend(bbox_to_anchor=(0.05, 1.18), loc="upper center", borderaxespad=0.0)

    plt.xlim(-4, 4)
    plt.ylim(-4, 4)

    plt.axis("equal")
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")

    # Print the mean absolute error, mean absolute percentage error and root mean square error on the
    # plot
    plt.text(
        0.05,
        0.95,
        f"MAE = {mae[0]:.2f} cm\nMAPE = {mape[0]:.2f} %\nRMSE = {rmse[0]:.2f} cm",
        horizontalalignment="left",
        verticalalignment="top",
        transform=plt.gca().transAxes,
    )
    if modified:
        hist_type = "modified"
    else:
        hist_type = "original"
    plt.title(f"{kernel_name}", fontdict={"fontsize": 10})
    plt.savefig(
        f"../figures/non_lin/gp_list/gp_actual_and_measured_positions_hist_{kernel_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    # Turn off y-axis
    plt.gca().axes.get_yaxis().set_visible(False)
