import numpy as np
from scipy.spatial import procrustes
import matplotlib.pyplot as plt

# Generate some example data
np.random.seed(0)
radius = 4.0
theta = np.linspace(0, 2 * np.pi, 53)
actual_location = xy_new_holes.T

# Introduce some noise to simulate measurement error
noise = np.random.normal(0, 0.2, actual_location.shape)
measured_location = max_count_coordinates.T

# Drop the indices where either the actual or measured location is NaN
nan_indices = np.where(np.isnan(actual_location) | np.isnan(measured_location))
actual_location = np.delete(actual_location, nan_indices, 0)
measured_location = np.delete(measured_location, nan_indices, 0)

# Use Procrustes analysis to find the transformation
mtx1, mtx2, disparity = procrustes(actual_location, measured_location)

new_xy_new_holes = actual_location - np.mean(actual_location, 0)
norm_xy_new_holes = np.linalg.norm(new_xy_new_holes)

new_mtx1 = norm_xy_new_holes * mtx1 + np.mean(actual_location, 0)

new_measured_location = measured_location - np.mean(measured_location, 0)
norm_measured_location = np.linalg.norm(new_measured_location)

new_mtx2 = mtx2 * norm_measured_location + np.mean(measured_location, 0)

# Get the new coordinates of the measured location
new_measured_location = mtx2.T

# Plot the results
plt.figure()
plt.scatter(actual_location[:, 0], actual_location[:, 1], s=10, label="Actual")
plt.scatter(
    measured_location[:, 0], measured_location[:, 1], s=5, label="Measured", alpha=0.5
)
plt.scatter(new_mtx2[:, 0], new_mtx2[:, 1], s=1, label="Procrustes", alpha=0.5)

# Make a circle of radius 4
circle = plt.Circle((0, 0), radius, color="k", fill=False)
plt.gca().add_artist(circle)

# Set the axis limits
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.legend()
plt.axis("equal")

# Save the figure
plt.savefig("../figures/non_lin/procrustes_example.png", dpi=300)
