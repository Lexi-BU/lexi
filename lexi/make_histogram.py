import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob

data_folder = "/home/vetinari/Desktop/git/Lexi-Bu/lexi/data/from_PIT/20230816/processed_data/sci/level_1c/"
file_list = np.sort(glob.glob(data_folder + "*.csv"))

# df_list = []
# for file in file_list:
#     df_list.append(pd.read_csv(file))
#
# df = pd.concat(df_list, ignore_index=True)

x_mcp_1 = df["x_mcp_1"]
y_mcp_1 = df["y_mcp_1"]

x_mcp_lin = df["x_mcp_lin"]
y_mcp_lin = df["y_mcp_lin"]

x_mcp_nln = df["x_mcp_nln"]
y_mcp_nln = df["y_mcp_nln"]

x_mcp = df["x_mcp"]
y_mcp = df["y_mcp"]

# Get rid of all the NaNs
x_mcp_1 = x_mcp_1[~np.isnan(x_mcp_1) & ~np.isnan(y_mcp_1)]
y_mcp_1 = y_mcp_1[~np.isnan(y_mcp_1) & ~np.isnan(x_mcp_1)]

x_mcp_lin = x_mcp_lin[~np.isnan(x_mcp_lin) & ~np.isnan(y_mcp_lin)]
y_mcp_lin = y_mcp_lin[~np.isnan(y_mcp_lin) & ~np.isnan(x_mcp_lin)]

x_mcp_nln = x_mcp_nln[~np.isnan(x_mcp_nln) & ~np.isnan(y_mcp_nln)]
y_mcp_nln = y_mcp_nln[~np.isnan(y_mcp_nln) & ~np.isnan(x_mcp_nln)]

x_mcp = x_mcp[~np.isnan(x_mcp) & ~np.isnan(y_mcp)]
y_mcp = y_mcp[~np.isnan(y_mcp) & ~np.isnan(x_mcp)]

# Define the x and y ranges for computing the histogram between -10 and 10
x_range = [-10, 10]
y_range = [-10, 10]
hist_range = [x_range, y_range]
c_min = 10
n_bins = 40

# Make a hostigram of the x_mcp_1 and y_mcp_1, and x_mcp_lin and y_mcp_lin, and x_mcp_nln and
# y_mcp_nln
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].hist2d(
    x_mcp_1,
    y_mcp_1,
    range=hist_range,
    bins=n_bins,
    cmap="inferno",
    norm=mpl.colors.LogNorm(vmin=1),
    cmin=c_min,
)
# Add the colorbar
cbar = fig.colorbar(ax[0, 0].collections[0], ax=ax[0, 0])
cbar.set_label("Counts")

# Add the title to the figure
ax[0, 0].set_title("Nonlinear correction on original data")

ax[0, 0].set_xlabel("x_mcp_1")
ax[0, 0].set_ylabel("y_mcp_1")

ax[0, 1].hist2d(
    x_mcp_lin,
    y_mcp_lin,
    range=hist_range,
    bins=n_bins,
    cmap="inferno",
    norm=mpl.colors.LogNorm(vmin=1),
    cmin=c_min,
)
# Add the colorbar
cbar = fig.colorbar(ax[0, 1].collections[0], ax=ax[0, 1])
cbar.set_label("Counts")

# Add the title to the figure
ax[0, 1].set_title("Linear correction on original data")

ax[0, 1].set_xlabel("x_mcp_lin")
ax[0, 1].set_ylabel("y_mcp_lin")

ax[1, 0].hist2d(
    x_mcp_nln,
    y_mcp_nln,
    range=hist_range,
    bins=n_bins,
    cmap="inferno",
    norm=mpl.colors.LogNorm(vmin=1),
    cmin=c_min,
)
# Add the colorbar
cbar = fig.colorbar(ax[1, 0].collections[0], ax=ax[1, 0])
cbar.set_label("Counts")

# Add the title to the figure
ax[1, 0].set_title("Nonlinear correction on linearly corrected data")

ax[1, 0].set_xlabel("x_mcp_nln")
ax[1, 0].set_ylabel("y_mcp_nln")

ax[1, 1].hist2d(
    x_mcp,
    y_mcp,
    range=hist_range,
    bins=n_bins,
    cmap="inferno",
    norm=mpl.colors.LogNorm(vmin=1),
    cmin=c_min,
)
# Add the colorbar
cbar = fig.colorbar(ax[1, 1].collections[0], ax=ax[1, 1])
cbar.set_label("Counts")

# Add the title to the figure
ax[1, 1].set_title("Original")

ax[1, 1].set_xlabel("x_mcp")
ax[1, 1].set_ylabel("y_mcp")

# Save the figure
fig.savefig("../figures/histograms.png", dpi=300, bbox_inches="tight")
