import importlib
import glob

import numpy as np

# import pandas as pd
import lxi_read_binary_data as lrbd
import matplotlib.pyplot as plt

importlib.reload(lrbd)

# Activate the latex text rendering
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

folder_val = "/home/vetinari/Desktop/git/Lexi-Bu/lexi/data/from_PIT/20230816/"
multiple_files = True
t_start = "2024-05-23 22:26:20"
t_end = "2024-05-23 22:32:20"

# Get the file name
file_val_list = np.sort(glob.glob(folder_val + "*.dat"))

for file_val in file_val_list[0:1]:
    file_name, df = lrbd.read_binary_file(
        file_val=folder_val, t_start=t_start, t_end=t_end, multiple_files=multiple_files
    )

    # Plot the x_mcp_lin and x_mcp_nln data as histograms with logarithmic x and y axes (x_range = [1e-3,
    # 5e0])
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.hist(df["x_mcp_lin"], bins=100, range=[1e-3, 5e0], log=True, label="Linear")
    ax.hist(
        df["x_mcp_nln"],
        bins=100,
        range=[1e-3, 5e0],
        log=True,
        label="Non-linear",
        alpha=0.3,
    )
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("Counts")
    ax.legend(loc="upper right")
    ax.grid(True)
    ax.set_title("Histogram of x data")
    ax.set_xlim([1e-3, 5e0])
    # ax.set_ylim([1e0, 1e5])

    ax.set_xscale("log")
    ax.set_yscale("log")

    file_date = file_val.split("/")[-1].split("_")[-2]
    # Save the figure
    fig.savefig(
        f"../figures/lexi_data_product_test_{file_date}.png",
        dpi=300,
        bbox_inches="tight",
    )
