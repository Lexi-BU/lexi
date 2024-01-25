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
multiple_files = False
t_start = "2024-05-23 22:26:20"
t_end = "2024-05-23 22:32:20"

# Get the file name
file_val_list = np.sort(glob.glob(folder_val + "*.dat"))

for file_val in file_val_list[:1]:
    file_name, df_sci, df_sci_l1b, df_sci_l1c, df_eph = lrbd.read_binary_file(
        file_val=file_val, t_start=t_start, t_end=t_end, multiple_files=multiple_files
    )
    # Make a time series plot of RA, dec
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    ax[0].scatter(
        df_sci_l1c.index,
        df_sci_l1c["ra_J2000_deg"],
        label="RA",
        s=0.1,
        c="r",
        marker=".",
        alpha=0.5,
    )
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("RA [deg]")
    ax[0].set_ylim(0, 360)
    ax[0].grid()

    ax[1].scatter(
        df_sci_l1c.index,
        df_sci_l1c["dec_J2000_deg"],
        label="dec",
        s=0.1,
        c="b",
        marker=".",
        alpha=0.5,
    )
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Dec [deg]")
    ax[1].set_ylim(-90, 90)
    ax[1].grid()

    file_date = file_name.split("/")[-1].split("_")[2]
    # Save the figure
    fig.savefig(f"../figures/RA_dec_{file_date}.png", dpi=300, bbox_inches="tight")
