import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
fn = ".lexi_data/20241114_LEXIAngleData_20250302Landing_rad.csv"

df = pd.read_csv(fn)

# Convert the epoch from unix time to datetime
df["epoch_utc"] = pd.to_datetime(df["epoch_utc"], unit="s")
# Set the timezones to UTC
if df["epoch_utc"].dt.tz is None:
    print("No timezone")
    df["epoch_utc"] = df["epoch_utc"].dt.tz_localize("UTC")
    print(df["epoch_utc"].dt.tz)

df = df.set_index("epoch_utc", inplace=False)
"""

fn = ".lexi_data/20241114_LEXIAngleData_20250302Landing.csv"

df = pd.read_csv(fn)

df["epoch_utc"] = pd.to_datetime(df["epoch_utc"])

# Convert the epoch to unix time
df["epoch_utc"] = pd.to_datetime(df["epoch_utc"]).astype(int) / 10**9

# Set the timezones to UTC
# df["epoch_utc"] = df["epoch_utc"].dt.tz_localize("UTC")

key_list = ["epoch_utc", "ra_mag", "dec_mag"]

df = df[key_list]

df = df.sort_values(by="epoch_utc")

# Set the index to the epoch_utc column
df = df.set_index("epoch_utc", inplace=False)

# Add the ra_mag and dec_mag to radians from degrees
df["ra_rad"] = np.radians(df["ra_mag"])
df["dec_rad"] = np.radians(df["dec_mag"])

# Add another column for ra and dec in degrees
# Remove the ra_mag and dec_mag columns
df = df.drop(columns=["ra_mag", "dec_mag"])

# Save the dataframe to a csv file
df.to_csv(".lexi_data/20241114_LEXIAngleData_20250302Landing_rad.csv")

# Plot the ra and dec
plt.figure()
plt.plot(
    df.index,
    df["ra_rad"],
    label="ra_rad",
    linestyle="--",
    marker=None,
    alpha=0.5,
    linewidth=2,
)
ax = plt.twinx()
ax.plot(
    df.index,
    df["ra_deg"],
    label="ra_deg",
    color="red",
    linestyle=None,
    marker="x",
    markersize=5,
    alpha=1,
    linewidth=0,
)
plt.show()

plt.figure()
plt.plot(
    df.index,
    df["dec_rad"],
    label="dec_rad",
    linestyle="--",
    marker=None,
    alpha=0.5,
    linewidth=2,
)
ax = plt.twinx()
ax.plot(
    df.index,
    df["dec_deg"],
    label="dec_deg",
    color="red",
    linestyle=None,
    marker="x",
    markersize=5,
    alpha=1,
    linewidth=0,
)
plt.show()
"""
