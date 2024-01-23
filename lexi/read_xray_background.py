import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read csv
df = pd.read_csv('sample_xray_background.csv',header=None)
print(df.shape)

# Now remap counts onto 2D grid of RA and DEC
# RA in degrees [0,360]
# DEC in degrees [-90,90]
resolution = len(df)
ra_deg = np.linspace(0,360,resolution) # RA in degrees [0,360]
dec_deg = np.linspace(-90,90,resolution) # DEC in degrees [-90,90]
x_deg, y_deg = np.meshgrid(ra_deg,dec_deg)
x_rad = np.deg2rad(x_deg) # Convert to radians
y_rad = np.deg2rad(y_deg)
counts = df.values

# PLOTTING
fig, ax = plt.subplots()
data = ax.contourf(x_deg,y_deg,counts,cmap='inferno')
ax.set_xlim([0,360])
ax.set_ylim([-90,90])
ax.set_xlabel('RA (deg)')
ax.set_ylabel('DEC (deg)')
ax.set_title('Sample X-ray Background')
fig.colorbar(data)
plt.show()

