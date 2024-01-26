import numpy as np
import matplotlib.pyplot as plt
import csv

# Set grid resolution for generating data
resolution = 100
ra_deg = np.linspace(0,360,resolution) # RA in degrees [0,360]
dec_deg = np.linspace(-90,90,resolution) # DEC in degrees [-90,90]
x_deg, y_deg = np.meshgrid(ra_deg,dec_deg)
x_rad = np.deg2rad(x_deg) # Convert to radians
y_rad = np.deg2rad(y_deg)
# Generating counts via cosine function
counts = abs(np.cos((x_rad+y_rad)/2*np.pi))

# Save counts to csv
with open('sample_xray_background.csv','w') as f:
    writer = csv.writer(f)
    writer.writerows(counts)

# PLOTTING
fig, ax = plt.subplots()
data = ax.contourf(x_deg,y_deg,counts,cmap='inferno')
ax.set_xlim([0,360])
ax.set_ylim([-90,90])
ax.set_xlabel('RA (deg)')
ax.set_ylabel('DEC (deg)')
fig.colorbar(data)
plt.show()

