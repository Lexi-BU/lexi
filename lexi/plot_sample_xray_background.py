import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from geopack import geopack

# USER DEFINED PARAMETERS
el = 20
az = 45
roll = 0

# Read csv
df = pd.read_csv('sample_xray_background.csv',header=None)
print(df.shape)
resolution = len(df)

# Now remap counts onto 2D grid of RA and DEC
ra_deg = np.linspace(0,360,resolution) # RA in degrees [0,360]
dec_deg = np.linspace(-90,90,resolution) # DEC in degrees [-90,90]
x_deg, y_deg = np.meshgrid(ra_deg,dec_deg)
x_rad = np.deg2rad(x_deg) # Convert to radians
y_rad = np.deg2rad(y_deg)
# Map counts onto this grid
counts = df.values

# PLOTTING
fig1, ax1 = plt.subplots()
data = ax1.contourf(x_deg,y_deg,counts,cmap='inferno')
ax1.set_xlim([0,360])
ax1.set_ylim([-90,90])
ax1.set_xlabel('RA (deg)')
ax1.set_ylabel('DEC (deg)')
fig1.colorbar(data)

# Add B-field lines
dt = datetime(1970, 3, 1).timestamp() # pick a time near equinox - could pick any time later
ps = geopack.recalc(dt)
fig2 = plt.figure()
ax2 = fig2.add_subplot(projection = '3d')

theta0_field = [55,80,130,150] # deg
for i in range(0,4):
    zs = 1.5*np.sin(np.deg2rad(theta0_field[i]))
    xs = 1.5*np.cos(np.deg2rad(theta0_field[i]))
    ys = 0
    x,y,z,xx,yy,zz = geopack.trace(xs,ys,zs,dir=1,rlim=100,r0=0.99999,parmod=2,exname='t89',inname='igrf',maxloop=10000)
    ax2.plot(xx, zz, zs=0, zdir='y',color='red',zorder=1)

# Now rotate plot such that it lies in XZ plane
data = ax2.contourf(x_deg/10,counts/10,y_deg/10,zdir='y',cmap='inferno',alpha=0.7)
ax2.set_zlim([-90/10,90/10])
ax2.set_xlim([0,360/10])
ax2.set_ylim([0, 1e-6])
ax2.yaxis.set_major_formatter(plt.NullFormatter())
ax2.axes.get_xaxis().set_visible(False)
ax2.set_aspect('equal')
ax2.set_xlabel('RA (deg)')
ax2.set_zlabel('DEC (deg)')
# ax2.set_ylabel('DEC (deg)')
fig2.colorbar(data)
ax2.view_init(elev=el, azim=az, roll=roll)
# plt.axis('off')
plt.show()

