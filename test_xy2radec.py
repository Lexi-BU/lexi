# crib to test functionality of xy2radec function
# BMW Jan 22, 2024

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from LEXI_fn import xy2radec

roll_const = 0. # fixed sample angle to use

# ------- Read in sample made up ephemeris file and manipulate ------
path = '/Users/bwalsh/Documents/Research/LEXI/Software/Coding_Blitz/data/'
ephem = pd.read_csv(path+'SAMPLE_LEXI_pointing_ephem.csv',sep=',')
time_eph = pd.to_datetime((ephem['[Epoch (UTC)]']).to_numpy()) # read the time data
RA_b_deg = ephem['[Magnetopause Track Ra (deg)]'].to_numpy() # ra of boresight
DEC_b_deg = ephem['[Magnetopause Track Dec (deg)]'].to_numpy() # dec of boresight
time_eph_uninx = (time_eph - pd.Timestamp("1970-01-01"))/pd.Timedelta('1s') # calculate unix datetime

# make up a fake fixed roll angle between detector and J2000 coordinates
roll_deg = np.zeros(np.size(RA_b_deg))
roll_deg[:] = roll_const # [deg] roll angle

# ------- Read in sample x-ray data  ------
lexi_data = pd.read_csv(path+'lexi_payload_1716501822_1605_1716502122_14724_sci_output.csv',sep=',')

# convert time and pointing to np array
time_lexi = pd.to_datetime((lexi_data['Date']).to_numpy())
lexi_x_cm = lexi_data['x_mcp_lin'].to_numpy() # x detector coorinate [cm]
lexi_y_cm = lexi_data['y_mcp_lin'].to_numpy() # y detector coordinate [cm]
time_lexi_uninx = (time_lexi - pd.Timestamp("1970-01-01", tz='UTC'))/pd.Timedelta('1s') # calculate unix datetime

# adjust to stretch over many days to match ephemeris data
time_lexi_uninx = (time_lexi_uninx - min(time_lexi_uninx))*1500.+ min(time_eph_uninx)

# Run function to get degrees in RA/DEC of each photon hit
RA_J2000_deg, DEC_J2000_deg = xy2radec(lexi_x_cm, lexi_y_cm, time_lexi_uninx, RA_b_deg,DEC_b_deg,roll_deg,time_eph_uninx)

# ======= Plotting diagnostics to be uncomments below ==============
# plot x,y MCP output to check
# fig=plt.figure()
# ax = plt.axes()
# xlim = -5, 5
# ylim = -5,5
# plt.xlabel('X [cm]')
# plt.ylabel('Y [cm]')
# plt.title('LEXI, hits')
# hb = ax.hexbin(lexi_x_cm, lexi_y_cm, gridsize=300, cmap='inferno')
# ax.set(xlim=xlim, ylim=ylim)
# # ax.set_title("Hexagon binning")
# cb = fig.colorbar(hb, ax=ax, label='counts')
# plt.show()


# # plot RA, DEC output to check
# fig=plt.figure()
# ax = plt.axes()
# xlim = DEC_J2000_deg.min(), DEC_J2000_deg.max()
# ylim = RA_J2000_deg.min(), RA_J2000_deg.max()
# plt.xlabel('RA [Deg]')
# plt.ylabel('DEC [Deg]')
# plt.title('LEXI, hits')
# hb = ax.hexbin(DEC_J2000_deg, RA_J2000_deg, gridsize=200, cmap='inferno')
# ax.set(xlim=xlim, ylim=ylim)
# # ax.set_title("Hexagon binning")
# cb = fig.colorbar(hb, ax=ax, label='counts')
# plt.show()


# # plot lineplot output to check
fig, ax = plt.subplots(2, 4, figsize=(9, 5))

ax[0,0].scatter(time_lexi_uninx, RA_J2000_deg,s=0.1)
ax[0,1].scatter(time_lexi_uninx, DEC_J2000_deg,s=0.1)
ax[0,0].set_ylabel('RA [DEG]')
ax[0,0].set_xlabel('Time [s]')
ax[0,1].set_ylabel('DEC [DEG]')
ax[0,1].set_xlabel('Time [s]')

ax[0,2].scatter(DEC_J2000_deg,RA_J2000_deg,s=0.1)
ax[0,2].set_ylabel('RA [DEG]')
ax[0,2].set_xlabel('DEC [DEG]')
ax[0,2].set(xlim=[-25,65], ylim=[-10,365])

ax[0,3].scatter(DEC_b_deg,RA_b_deg,s=0.1)
ax[0,3].set_ylabel('RA [DEG]')
ax[0,3].set_xlabel('DEC [DEG]')
ax[0,3].set(xlim=[-25,65], ylim=[-10,365])


xlim = DEC_J2000_deg.min(), DEC_J2000_deg.max()
ylim = RA_J2000_deg.min(), RA_J2000_deg.max()
ax[1,0].set_ylabel('RA [DEG]')
ax[1,0].set_xlabel('DEC [DEG]')
hb = ax[1,0].hexbin(DEC_J2000_deg, RA_J2000_deg, gridsize=100, cmap='inferno')
ax[1,0].set(xlim=xlim, ylim=ylim)
# ax.set_title("Hexagon binning")
cb = fig.colorbar(hb, ax=ax[1,0], label='counts')

plt.show()
