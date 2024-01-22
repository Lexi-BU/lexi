# Input time of LEXI counts, x, y (detector), time_eph, RA_look, DEC_look, roll angle
# output time, RA, DEC of each photon
# BMW 22 Jan 2024

import numpy as np


def xy2radec(LEXI_x, LEXI_y, LEXI_time_unix, Ephem_RA_b_deg,Ephem_DEC_b_deg,Ephem_Roll_b_deg,Ephem_time_unix):
    time_lexi_uninx = LEXI_time_unix # time of LEXI counts in UNIX time in seconds
    time_eph_uninx = Ephem_time_unix # time of ephemeris in UNIX time in seconds
    RA_b_deg = Ephem_RA_b_deg # Ephemeris from lander RA of LEXI boresight in J2000 in degrees
    DEC_b_deg = Ephem_DEC_b_deg # Ephemeris from lander DEC of LEXI boresight in J2000
    roll_deg = Ephem_Roll_b_deg # Ephemeris from lander roll anfle of LEXI boresight in J2000.  THIS NEEDS TO BE CHECKED HOW THIS WILL BE DEFINED

    lexi_x_cm = LEXI_x # LEXI count position in cm
    lexi_y_cm = LEXI_y # LEXI count position in cm

    cm2deg = 4.55/(4.00*0.9375) # constant to convert to deg space [deg/cm]

    # LINE BELOW WILL NEED TO BE RE_WRITTEN.  INTERPOLATING WILL LIKELY NEED TO INTERPOLATE FROM EPHEM TO LEXI
    # interpolate the ra_b, dec_b, roll to match the time steps of the LEXI photon strikes
    RA_b_interp_deg = np.interp(time_lexi_uninx,time_eph_uninx,RA_b_deg)
    DEC_b_interp_deg = np.interp(time_lexi_uninx,time_eph_uninx,DEC_b_deg)
    roll_interp_deg = np.interp(time_lexi_uninx,time_eph_uninx,roll_deg)

    # # Convert photon detections to polar in detector coordinates
    r_det_cm = np.sqrt(lexi_x_cm**2+lexi_y_cm**2)
    theta_det_deg = np.arctan2(lexi_y_cm,lexi_x_cm)

    # rotate to J2000 coord frame and convert to deg
    r_J2000_deg = r_det_cm*cm2deg
    theta_J2000_deg = theta_det_deg - roll_interp_deg

    # convert back to cartesian
    x_J2000_deg = r_J2000_deg*np.cos(theta_J2000_deg)
    y_J2000_deg = r_J2000_deg*np.sin(theta_J2000_deg)

    # convert to RA/DEC using pointing of boresight
    RA_J2000_deg = x_J2000_deg + RA_b_interp_deg
    DEC_J2000_deg = y_J2000_deg + DEC_b_interp_deg

    return RA_J2000_deg, DEC_J2000_deg 