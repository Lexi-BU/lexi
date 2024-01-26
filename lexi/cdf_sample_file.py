from spacepy.pycdf import CDF

dat = CDF("~/Downloads/mms1_fpi_fast_sitl_20150801132440_v0.0.0.cdf")

# Print out the attributes of the CDF file
print(dat.attrs)

# Print out the variables in the CDF file
# print(dat)

# Print out the variable attributes
# print(dat["mms1_des_energy_fast"].attrs)
