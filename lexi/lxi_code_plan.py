# Define class called LEXI with a list of different functions that can be called
import numpy as np
import pandas as pd
from .lxi_exposure_map_fnc import exposure_map


# TODO: Rewrite all docstrings

class LEXI:

    def __init__(self, input_params):

        self.LEXI_FOV = 9.1 # LEXI field of view in degrees

        # Set up input parameters

        # ZLC: should probably clarify what "the dataframe" is supposed to refer to (final? expmaps? skybgs? other?)
        # ZLC: Also, should make sure these are actually respected, else remove them
        self.save_df = input_params.get("save_df", False) # If True, save the dataframe to a file
        self.filename = input_params.get("filename", "../data/LEXI_pointing_ephem_highres") # filename to save df to
        self.filetype = input_params.get("filetype", "pkl") # Filetype to save df to. Options: 'csv','pkl'

        self.interp_method = input_params.get("interp_method", "ffill") # Interpolation method
        self.background_correction_on = input_params.get("background_correction_on", True)

        # ZLC: should document what time formats are appropriate
        # ZLC: should check this is not None and is [a, b]; maybe change them to pd.Timestamps right here
        # and raise error if cannot
        self.t_range = input_params.get("t_range")
        # exposure map step time in seconds: Ephemeris data will be resampled and interpolated to this
        # time resolution; then, for each look-direction datum,
        # this many seconds are added to each in-FOV cell in the exposure map.
        # ZLC: What is a reasonable value for this? I think 0.01 is way too small?
        self.t_step = input_params.get("t_step", 0.01)
        # integration time in seconds for lexi histograms and exposure maps
        self.t_integrate = input_params.get("t_integrate", 60)

        self.ra_range = input_params.get("ra_range", [325.0, 365.0]) # RA range in degrees for plotted histogram
        self.dec_range = input_params.get("dec_range", [-21.0, 6.0]) # DEC range in degrees for plotted histogram
        self.ra_res = input_params.get("ra_res", 0.1) # RA res in degrees. Ideal value is 0.1 deg
        self.dec_res = input_params.get("dec_res", 0.1) # DEC res in degrees. Ideal value is 0.1 deg
        # TODO Add "nbins" option, decide which one "wins" if both supplied.
        # Do the conversion here; then all the other functions can just assume one or the other.


    # Define the first function called "get_spc_prams" that takes time as an argument and returns the
    # time, look direction, and roll angle of the lander
    def get_spc_prams(self):
        # Define the time, look direction (gamma) and the roll angle (phi) of the lander
        # Define the format of each of the inputs
        #    --time: [start time, end time]
        #    This will be a list of two elements, the first element is the start time and the second
        #    element is the end time. Each element of the list can be either of the following types:
        #    1. A string in the format 'YYYY-MM-DDTHH:MM:SS' (e.g. '2022-01-01T00:00:00')
        #    2. A datetime object
        #    3. A float in the format of a UNIX timestamp (e.g. 1640995200.0)

        #    --time_resolution: float
        #    This will be a float that defines the time resolution of the data in seconds

        #    --gamma: 1-d array of floats in the range of [0, 2*pi] where the length of the array
        #    will be defined by the number of data points in the time range as derived from reading
        #    the lander ephemeris file

        #    --alpha: 1-d array of floats in the range of [0, 2*pi] where the length of the array
        #    will be defined by the number of data points in the time range as derived from reading
        #    the lander ephemeris file

        #    --time_array: 1-d array of floats in the range of [time[0], time[1]] where the length of
        #    the array will be defined by the number of data points in the time range as derived from
        #    reading the lander ephemeris file

        # If a time resolution is given then the code will also interpolate the ephemeris data to the
        # given time resolution

        # Note: Current sample data does not include roll angle.
        # Roll angle = gimbal + lander + altitude

        # TODO: Need to decide on path to the actual spc params file...
        df = pd.read_csv("data/sample_lexi_pointing_ephem_edited.csv")
        # Make datetime index from epoch_utc
        df.index = pd.DatetimeIndex(df.epoch_utc)
        # Slice, resample, interpolate
        # TODO: get interpolation method from user dict instead of hardcoding 'ffill'
        return df[pd.Timestamp(self.t_range[0]):pd.Timestamp(self.t_range[1])].resample(pd.Timedelta(self.t_step, unit='s')).ffill()



    # Define a second function which takes the following list of arguments:
    #    --time
    #    --RA
    #    --DEC
    #    --binsize
    #    --nbins
    #    --integration_time
    # The function then computes the sky background and returns the sky background
    def get_sky_background(self):
        # Define the format of each of the inputs
        #    --time: [start time, end time] (see above for the format of the time input)
        #    --RA: [start RA, end RA]
        #    This will be a list of two elements, the first element is the start RA and the second
        #    element is the end RA. Each element of the list can be either of the following types:
        #    1. A string in the format 'HH:MM:SS' (e.g. '00:00:00')
        #    2. A float in the format of decimal hours (e.g. 0.0)

        #    --DEC: [start DEC, end DEC]
        #    This will be a list of two elements, the first element is the start DEC and the second
        #    element is the end DEC. Each element of the list can be either of the following types:
        #    1. A float in the format of decimal degrees (e.g. 0.0)

        #    --binsize: [RA binsize, DEC binsize] (optional)
        #    This will be a float that defines the size of the bins in degrees for DEC and in minutes
        #    for RA.

        #    --nbins: int or 1x2 array (optional)
        #    This will be an integer that defines the number of bins in the RA and DEC directions

        #    --integration_time: float
        #    This will be a float that defines the integration time of the data in seconds

        # The code will then compute the sky background for the given time, RA, DEC, binsize, nbins,
        # and integration time and by accessing the get_spc_prams function to get the time, look
        # direction (gamma) and the roll angle (phi) of the lander. The code will then return the
        # sky background

        # Get exposure maps
        exposure_maps = exposure_map(
                spc_df = self.get_spc_prams(),
                t_range = self.t_range,
                t_integrate = self.t_integrate,
                t_step = self.t_step,
                lexi_fov = self.LEXI_FOV,
                ra_range = self.ra_range,
                dec_range = self.dec_range,
                ra_res = self.ra_res,
                dec_res = self.dec_res,
                save_maps = False # TODO: check if save_df param was for only final dfs, or for these too
                )

        # Get ROSAT background
        # Ultimately someone is supposed to provide this file and we will have it saved somewhere static.
        # For now, this is Cadin's sample xray data:
        rosat_df = pd.read_csv('sample_xray_background.csv',header=None)
        # Slice to RA/DEC range, interpolate to RA/DEC res
        # For now just interpolate Cadin data:

        breakpoint()
        # TODO: when using actual data, check that axes are correct (index/column to ra/dec)
        rosat_df.index = np.linspace(self.ra_range[0], self.ra_range[1], 100)
        rosat_df.columns = np.linspace(self.dec_range[0], self.dec_range[1], 100)
        # TODO: Problemino:
        # If the new resolution is not a perfect multiple of the old resolution (the general case),
        # then the reindex will throw out most of the rows/columns...
        # You could upsample all the way to the LCM and downsample again, but with the right (wrong) numbers
        # that will probably land you with an unworkably enormous array here...
        # There's no way this is not a solved pandas problem -_-
        bigrosat = rosat_df.reindex(index=np.arange(self.ra_range[0], self.ra_range[1], self.ra_res),
                                    columns=np.arange(self.dec_range[0], self.dec_range[1], self.dec_res))#.interpolate().interpolate(axis=1)
        # TODO: In general (for ephemeris data as well as here) wouldn't we rather do linear interpolation than ffill???
        bigrosat = bigrosat.interpolate()
        bigrosat = bigrosat.interpolate(axis=1)

        # Sanity check this "multiply" step by literally making a rectangle:
        rectangle = np.full((400,270), 1)
        rectangle[0:100,0:70] = 0
        #return [rectangle]
        #return [e * rectangle for e in exposure_maps] # K well this works as expected...

        # What if I just look at bigrosat
        return [bigrosat]

        # Multiply each exposure map (seconds) with the ROSAT background (counts/sec)
        #sky_backgrounds = exposure_maps * bigrosat
        # Blagh, right now it won't do this because the dimensions... are correct but I guess are not labeled the same???
        # anyway, pandas can't tell that the dimensions are OK.
        # For now, doing it manually:
        # TODO fix
        sky_backgrounds = [e * bigrosat for e in exposure_maps]

        return sky_backgrounds

    # Define a third function which takes the following list of arguments:
    #    --time
    #    --RA
    #    --DEC
    #    --binsize
    #    --nbins
    #    --integration_time
    #    --background_correction
    # The function then makes the background corrected image from LEXI data and returns the
    # background corrected image
    def get_background_corrected_image(
        self, time, RA, DEC, binsize, nbins, integration_time, background_correction
    ):
        # Define the format of each of the inputs
        #    --time: [start time, end time] (see above for the format of the time input)
        #    --RA: [start RA, end RA] (see above for the format of the RA input)
        #    --DEC: [start DEC, end DEC] (see above for the format of the DEC input)
        #    --binsize: [RA binsize, DEC binsize] (optional) (see above for the format of the binsize
        #    input)
        #    --nbins: int or 1x2 array (optional) (see above for the format of the nbins input)
        #    --integration_time: float (see above for the format of the integration_time input)
        #    --background_correction: bool
        #    This will be a boolean that defines whether or not the background should be corrected
        #    for the image. If True, the background will be corrected.
        # Define the background corrected image
        if background_correction == True:
            background_corrected_image = 1
        background_corrected_image = 0
        return background_corrected_image
