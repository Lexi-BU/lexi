# Define class called LEXI with a list of different functions that can be called
import numpy as np
import pandas as pd
from spacepy import pycdf
from .lxi_exposure_map_fnc import exposure_map
import warnings


# TODO: Rewrite all docstrings

class LEXI:

    def __init__(self, input_params):
        """
        TODO all those comments below should probably just go in a nice docstring...
        """

        self.LEXI_FOV = 9.1 # LEXI field of view in degrees

        # Set up input parameters

        # ZLC: should probably clarify what "the dataframe" is supposed to refer to (final? expmaps? skybgs? other?)
        # ZLC: Also, should make sure these are actually respected, else remove them
        self.save_df = input_params.get("save_df", False) # If True, save the dataframe to a file
        self.filename = input_params.get("filename", "../data/LEXI_pointing_ephem_highres") # filename to save df to
        self.filetype = input_params.get("filetype", "pkl") # Filetype to save df to. Options: 'csv','pkl'

        # Interpolation method used when upsampling/resampling ephemeris data, ROSAT data
        self.interp_method = input_params.get("interp_method", "index")
        if self.interp_method not in ["linear", "index", "values", "pad"]:
            warnings.warn(f"Requested integration method '{self.interp_method}' not currently supported; "
                f"defaulting to 'index'. Currently supported interpolation methods include: "
                f"linear, index, values, pad. See pandas.DataFrame.interpolate documentation "
                f"for more information.")
            self.interp_method = "index"
        # Toggle background correction
        self.background_correction_on = input_params.get("background_correction_on", True)

        # Time range to consider. [start time, end time]
        # Times can be expressed in the following formats:
        #    1. A string in the format 'YYYY-MM-DDTHH:MM:SS' (e.g. '2022-01-01T00:00:00')
        #    2. A datetime object
        #    3. A float in the format of a UNIX timestamp (e.g. 1640995200.0)
        self.t_range = input_params.get("t_range")
        if type(self.t_range) not in [tuple, list]:
            raise TypeError(f"t_range should be a tuple or list; instead got {type(self.t_range)}")
        if len(self.t_range) != 2:
            raise ValueError(f"t_range should contain exactly 2 elements; instead got {len(self.t_range)}")
        try:
          # (the 'unit' arg is only used if input is of type int/float, so no type checking is needed here)
          self.t_range = (pd.Timestamp(self.t_range[0], unit='s'), pd.Timestamp(self.t_range[1], unit='s'))
        except ValueError as err:
            raise ValueError(f"Could not process the given t_range: {err}. Check that t_range is in "
                             f"one of the allowed formats.")

        # exposure map step time in seconds: Ephemeris data will be resampled and interpolated to this
        # time resolution; then, for each look-direction datum,
        # this many seconds are added to each in-FOV cell in the exposure map.
        # ZLC: What is a reasonable value for this? I think 0.1 is way too small?
        self.t_step = input_params.get("t_step", 0.1)
        # integration time in seconds for lexi histograms and exposure maps
        self.t_integrate = input_params.get("t_integrate", 60*10)


        # RA range to plot, in degrees. [start RA, end RA]
        self.ra_range = input_params.get("ra_range", [0.0, 360.0])
        # DEC range to plot, in degrees. [start DEC, end DEC]
        self.dec_range = input_params.get("dec_range", [-90.0, 90.0])
        # RA resolution to plot at, in degrees. Ideal value is 0.1 deg.
        self.ra_res = input_params.get("ra_res", 0.1)
        # DEC resolution to plot at, in degrees. Ideal value is 0.1 deg.
        self.dec_res = input_params.get("dec_res", 0.1)
        # Alternative to ra_res/dec_res: nbins defines the number of bins in the RA and DEC directions.
        # Either a scalar integer or [ra_nbins, dec_nbins].
        if input_params.get("nbins") is not None:
            nbins = input_params["nbins"]
            ra_nbins, dec_nbins = (nbins[0],nbins[1]) if type(nbins) in [tuple, list] else (nbins, nbins)
            if input_params.get("ra_res") or input_params.get("dec_res"):
                print(f"Both (ra_res and/or dec_res) and nbins were specified; ignoring res value and setting "
                      f"RA nbins: {ra_nbins}, DEC nbins: {dec_nbins}")
            self.ra_res = (self.ra_range[1] - self.ra_range[0]) / ra_nbins
            self.dec_res = (self.dec_range[1] - self.dec_range[0]) / dec_nbins


    def get_spc_prams(self):
        """
        Returns a dataframe containing all ephemeris data (time, look direction RA, look direction DEC,
        roll angle, and potentially other columns), sliced to t_range and interpolated to t_step using
        interp_method.
        """
        # Note: Current sample data does not include roll angle.
        # Roll angle = gimbal + lander + altitude

        # TODO: Need to decide on path to the actual spc params file...
        df = pd.read_csv("data/sample_lexi_pointing_ephem_edited.csv")
        # Make datetime index from epoch_utc
        df.index = pd.DatetimeIndex(df.epoch_utc)
        # Slice, resample, interpolate
        dfslice = df[self.t_range[0]:self.t_range[1]]
        dfresamp = dfslice.resample(pd.Timedelta(self.t_step, unit='s'))
        dfinterp = dfresamp.interpolate(method=self.interp_method)
        return dfinterp



    def get_sky_background(self):
        """
        Returns an array of ROSAT sky background images, corrected for LEXI exposure time.
        Shape: num-images.ra-pixels.dec-pixels, where num-images depends on t_range and
        t_integrate, ra-pixels depends on ra_range and ra_res, and dec-pixels depends on
        dec_range and dec_res.
        """
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

        # TODO: when using actual data, check that axes are correct (index/column to ra/dec)
        rosat_df.index = np.linspace(self.ra_range[0], self.ra_range[1], 100)
        rosat_df.columns = np.linspace(self.dec_range[0], self.dec_range[1], 100)
        # Reindex to include desired RA/DEC indices (but don't throw out old indices yet; need for interpolation)
        desired_ra_idx = np.arange(self.ra_range[0], self.ra_range[1], self.ra_res)
        desired_dec_idx = np.arange(self.dec_range[0], self.dec_range[1], self.dec_res)
        rosat_enlarged_idx = rosat_df.reindex(index=np.union1d(rosat_df.index, desired_ra_idx),
                                              columns=np.union1d(rosat_df.columns, desired_dec_idx))
        # Interpolate and then throw out the old indices to get correct dimensions
        rosat_interpolated = rosat_enlarged_idx.interpolate(method=self.interp_method).interpolate(method=self.interp_method,axis=1)
        rosat_resampled = rosat_interpolated.reindex(index=desired_ra_idx,columns=desired_dec_idx)

        # Multiply each exposure map (seconds) with the ROSAT background (counts/sec)
        sky_backgrounds = [e * rosat_resampled for e in exposure_maps]

        return sky_backgrounds

    def get_background_corrected_image(self):
        """
        Returns an array of LEXI science histograms. Shape: num-images.ra-pixels.dec-pixels,
        where num-images depends on t_range and t_integrate, ra-pixels depends on ra_range and
        ra_res, and dec-pixels depends on dec_range and dec_res.
        """
        # TODO: Get actual timeseries data
        # With ra/dec resolution of 3, and no bg correction, this will draw a smiley face;
        # bg correction lops off the left side of the smile
        photons = pd.DataFrame({"ra_J2000_deg":[355,355,335,333,330,333,335],
                                "dec_J2000_deg":[-15,-5,-15,-14,-10, -6, -5]},
                                index=[pd.Timestamp("Jul 08 2024 15:01:00.000000000"),
                                       pd.Timestamp("Jul 08 2024 15:02:00.000000000"),
                                       pd.Timestamp("Jul 08 2024 15:03:00.000000000"),
                                       pd.Timestamp("Jul 08 2024 15:04:00.000000000"),
                                       pd.Timestamp("Jul 08 2024 15:05:00.000000000"),
                                       pd.Timestamp("Jul 08 2024 15:06:00.000000000"),
                                       pd.Timestamp("Jul 08 2024 15:07:00.000000000"),
                                       ])

        # For now try reading CDF just from here
        photons_cdf = pycdf.CDF("data/from_PIT/20230816/processed_data/sci/level_1c/cdf/1.0.0/lexi_payload_1716500621_21694_level_1c_1.0.0.cdf")
        photons_data = photons_cdf.copy()
        photons_cdf.close()
        photons = pd.DataFrame({key:photons_data[key] for key in photons_data.keys()})
        # Make datetime index from epoch_utc
        photons.index = pd.DatetimeIndex(photons.Epoch)
        print(f"Extrema: RA min {photons.ra_J2000_deg.min()}, RA max {photons.ra_J2000_deg.max()}, "
              f"DEC min {photons.dec_J2000_deg.min()}, DEC max {photons.dec_J2000_deg.max()}")

        # Set up coordinate grid for lexi histograms
        ra_grid = np.arange(self.ra_range[0], self.ra_range[1], self.ra_res)
        dec_grid = np.arange(self.dec_range[0], self.dec_range[1], self.dec_res)

        # Slice to relevant time range; make groups of rows spanning t_integration
        integ_groups = photons[self.t_range[0]:self.t_range[1]].resample(pd.Timedelta(self.t_integrate, unit='s'))

        # Make as many empty lexi histograms as there are integration groups
        histograms = np.zeros((len(integ_groups), len(ra_grid), len(dec_grid)))

        for (hist_idx, (_,group)) in enumerate(integ_groups):
            # Loop through each photon strike and add it to the map
            for row in group.itertuples():
                try:
                    ra_idx = np.nanargmin(np.where(ra_grid%360 >= row.ra_J2000_deg%360, 1, np.nan))
                    dec_idx = np.nanargmin(np.where(dec_grid%360 >= row.dec_J2000_deg%360, 1, np.nan))
                    histograms[hist_idx][ra_idx][dec_idx] += 1
                except ValueError:
                    pass # photon was out of bounds on one or both axes

        # Early exit if no bg correction
        if not self.background_correction_on:
            return histograms

        # Else make background corrected images
        sky_backgrounds = self.get_sky_background()
        bgcorr_histograms = np.maximum(histograms - sky_backgrounds, 0)
        return bgcorr_histograms

        # TODO make FITS files
