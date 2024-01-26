# Define class called LEXI with a list of different functions that can be called
import numpy as np
import pandas as pd
import urllib.request
from pathlib import Path
from spacepy import pycdf
import matplotlib as mpl
import matplotlib.pyplot as plt

import warnings


# TODO: Rewrite all docstrings

class LEXI:

    def __init__(self, input_params):
        """
        TODO all those comments below should probably just go in a nice docstring...
        """

        # ==============================================================================
        #                              Constants
        # ==============================================================================

        # LEXI field of view in degrees
        self.LEXI_FOV = 9.1

        # Link to the CDAweb website, from which ephemeris data are pulled
        self.CDA_LINK = "https://cdaweb.gsfc.nasa.gov/pub/data/lexi/ephemeris/"

        # ==============================================================================
        #                      User-defined input parameters
        # ==============================================================================

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
        Gets spacecraft ephemeris data for the given t_range by downloading the appropriate file(s)
        from the NASA CDAweb website.

        Returns a dataframe containing all ephemeris data (time, look direction RA, look direction DEC,
        roll angle, and potentially other columns), sliced to t_range and interpolated to t_step using
        interp_method.
        """
        # TODO: REMOVE ME once we start using real ephemeris data
        df = pd.read_csv("data/sample_lexi_pointing_ephem_edited.csv")
        df.index = pd.DatetimeIndex(df.epoch_utc)
        dfslice = df[self.t_range[0]:self.t_range[1]]
        dfresamp = dfslice.resample(pd.Timedelta(self.t_step, unit='s'))
        dfinterp = dfresamp.interpolate(method=self.interp_method)
        return dfinterp
        # (end of chunk that must be removed once we start using real ephemeris data)


        # Get the year, month, and day of the start and stop times
        start_time = self.t_range[0]
        stop_time = self.t_range[1]

        start_year = start_time.year
        start_month = start_time.month
        start_day = start_time.day

        stop_year = stop_time.year
        stop_month = stop_time.month
        stop_day = stop_time.day

        # Given that ephemeris files are named in the the format of lexi_ephm_YYYYMMDD_v01.cdf, get a
        # list of all the files that are within the time range of interest
        file_list = []
        for year in range(start_year, stop_year + 1):
            for month in range(start_month, stop_month + 1):
                for day in range(start_day, stop_day + 1):
                    # Create a string for the date in the format of YYYYMMDD
                    date_string = str(year) + str(month).zfill(2) + str(day).zfill(2)

                    # Create a string for the filename
                    filename = "lexi_ephm_" + date_string + "_v01.cdf"

                    # Create a string for the full link to the file
                    link = self.CDA_LINK + filename

                    # Try to open the link, if it doesn't exist then skip to the next date
                    try:
                        urllib.request.urlopen(link)
                    except urllib.error.HTTPError:
                        # Print in that the file doesn't exist or is unavailable for download from the CDAweb website
                        warnings.warn(
                            f"Following file is unavailable for downloading or doesn't exist. Skipping the file: \033[93m {filename}\033[0m"
                        )
                        continue

                    # If the link exists, then check if the date is within the time range of interest
                    # If it is, then add it to the list of files to download
                    if (
                        (year == start_year)
                        and (month == start_month)
                        and (day < start_day)
                    ):
                        continue
                    elif (year == stop_year) and (month == stop_month) and (day > stop_day):
                        continue
                    else:
                        file_list.append(filename)

        # Download the files in the file list to the data/ephemeris directory
        data_dir = Path(__file__).resolve().parent.parent / "data/ephemeris"
        # If the data directory doesn't exist, then create it
        Path(data_dir).mkdir(parents=True, exist_ok=True)

        # Download the files in the file list to the data/ephemeris directory
        for file in file_list:
            urllib.request.urlretrieve(self.CDA_LINK + file, data_dir / file)

        # Read the files into a single dataframe
        df_list = []
        for file in file_list:
            eph_data = pycdf.CDF(file)
            # Save the data to a dataframe
            df = pd.DataFrame()
            df["epoch_utc"] = eph_data["Epoch"]
            df["ra"] = eph_data["RA"]
            df["dec"] = eph_data["DEC"]
            df["roll"] = eph_data["ROLL"]

            # Set the index to be the epoch_utc column
            df = df.set_index("epoch_utc", inplace=False)
            # Set the timezone to UTC
            df = df.tz_localize("UTC")
            # Append the dataframe to the list of dataframes
            df_list.append(df)

        # Concatenate the list of dataframes into a single dataframe
        df = pd.concat(df_list)

        # Sort the dataframe by the index
        df = df.sort_index()

        # Remove any duplicate rows
        df = df[~df.index.duplicated(keep="first")]

        # Remove any rows that have NaN values
        df = df.dropna()

        # Slice, resample, interpolate
        dfslice = df[self.t_range[0]:self.t_range[1]]
        dfresamp = dfslice.resample(pd.Timedelta(self.t_step, unit='s'))
        dfinterp = dfresamp.interpolate(method=self.interp_method)

        return dfinterp


    def vignette(self, d):
        """
        Function to calculate the vignetting factor for a given distance from boresight

        Parameters
        ----------
        d : float
            Distance from boresight in degrees

        Returns
        -------
        f : float
            Vignetting factor
        """

        # Set the vignetting factor
        # f = 1.0 - 0.5 * (d / (LEXI_FOV * 0.5)) ** 2
        f = 1

        return f


    def get_exposure_maps(self, save_maps=False):
        """
        Returns an array of exposure maps, made according to the ephemeris data and the
        specified time/integration/resolution parameters.
        Shape: num-images.ra-pixels.dec-pixels, where num-images depends on t_range and
        t_integrate, ra-pixels depends on ra_range and ra_res, and dec-pixels depends on
        dec_range and dec_res.
        """
        try:
            # Read the exposure map from a pickle file
            # TODO: Must match filename to the save_maps step; and the filename should include ALL the params
            exposure = np.load(
                f"../data/exposure_map_rares_{self.ra_res}_decres_{self.dec_res}_tstep_{self.t_step}.npy"
            )
            print("Exposure map loaded from file \n")
        except FileNotFoundError:
            print("Exposure map not found, computing now. This may take a while \n")

            spc_df = self.get_spc_prams()

            # TODO: REMOVE ME once we start using real ephemeris data
            # The sample ephemeris data uses column names "mp_ra" and "mp_dec" for look direction;
            # in the final lexi ephemeris files on CDAweb, this will be called just "ra" and "dec".
            # Therefore...
            spc_df['ra'] = spc_df.mp_ra
            spc_df['dec'] = spc_df.mp_dec
            # (end of chunk that must be removed once we start using real ephemeris data)

            # Set up coordinate grid
            ra_grid = np.arange(self.ra_range[0], self.ra_range[1], self.ra_res)
            dec_grid = np.arange(self.dec_range[0], self.dec_range[1], self.dec_res)
            ra_grid_arr = np.tile(ra_grid, (len(dec_grid), 1)).transpose()
            dec_grid_arr = np.tile(dec_grid, (len(ra_grid), 1))

            # Slice to relevant time range; make groups of rows spanning t_integration
            integ_groups = spc_df[self.t_range[0]:self.t_range[1]].resample(pd.Timedelta(self.t_integrate, unit='s'))

            # Make as many empty exposure maps as there are integration groups
            exposure_maps = np.zeros((len(integ_groups), len(ra_grid), len(dec_grid)))

            # TODO: Can we figure out a way to do this not in a loop??? Cannot be vectorized...
            # Loop through each pointing step and add the exposure to the map
            for (map_idx, (_,group)) in enumerate(integ_groups):
                for row in group.itertuples():
                  # Get distance in degrees to the pointing step
                  # Wrap-proofing: First make everything [0,360), then +-360 on second operand
                  ra_diff  = np.minimum(abs((ra_grid_arr%360)-(row.ra%360))
                                       ,abs((ra_grid_arr%360)-(row.ra%360-360))
                                       ,abs((ra_grid_arr%360)-(row.ra%360+360)))
                  dec_diff = np.minimum(abs((dec_grid_arr%360)-(row.dec%360))
                                       ,abs((dec_grid_arr%360)-(row.dec%360-360))
                                       ,abs((dec_grid_arr%360)-(row.dec%360+360)))
                  r = np.sqrt(ra_diff ** 2 + dec_diff ** 2)
                  # Make an exposure delta for this span
                  exposure_delt = np.where(
                      (r < self.LEXI_FOV * 0.5), self.vignette(r) * self.t_step, 0
                  )
                  exposure_maps[map_idx] += exposure_delt  # Add the delta to the full map
                print(
                    f"Computing exposure map ==> \x1b[1;32;255m {np.round(map_idx/len(integ_groups)*100, 6)}\x1b[0m % complete",
                    end="\r",
                )

            # TODO: see above re filename and matching
            if save_maps:
                # Save the exposure map array to a pickle file
                np.save(
                    f"exposure_map_rares_{self.ra_res}_decres_{self.dec_res}_tstep_{self.t_step}_t0_{self.t_range[0]}_tf_{self.t_range[1]}.npy",
                    exposure_maps,
                )

        return exposure_maps


    def get_sky_background(self):
        """
        Returns an array of ROSAT sky background images, corrected for LEXI exposure time.
        Shape: num-images.ra-pixels.dec-pixels, where num-images depends on t_range and
        t_integrate, ra-pixels depends on ra_range and ra_res, and dec-pixels depends on
        dec_range and dec_res.
        """
        # Get exposure maps
        exposure_maps = self.get_exposure_maps(
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


    def array_to_image(
        self,
        input_array: np.ndarray,
        x_range: list = None,
        y_range: list = None,
        v_min: float = None,
        v_max: float = None,
        cmap: str = "viridis",
        norm: mpl.colors.Normalize = None,
        norm_type: str = "linear",
        aspect: str = "auto",
        figure_title: str = None,
        show_colorbar: bool = True,
        cbar_label: str = None,
        cbar_orientation: str = "vertical",
        show_axes: bool = True,
        display: bool = False,
        figure_size: tuple = (10, 10),
        figure_format: str = "png",
        figure_font_size: float = 12,
        save: bool = False,
        save_path: str = None,
        save_name: str = None,
        dpi: int = 300,
        dark_mode: bool = False,
    ):
        """
        Convert a 2D array to an image.

        Parameters
        ----------
        input_array : np.ndarray
            2D array to convert to an image.
        x_range : list, optional
            Range of the x-axis.  Default is None.
        y_range : list, optional
            Range of the y-axis.  Default is None.
        v_min : float, optional
            Minimum value of the colorbar.  If None, then the minimum value of the input array is used.
            Default is None.
        v_max : float, optional
            Maximum value of the colorbar.  If None, then the maximum value of the input array is used.
            Default is None.
        cmap : str, optional
            Colormap to use.  Default is 'viridis'.
        norm : mpl.colors.Normalize, optional
            Normalization to use for the colorbar colors.  Default is None.
        norm_type : str, optional
            Normalization type to use.  Options are 'linear' or 'log'.  Default is 'linear'.
        aspect : str, optional
            Aspect ratio to use.  Default is 'auto'.
        figure_title : str, optional
            Title of the figure.  Default is None.
        show_colorbar : bool, optional
            If True, then show the colorbar.  Default is True.
        cbar_label : str, optional
            Label of the colorbar.  Default is None.
        cbar_orientation : str, optional
            Orientation of the colorbar.  Options are 'vertical' or 'horizontal'.  Default is 'vertical'.
        show_axes : bool, optional
            If True, then show the axes.  Default is True.
        display : bool, optional
            If True, then display the figure.  Default is False.
        figure_size : tuple, optional
            Size of the figure.  Default is (10, 10).
        figure_format : str, optional
            Format of the figure.  Default is 'png'.
        figure_font_size : float, optional
            Font size of the figure.  Default is 12.
        save : bool, optional
            If True, then save the figure.  Default is False.
        save_path : str, optional
            Path to save the figure to.  Default is None.
        save_name : str, optional
            Name of the figure to save.  Default is None.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        ax : matplotlib.axes._subplots.AxesSubplot
            Axes object.
        """
        # Try to use latex rendering
        try:
            plt.rc("text", usetex=True)
            plt.rc("font", family="serif")
            plt.rc("font", size=figure_font_size)
        except Exception:
            pass

        # Check whether input_array is a 2D array
        if len(input_array.shape) != 2:
            raise ValueError("input_array must be a 2D array")

        # Check whether x_range is a list
        if x_range is not None:
            if not isinstance(x_range, list):
                raise ValueError("x_range must be a list")
            if len(x_range) != 2:
                raise ValueError("x_range must be a list of length 2")
        else:
            x_range = [self.ra_range[0], self.ra_range[1]]

        # Check whether y_range is a list
        if y_range is not None:
            if not isinstance(y_range, list):
                raise ValueError("y_range must be a list")
            if len(y_range) != 2:
                raise ValueError("y_range must be a list of length 2")
        else:
            y_range = [self.dec_range[0], self.dec_range[1]]

        # Check whether input_dict is a dictionary

        if dark_mode:
            plt.style.use("dark_background")
            facecolor = "k"
            edgecolor = "w"
        else:
            plt.style.use("default")
            facecolor = "w"
            edgecolor = "k"

        if v_min is None and v_max is None:
            if norm_type == "linear":
                v_min = 0.9 * np.nanmin(input_array)
                v_max = 1.1 * np.nanmax(input_array)
                norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
            elif norm_type == "log":
                v_min = np.nanmin(input_array)
                if v_min <= 0:
                    v_min = 1e-10
                v_max = np.nanmax(input_array)
                norm = mpl.colors.LogNorm(vmin=v_min, vmax=v_max)
        elif v_min is not None and v_max is not None:
            if norm_type == "linear":
                norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
            elif norm_type == "log":
                if v_min <= 0:
                    v_min = 1e-10
                norm = mpl.colors.LogNorm(vmin=v_min, vmax=v_max)
        else:
            raise ValueError(
                "Either both v_min and v_max must be specified or neither can be specified"
            )

        # Create the figure
        fig, ax = plt.subplots(
            figsize=figure_size, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor
        )

        # Plot the image
        im = ax.imshow(
            np.transpose(input_array),
            cmap=cmap,
            norm=norm,
            extent=[
                x_range[0],
                x_range[1],
                y_range[0],
                y_range[1],
            ],
            origin="lower",
            aspect=aspect,
        )

        # Set the tick label size
        ax.tick_params(labelsize=0.8 * figure_font_size)

        if show_colorbar:
            if cbar_label is None:
                cbar_label = "Value"
            if cbar_orientation == "vertical":
                cax = fig.add_axes(
                    [
                        ax.get_position().x1 + 0.01,
                        ax.get_position().y0,
                        0.02,
                        ax.get_position().height,
                    ]
                )
            elif cbar_orientation == "horizontal":
                cax = fig.add_axes(
                    [
                        ax.get_position().x0,
                        ax.get_position().y1 + 0.01,
                        ax.get_position().width,
                        0.02,
                    ]
                )
            ax.figure.colorbar(
                im,
                cax=cax,
                orientation=cbar_orientation,
                label=cbar_label,
                pad=0.01,
            )
            # Set the colorbar tick label size
            cax.tick_params(labelsize=0.6 * figure_font_size)
            # Set the colorbar label size
            cax.yaxis.label.set_size(0.9 * figure_font_size)

            # If the colorbar is horizontal, then set the location of the colorbar label and the tick
            # labels to be above the colorbar
            if cbar_orientation == "horizontal":
                cax.xaxis.set_ticks_position("top")
                cax.xaxis.set_label_position("top")
                cax.xaxis.tick_top()
            if cbar_orientation == "vertical":
                cax.yaxis.set_ticks_position("right")
                cax.yaxis.set_label_position("right")
                cax.yaxis.tick_right()
        if not show_axes:
            ax.axis("off")
        else:
            ax.set_xlabel("RA [$^\\circ$]", labelpad=0, fontsize=figure_font_size)
            ax.set_ylabel("DEC [$^\\circ$]", labelpad=0, fontsize=figure_font_size)
            ax.set_title(figure_title, fontsize=1.2 * figure_font_size)

        if save:
            if save_path is None:
                save_path = Path(__file__).resolve().parent.parent / "figures"
            if not save_path.exists():
                save_path.mkdir(parents=True, exist_ok=True)
            if save_name is None:
                save_name = "array_to_image"

            save_name = save_name + "." + figure_format
            plt.savefig(
                save_path / save_name, format=figure_format, dpi=dpi, bbox_inches="tight"
            )
            print(f"Saved figure to {save_path / save_name}")

        if display:
            plt.show()

        return fig, ax
