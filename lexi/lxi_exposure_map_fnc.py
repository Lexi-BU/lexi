import time

import bokeh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.models import HoverTool
from bokeh.plotting import figure
from matplotlib import cm


def vignette(d):
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


# Function to interpolate the pointing data to a given resolution
def interpolate_pointing(
    df,
    t_res="100L",
    method="ffill",
    save_df=False,
    filename="../data/LEXI_pointing_ephem_highres",
    filetype="pkl",
):
    """
    Function to interpolate the pointing data to a given resolution

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with pointing data
    t_res : string
        Resolution to interpolate to.  Default is 100 ms
    method : string
        Interpolation method.  Default is forward fill
    save_df : bool
        If True, save the dataframe to a file.  Default is False
    filename : string
        Filename to save the dataframe to.  Default is '../data/LEXI_pointing_ephem_highres'
    filetype : string
        Filetype to save the dataframe to.  Default is 'pkl'. Options are 'csv' or 'pkl'

    Returns
    -------
    df : pandas dataframe
        Dataframe with interpolated pointing data

    Raises
    ------
    TypeError
        If df is not a pandas dataframe
    ValueError
        If df is empty
    TypeError
        If t_res is not a string
    TypeError
        If index is not a datetime object
    ValueError
        If method is not 'ffill' or 'bfill'
    TypeError
        If filename is not a string
    """

    # Check if df is a pandas dataframe and is not empty
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")
    if df.empty:
        raise ValueError("df cannot be empty")
    if not isinstance(t_res, str):
        raise TypeError(
            "t_res must be a string. Options are '100L' or '1S' or any other pandas"
            "resample string"
        )

    # Check if the index is a datetime object
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("index must be a datetime object")

    if method not in ["ffill", "bfill"]:
        raise ValueError("method must be 'ffill' or 'bfill'")

    # Interpolate the even time steps to get 100 ms resolution
    # Print that the interpolation is happening
    print(
        f"Interpolating the pointing data to {t_res} resolution. This may take a while\n"
    )
    df_intrp = df.resample(t_res).interpolate(method=method)

    if save_df:
        print(f"Trying to save dataframe to {filename}_tres_{t_res}.{filetype}\n")
        # Check if filename is given, if not then print saying that default will be used
        if filename is None:
            print(f"No filename given, default will be used: {filename}")
        # Check if filename is a string
        if not isinstance(filename, str):
            raise TypeError("filename must be a string")

        # Save the dataframe to proper file type
        if filetype == "csv":
            df_intrp.to_csv(f"{filename}_tres_{t_res}.{filetype}", index=False)
        elif filetype == "pkl":
            df_intrp.to_pickle(f"{filename}_tres_{t_res}.{filetype}")
        else:
            raise ValueError("filetype must be 'csv' or 'pkl'")
        print(f"Dataframe saved to {filename}_tres_{t_res}.{filetype}\n")

    return df_intrp


def read_create_df(input_dict=None, filename=None):
    """
    Function to read in the high res pointing file, if it doesn't exist then make it

    Parameters
    ----------
    input_dict : dict
        Dictionary of inputs.  Default is None
    filename : string
        Filename to read in.  Default is None

    Returns
    -------
    df : pandas dataframe
        Dataframe with interpolated pointing data

    Raises
    ------
    TypeError
        If input_dict is not a dictionary
    TypeError
        If filename is not a string
    """

    # Check if input_dict is a dictionary
    if not isinstance(input_dict, dict):
        raise TypeError("input_dict must be a dictionary")

    # Check if filename is a string
    if not isinstance(filename, str):
        # Check if filename is None
        if filename is None:
            filename = (
                f"../data/LEXI_pointing_ephem_highres_tres_{input_dict['t_res']}.pkl"
            )
        else:
            raise TypeError("filename must be a string")

    try:
        df = pd.read_pickle(filename)
        print("\nHigh res pointing file loaded from file \n")
    except FileNotFoundError:
        print(
            "\nHigh res pointing file not found, computing now. This may take a while. \n"
        )
        ephem = pd.read_csv("../data/sample_lexi_pointing_ephem_edited.csv", sep=",")

        # Set 'epoch_utc' column to datetime object, and set time to UTC
        ephem["epoch_utc"] = pd.to_datetime(ephem["epoch_utc"], utc=True)

        # Set index to 'epoch_utc' column, also keep the column
        ephem = ephem.set_index("epoch_utc", drop=False)

        # Sort by time
        ephem = ephem.sort_index()

        df = interpolate_pointing(
            ephem,
            t_res=input_dict["t_res"],
            method=input_dict["method"],
            save_df=input_dict["save_df"],
            filename=input_dict["filename"],
            filetype=input_dict["filetype"],
        )

    return df



# TODO: All parameters should just be required; then pass defaults from the LEXI class.
# Maybe do that after we decide what is a method of the LEXI class and what is "private".
def exposure_map(
        spc_df,    # spacecraft orientation dataframe with timestamps,
                   # look-directions (gamma) and roll-angles (phi).
                   # Roll angles are not used in this function.
        t_range, # [start time, end time] (unit/format?)
        t_integrate, # integration interval for final LEXI histograms (in seconds)
        t_step=0.01, # exposure map step time in seconds: For each look-direction in
                     # [insert name of dataframe arg here],
                     # this many seconds are added to each in-FOV cell in the exposure map.
                     # This must correspond to the resampling/interpolation interval
                     # given to [insert name of function here].
        lexi_fov=9.1, # LEXI FOV in degrees
        ra_range=[325.0, 365.0], # [start RA, end RA]
        dec_range=[-21.0, 6.0], # [start DEC, end DEC]
        ra_res=0.1, # RA resolution in degrees. Ideal value is 0.1 deg
        dec_res=0.1, # DEC resolution in degrees. Ideal value is 0.1 deg
        save_maps=False):
    """
    (TODO: Docstring)
    Returns an array of exposure maps.
    """
    try:
        # Read the exposure map from a pickle file
        # TODO: Must match filename to the save_maps step; and the filename should include ALL the params
        exposure = np.load(
            f"../data/exposure_map_rares_{ra_res}_decres_{dec_res}_tstep_{t_step}.npy"
        )
        print("Exposure map loaded from file \n")
    except FileNotFoundError:
        print("Exposure map not found, computing now. This may take a while \n")

        # Set up coordinate grid
        ra_grid = np.arange(ra_range[0], ra_range[1], ra_res)
        dec_grid = np.arange(dec_range[0], dec_range[1], dec_res)
        ra_grid_arr = np.tile(ra_grid, (len(dec_grid), 1)).transpose()
        dec_grid_arr = np.tile(dec_grid, (len(ra_grid), 1))

        # Slice to relevant time range; make groups of rows spanning t_integration
        integ_groups = spc_df[pd.Timestamp(t_range[0]):pd.Timestamp(t_range[1])].resample(pd.Timedelta(t_integrate, unit='s'))

        # Make as many empty exposure maps as there are integration groups
        exposure_maps = np.zeros((len(integ_groups), len(ra_grid), len(dec_grid)))

        # TODO: Can we figure out a way to do this not in a loop??? Cannot be vectorized...
        # Loop through each pointing step and add the exposure to the map
        for (map_idx, (name, group)) in enumerate(integ_groups):
            # TODO remove prints
            print("Group: ", name)
            print(group, end="\n\n")

            for row in group.itertuples():
              # Get distance in degrees to the pointing step
              # Wrap-proofing: First make everything [0,360), then +-360 on second operand
              ra_diff  = min(abs((ra_grid_arr%360)-(row.mp_ra%360))
                            ,abs((ra_grid_arr%360)-(row.mp_ra%360-360))
                            ,abs((ra_grid_arr%360)-(row.mp_ra%360+360)))
              dec_diff = min(abs((dec_grid_arr%360)-(row.mp_dec%360))
                            ,abs((dec_grid_arr%360)-(row.mp_dec%360-360))
                            ,abs((dec_grid_arr%360)-(row.mp_dec%360+360)))
              r = np.sqrt(ra_diff ** 2 + dec_diff ** 2)
              # Make an exposure delta for this span
              exposure_delt = np.where(
                  (r < lexi_fov * 0.5), vignette(r) * t_step, 0
              )
              exposure_maps[map_idx] += exposure_delt  # Add the delta to the full map

        # TODO: Print progress. But this depends on whether we keep the loop or not
        """
        # Check how long it takes to run a single step and based on that estimate how long it will take
        # to run
        start_time_loop = time.time()
        r = np.sqrt((ra_grid_arr - df.lookdir_ra[0]) ** 2 + (dec_grid_arr - df.lookdir_dec[0]) ** 2)
        exposure_delt = np.where(
            (r < lexi_fov * 0.5), vignette(r) * t_step, 0
        )
        exposure += exposure_delt
        end_time_loop = time.time()
        print(
            f"Estimated time to run: \x1b[1;32;255m{np.round((end_time_loop - start_time_loop) * len(df) / 60, 1)} \x1b[0m minutes"
        )
        [...]
        for i in range(len(df)):
            # Print the progress in terminal in percentage complete without a new line for each one percent
            # increase
            print(
                f"Computing exposure map ==> \x1b[1;32;255m {np.round(i/len(df)*100, 6)}\x1b[0m % complete",
                end="\r",
            )
        # #return exposure
        """

        # TODO: Confirm this is fine: I am defining t_step to be in seconds, therefore no need for normalization.
        # since the ROSAT data are in counts/sec anyway
        #exposure = exposure * int(input_dict["t_res"][:-1]) / 1000

        # TODO: see above re filename and matching
        if save_maps:
            # Save the exposure map array to a pickle file
            np.save(
                #f"../data/exposure_map_rares_{ra_res}_decres_{dec_res}_tstep_{t_step}_t0_{t_range[0]}_tf_{t_range[1]}.npy",
                f"exposure_map_rares_{ra_res}_decres_{dec_res}_tstep_{t_step}_t0_{t_range[0]}_tf_{t_range[1]}.npy",
                exposure_maps,
            )

    return exposure_maps


def matplotlib_figure(
    df=None,
    input_dict=None,
    exposure=None,
    start_string=None,
    stop_string=None,
    v_min=None,
    v_max=None,
    norm_type="log",
    display=False,
    filename=None,
    figure_format="png",
):
    """
    Function to create and save a matplotlib figure of the exposure map

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with interpolated pointing data
    input_dict : dict
        Dictionary of inputs.  Default is None
    exposure : numpy array
        Exposure map.  Default is None
    start_string : string
        Start time of the exposure map.  Default is None
    stop_string : string
        Stop time of the exposure map.  Default is None
    v_min : float
        Minimum value for the colorbar.  Default is None
    v_max : float
        Maximum value for the colorbar.  Default is None
    norm_type : string
        Type of normalization to use.  Default is 'log'
    display : bool
        If True, display the plot in the browser.  Default is False
    filename : string
        Filename to save the figure to.  Default is None
    figure_format : string
        Format to save the figure to.  Default is 'png'

    Returns
    -------
    axs : matplotlib axes
        Axes of the figure

    Raises
    ------
    TypeError
        If df is not a pandas dataframe
    TypeError
        If input_dict is not a dictionary
    TypeError
        If exposure is not a numpy array
    TypeError
        If start_string is not a string
    TypeError
        If stop_string is not a string
    TypeError
        If v_min is not a float or an int
    TypeError
        If v_max is not a float or an int
    TypeError
        If filename is not a string
    TypeError
        If figure_format is not a string
    """

    # Check if df is a pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")

    # Check if input_dict is a dictionary
    if not isinstance(input_dict, dict):
        raise TypeError("input_dict must be a dictionary")

    # Check if exposure is a numpy array
    if not isinstance(exposure, np.ndarray):
        raise TypeError("exposure must be a numpy array")

    # Check if start_string is a string
    if not isinstance(start_string, str):
        raise TypeError("start_string must be a string")

    # Check if stop_string is a string
    if not isinstance(stop_string, str):
        raise TypeError("stop_string must be a string")

    # Check if v_min is not None and is a float or an int
    if v_min is not None:
        if not isinstance(v_min, (float, int)):
            raise TypeError("v_min must be a float or an int")

    # Check if v_max is not None and is a float or an int
    if v_max is not None:
        if not isinstance(v_max, (float, int)):
            raise TypeError("v_max must be a float or an int")

    # Check if filename is a string
    if not isinstance(filename, str):
        # Check if filename is None
        if filename is None:
            filename = f"../figures/exposure_map_test_xres_{input_dict['x_res']}_yres_{input_dict['y_res']}_tres_{input_dict['t_res']}.{figure_format}"
        else:
            raise TypeError("filename must be a string")

    # Check if figure_format is a string
    if not isinstance(figure_format, str):
        raise TypeError("figure_format must be a string")

    # Set latex use to true
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    min_exposure = np.nanmin(exposure)
    max_exposure = np.nanmax(exposure)

    # If either v_min or v_max is None, then set them to the min and max exposure values
    if v_min is None or v_max is None:
        if norm_type == "linear":
            # Set the min and max values for the colorbar
            v_min = 0.9 * min_exposure
            v_max = 1.1 * max_exposure
            norm = cm.colors.Normalize(vmin=v_min, vmax=v_max)
        elif norm_type == "log":
            # Set the min and max values for the colorbar
            v_min = 10
            v_max = 1.1 * max_exposure
            norm = cm.colors.LogNorm(vmin=v_min, vmax=v_max)

    # Set plotstyle to dark background
    plt.style.use("default")

    # Plot the exposure map and set grid to on
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300, facecolor="w", edgecolor="g")
    ax.grid(True)
    # Set minor grid lines
    ax.minorticks_on()
    ax.grid(which="major", linestyle="-", linewidth="0.5", color="m", alpha=0.1)
    ax.grid(which="minor", linestyle="-", linewidth="0.2", color="red", alpha=0.1)

    plt.xlabel("RA [$^\\circ$]")
    plt.ylabel("DEC [$^\\circ$]")
    plt.title(f"LEXI: {start_string} - {stop_string}")

    im = ax.imshow(
        np.transpose(exposure),
        cmap="Blues",
        extent=[
            input_dict["xrange"][0],
            input_dict["xrange"][1],
            input_dict["yrange"][0],
            input_dict["yrange"][1],
        ],
        origin="lower",
        aspect="equal",
        norm=norm,
    )

    # Add n_steps and step as text to the plot
    plt.text(
        0.05,
        0.95,
        f"n_steps = {len(df)} \n step = {input_dict['step']} s \n Exposure = [{min_exposure:.2f}, {max_exposure:.2f}]",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=12,
    )

    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
    )
    plt.colorbar(im, cax=cax, label="Time in each pixel [s]")

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )

    print(f"Exposure map saved to file : {filename} \n")

    if display:
        plt.show()


def bokeh_figure(
    input_dict=None,
    exposure=None,
    start_string=None,
    stop_string=None,
    v_min=None,
    v_max=None,
    display=False,
):
    """
    Function to create and save an interactive plot of the exposure map so that the user can hover
    over the map and see the ra, dec and the exposure time

    Parameters
    ----------
    input_dict : dict
        Dictionary of inputs.  Default is None
    exposure : numpy array
        Exposure map.  Default is None
    start_string : string
        Start time of the exposure map.  Default is None
    stop_string : string
        Stop time of the exposure map.  Default is None
    v_min : float
        Minimum value for the colorbar.  Default is None
    v_max : float
        Maximum value for the colorbar.  Default is None
    display : bool
        If True, display the plot in the browser.  Default is False

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If input_dict is not a dictionary
    TypeError
        If exposure is not a numpy array
    TypeError
        If start_string is not a string
    TypeError
        If stop_string is not a string
    TypeError
        If v_min is not a float or an int
    TypeError
        If v_max is not a float or an int
    """

    # Check if input_dict is a dictionary
    if not isinstance(input_dict, dict):
        raise TypeError("input_dict must be a dictionary")

    # Check if exposure is a numpy array
    if not isinstance(exposure, np.ndarray):
        raise TypeError("exposure must be a numpy array")

    # Check if start_string is a string
    if not isinstance(start_string, str):
        raise TypeError("start_string must be a string")

    # Check if stop_string is a string
    if not isinstance(stop_string, str):
        raise TypeError("stop_string must be a string")

    # Check if v_min is a float or an int
    if v_min is None or v_max is None:
        # Set the min and max values for the colorbar
        v_min = 0.9 * np.nanmin(exposure)
        v_max = 1.1 * np.nanmax(exposure)

    if not isinstance(v_min, (float, int)):
        raise TypeError("v_min must be a float or an int")

    # Check if v_max is a float or an int
    if not isinstance(v_max, (float, int)):
        raise TypeError("v_max must be a float or an int")

    # Set the output file name
    output_file = f"../figures/exposure_map_test_xres_{input_dict['x_res']}_yres_{input_dict['y_res']}_tres_{input_dict['t_res']}.html"

    # Set the figure title
    title = f"LEXI: {start_string} - {stop_string}"

    # Set the hover tool tip
    hover = HoverTool(
        tooltips=[
            ("RA", "$x"),
            ("DEC", "$y"),
            ("Exposure", "@image"),
        ]
    )

    # Set the figure
    p = figure(
        title=title,
        x_axis_label="RA [deg]",
        y_axis_label="DEC [deg]",
        x_range=(input_dict["xrange"][0], input_dict["xrange"][1]),
        y_range=(input_dict["yrange"][0], input_dict["yrange"][1]),
        tools=[
            hover,
            "pan",
            "crosshair",
            "wheel_zoom",
            "box_zoom",
            "reset",
            "save",
            "help",
        ],
        match_aspect=True,
    )

    # Set the color mapper
    color_mapper = bokeh.models.LinearColorMapper(
        palette=list(reversed(bokeh.palettes.Blues256)), low=v_min, high=v_max
    )

    # Set the color bar
    color_bar = bokeh.models.ColorBar(
        color_mapper=color_mapper,
        label_standoff=12,
        border_line_color=None,
        location=(0, 0),
        title="Time in each pixel [s]",
        # ticker=bokeh.models.LogTicker(desired_num_ticks=10),
        ticker=bokeh.models.BasicTicker(desired_num_ticks=10),
    )

    # Set the plot
    p.image(
        image=[exposure],
        x=input_dict["xrange"][0],
        y=input_dict["yrange"][0],
        dw=input_dict["xrange"][1] - input_dict["xrange"][0],
        dh=input_dict["yrange"][1] - input_dict["yrange"][0],
        color_mapper=color_mapper,
    )

    # Add the color bar to the plot
    p.add_layout(color_bar, "right")

    # Save the plot to an html file and show it in the browser with grid lines
    bokeh.io.save(p, filename=output_file, title=title, resources=bokeh.resources.CDN)

    if display:
        bokeh.io.show(p)

    return None
