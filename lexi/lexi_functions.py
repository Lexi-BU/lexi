import numbers
import numpy as np
import pandas as pd
import pytz
import pickle
import urllib.request
from pathlib import Path
from cdflib import CDF
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings


# Define a list of global variables
# Define the field of view of LEXI in degrees
LEXI_FOV = 9.1


def validate_input(key, value):
    if key == "t_range":
        if not isinstance(value, list):
            raise ValueError("t_range must be a list")
        if len(value) != 2:
            raise ValueError("t_range must have two elements")
        # Check that all elements are either strings, or datetime objects or Timestamps
        if not all(isinstance(x, (str, pd.Timestamp)) for x in value):
            raise ValueError("t_range elements must be strings or datetime objects")

    if key == "time_zone":
        if not isinstance(value, str):
            raise ValueError("time_zone must be a string")
        if len(value) == 0:
            raise ValueError("time_zone must not be an empty string")
        # Check that the timezone is valid
        if value not in pytz.all_timezones:
            # Print a warning that the provided timezone is not valid and set it to UTC
            warnings.warn(
                f"\n \033[1;92m Timezone '{value}' \033[1;91mis not valid. Setting timezone to UTC \033[0m \n"
            )
            return False

    if key == "t_step":
        if not isinstance(value, (int, float)) or value < 0:
            warnings.warn(
                "\n \033[1;92m t_step \033[1;91m must be a positive integer or float.\033[0m \n"
            )
            return False

    if key == "ra_range":
        # Check if the ra_range is a list, tuple, or numpy array
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise ValueError("ra_range must be a list, tuple, or numpy array")
        if len(value) != 2:
            raise ValueError("ra_range must have two elements")
        # Check if all elements are numbers
        if not all(isinstance(x, numbers.Number) for x in value):
            warnings.warn(
                "\n \033[1;91m ra_range elements must be numbers. Setting ra_range to default value of from the spacecraft ephemeris file. \033[0m \n"
            )
            return False
        if value[0] < 0 or value[0] >= 360:
            warnings.warn(
                "\n \033[1;92m ra_range start \033[1;91m must be in the range [0, 360). Setting ra_range to default value of from the spacecraft ephemeris file. \033[0m \n"
            )
            return False
        if value[1] <= 0 or value[1] > 360:
            warnings.warn(
                "\n \033[1;92m ra_range stop \033[1;92m must be in the range (0, 360]. Setting ra_range to default value of from the spacecraft ephemeris file. \033[0m \n"
            )
            return False

    if key == "dec_range":
        # Check if the dec_range is a list, tuple, or numpy array
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise ValueError("dec_range must be a list, tuple, or numpy array")
        if len(value) != 2:
            raise ValueError("dec_range must have two elements")
        # Check if all elements are numbers
        if not all(isinstance(x, numbers.Number) for x in value):
            warnings.warn(
                "\n \033[1;91m dec_range elements must be numbers. Setting dec_range to default value of from the spacecraft ephemeris file. \033[0m \n"
            )
            return False
        if value[0] < -90 or value[0] > 90:
            warnings.warn(
                "\n \033[1;92m dec_range start \033[1;91m must be in the range [-90, 90]. Setting dec_range to default value of from the spacecraft ephemeris file. \033[0m \n"
            )
            return False
        if value[1] <= -90 or value[1] > 90:
            warnings.warn(
                "\n \033[1;92m dec_range stop \033[1;91m must be in the range (-90, 90]. Setting dec_range to default value of from the spacecraft ephemeris file. \033[0m \n"
            )
            return False

    if key == "ra_res":
        if not isinstance(value, numbers.Number):
            warnings.warn(
                "\n \033[1;92m ra_res \033[1;91m must be a positive number. Setting ra_res to default value of \033[1;92m 0.1 \033[0m \n"
            )
            return False
        if value <= 0:
            warnings.warn(
                "\n \033[1;92m ra_res \033[1;91m must be a positive number. Setting ra_res to default value of \033[1;92m 0.1 \033[0m \n"
            )
            return False

    if key == "dec_res":
        if not isinstance(value, numbers.Number):
            warnings.warn(
                "\n \033[1;92m dec_res \033[1;91m must be a positive number. Setting dec_res to default value of \033[1;92m 0.1 \033[0m \n"
            )
            return False
        if value <= 0:
            warnings.warn(
                "\n \033[1;92m dec_res \033[1;91m must be a positive number. Setting dec_res to default value of \033[1;92m 0.1 \033[0m \n"
            )
            return False

    if key == "t_integrate":
        if not isinstance(value, numbers.Number):
            warnings.warn(
                "\n \033[1;92m t_integrate \033[1;91m must be a positive number. Setting t_integrate to default value \033[0m \n"
            )
            return False
        if value <= 0:
            return False

    if key == "interp_method":
        if not isinstance(value, str):
            raise ValueError("interp_method must be a string")
        if value not in [
            "linear",
            "nearest",
            "zero",
            "slinear",
            "quadratic",
            "cubic",
        ]:
            warnings.warn(
                f"\n \033[1;92m Interpolation method '{value}' \033[1;91m is not a valid interpolation method. Setting interpolation method to \033[1;92m 'linear' \033[0m \n"
            )
            return False

    if key == "background_correction_on":
        if not isinstance(value, bool):
            raise ValueError("background_correction_on must be a boolean")

    if key == "save_df":
        if not isinstance(value, bool):
            raise ValueError("save_df must be a boolean")

    if key == "filename":
        if not isinstance(value, str):
            raise ValueError("filename must be a string")
        if len(value) == 0:
            raise ValueError("filename must not be an empty string")

    if key == "filetype":
        if not isinstance(value, str):
            raise ValueError("filetype must be a string")
        if len(value) == 0:
            raise ValueError("filetype must not be an empty string")
        if value not in ["pkl", "p", "csv"]:
            raise ValueError("filetype must be one of 'pkl', 'p' or 'csv")

    if key == "save_exposure_maps":
        if not isinstance(value, bool):
            raise ValueError("save_exposure_maps must be a boolean")

    if key == "save_sky_backgrounds":
        if not isinstance(value, bool):
            raise ValueError("save_sky_backgrounds must be a boolean")

    if key == "save_lexi_images":
        if not isinstance(value, bool):
            raise ValueError("save_lexi_images must be a boolean")

    return True


def get_spc_prams(
    t_range=None,
    time_zone="UTC",
    t_step=5,
    interp_method=None,
    verbose=True,
):
    # Validate t_range
    t_range_validated = validate_input("t_range", t_range)

    if t_range_validated:
        # If t_range elements are strings, convert them to datetime objects
        if isinstance(t_range[0], str):
            t_range[0] = pd.to_datetime(t_range[0])
        if isinstance(t_range[1], str):
            t_range[1] = pd.to_datetime(t_range[1])
        # Validate time_zone, if it is not valid, set it to UTC
        if time_zone is not None:
            time_zone_validated = validate_input("time_zone", time_zone)
            if time_zone_validated:
                # Set the timezone to the t_range
                t_range[0] = t_range[0].tz_localize(time_zone)
                t_range[1] = t_range[1].tz_localize(time_zone)
                if verbose:
                    print(f"Timezone set to \033[1;92m {time_zone} \033[0m \n")
            else:
                t_range[0] = t_range[0].tz_localize("UTC")
                t_range[1] = t_range[1].tz_localize("UTC")
                if verbose:
                    print("Timezone set to \033[1;92m UTC \033[0m \n")

    # Validate t_step
    t_step_validated = validate_input("t_step", t_step)
    if not t_step_validated:
        t_step = 5

    # Validate interp_method
    interp_method_validated = validate_input("interp_method", interp_method)
    if not interp_method_validated:
        interp_method = "linear"

    # TODO: REMOVE ME once we start using real ephemeris data
    df = pd.read_csv("../data/sample_lexi_pointing_ephem_edited.csv")
    # Convert the epoch_utc column to a datetime object
    df["epoch_utc"] = pd.to_datetime(df["epoch_utc"])
    # Set the index to be the epoch_utc column and remove the epoch_utc column
    df = df.set_index("epoch_utc", inplace=False)
    # Set the timezone to UTC
    df = df.tz_localize("UTC")

    if df.index[0] > t_range[0] or df.index[-1] < t_range[1]:
        warnings.warn(
            "Ephemeris data do not cover the full time range requested."
            "End regions will be forward/backfilled."
        )
        # Add the just the two endpoints to the index
        df = df.reindex(
            index=np.union1d(pd.date_range(t_range[0], t_range[1], periods=2), df.index)
        )

    dfslice = df[t_range[0] : t_range[1]]
    dfresamp = dfslice.resample(pd.Timedelta(t_step, unit="s"))
    dfinterp = dfresamp.interpolate(method=interp_method, limit_direction="both")
    return dfinterp

    # (end of chunk that must be removed once we start using real ephemeris data)

    # Get the year, month, and day of the start and stop times
    start_time = t_range[0]
    stop_time = t_range[1]

    start_year = start_time.year
    start_month = start_time.month
    start_day = start_time.day

    stop_year = stop_time.year
    stop_month = stop_time.month
    stop_day = stop_time.day

    # Link to the CDAweb website, from which ephemeris data are pulled
    # CDA_LINK = "https://cdaweb.gsfc.nasa.gov/pub/data/lexi/ephemeris/"
    # TODO: Change this to the correct link once we start using real ephemeris data
    CDA_LINK = (
        "https://cdaweb.gsfc.nasa.gov/pub/data/ulysses/plasma/swics_cdaweb/scs_m1/2001/"
    )

    # Given that ephemeris files are named in the the format of lexi_ephm_YYYYMMDD_v01.cdf, get a
    # list of all the files that are within the time range of interest
    file_list = []
    for year in range(start_year, stop_year + 1):
        for month in range(start_month, stop_month + 1):
            for day in range(start_day, stop_day + 1):
                # Create a string for the date in the format of YYYYMMDD
                date_string = str(year) + str(month).zfill(2) + str(day).zfill(2)

                # Create a string for the filename
                # filename = "lexi_ephm_" + date_string + "_v01.cdf"
                # TODO: Change this to the correct filename format once we start using real ephemeris data
                filename = "uy_m1_scs_" + date_string + "_v02.cdf"

                # Create a string for the full link to the file
                link = CDA_LINK + filename

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
    if not verbose:
        print("Downloading ephemeris files\n")
    for file in file_list:
        # If the file already exists, then skip to the next file
        if (data_dir / file).exists():
            if verbose:
                print(f"File already exists ==> \033[92m {file}\033[0m \n")
            continue
        # If the file doesn't exist, then download it
        urllib.request.urlretrieve(CDA_LINK + file, data_dir / file)
        if verbose:
            print(f"Downloaded ==> \033[92m {file}\033[0m \n")

    # Read the files into a single dataframe
    df_list = []
    if not verbose:
        print("Reading ephemeris files\n")
    for file in file_list:
        if verbose:
            print(f"Reading ephemeris file ==> \033[92m {file}\033[0m \n")
        # Get the file path
        file = data_dir / file
        eph_data = CDF(file)

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

    # If the ephemeris data do not span the t_range, send warning
    if df.index[0] > t_range[0] or df.index[-1] < t_range[1]:
        warnings.warn(
            "Ephemeris data do not cover the full time range requested."
            "End regions will be forward/backfilled."
        )
        # Add the just the two endpoints to the index
        df = df.reindex(
            index=np.union1d(pd.date_range(t_range[0], t_range[1], periods=2), df.index)
        )

    # Slice, resample, interpolate
    dfslice = df[t_range[0] : t_range[1]]
    dfresamp = dfslice.resample(pd.Timedelta(t_step, unit="s"))
    # Validate interp_method
    interp_method_validated = validate_input("interp_method", interp_method)
    if interp_method_validated:
        dfinterp = dfresamp.interpolate(method=interp_method, limit_direction="both")

    return dfinterp


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


def get_exposure_maps(
    t_range=None,
    time_zone="UTC",
    interp_method=None,
    t_step=5,
    ra_range=[0, 360],
    dec_range=[-90, 90],
    ra_res=0.1,
    dec_res=0.1,
    t_integrate=None,
    save_exposure_map_file=False,
    save_exposure_map_image=False,
    verbose=True,
):

    # Validate t_step
    t_step_validated = validate_input("t_step", t_step)
    if not t_step_validated:
        t_step = 5
        if verbose:
            print(
                f"\033[1;91m Time step \033[1;92m (t_step) \033[1;91m not provided. Setting time step to \033[1;92m {t_step} seconds \033[0m\n"
            )

    # Validate ra_range
    ra_range_validated = validate_input("ra_range", ra_range)

    # Validate dec_range
    dec_range_validated = validate_input("dec_range", dec_range)

    # Validate ra_res
    ra_res_validated = validate_input("ra_res", ra_res)
    if not ra_res_validated:
        ra_res = 0.1

    # Validate dec_res
    dec_res_validated = validate_input("dec_res", dec_res)
    if not dec_res_validated:
        dec_res = 0.1

    print(t_range)
    # Get spacecraft ephemeris data
    spc_df = get_spc_prams(
        t_range=t_range,
        time_zone=time_zone,
        interp_method=interp_method,
        verbose=verbose,
    )

    # Validate t_integrate
    if t_integrate is None:
        # If t_integrate is not provided, set it to the timedelta of the provided t_range
        t_integrate = (t_range[1] - t_range[0]).total_seconds()
        if verbose:
            print(
                f"\033[1;91m Integration time \033[1;92m (t_integrate) \033[1;91m not provided. Setting integration time to the time span of the spacecraft ephemeris data: \033[1;92m {t_integrate} seconds \033[0m\n"
            )
    else:
        t_integrate_validated = validate_input("t_integrate", t_integrate)
        if not t_integrate_validated:
            t_integrate = (t_range[1] - t_range[0]).total_seconds()
            if verbose:
                print(
                    f"\033[1;91m Integration time \033[1;92m (t_integrate) \033[1;91m not provided. Setting integration time to the time span of the spacecraft ephemeris data: \033[1;92m {t_integrate} seconds \033[0m\n"
                )

    # TODO: REMOVE ME once we start using real ephemeris data
    # The sample ephemeris data uses column names "mp_ra" and "mp_dec" for look direction;
    # in the final lexi ephemeris files on CDAweb, this will be called just "RA" and "DEC".
    # Therefore...
    spc_df["RA"] = spc_df.mp_ra
    spc_df["DEC"] = spc_df.mp_dec
    # (end of chunk that must be removed once we start using real ephemeris data)

    # Set up coordinate grid
    if ra_range_validated:
        ra_arr = np.arange(ra_range[0], ra_range[1], ra_res)
    else:
        ra_range = np.array([np.nanmin(spc_df["RA"]), np.nanmax(spc_df["RA"])])
        ra_arr = np.arange(ra_range[0], ra_range[1], ra_res)
        if verbose:
            print(
                f"\033[1;91m RA range \033[1;92m (ra_range) \033[1;91m not provided. Setting RA range to the range of the spacecraft ephemeris data: \033[1;92m {ra_range} \033[0m\n"
            )

    if dec_range_validated:
        dec_arr = np.arange(dec_range[0], dec_range[1], dec_res)
    else:
        dec_range = np.array([np.nanmin(spc_df["DEC"]), np.nanmax(spc_df["DEC"])])
        dec_arr = np.arange(dec_range[0], dec_range[1], dec_res)

    ra_grid = np.tile(ra_arr, (len(dec_arr), 1)).transpose()
    dec_grid = np.tile(dec_arr, (len(ra_arr), 1))

    try:
        # Read the exposure map from a pickle file, if it exists
        # Define the folder where the exposure maps are saved
        save_folder = Path(__file__).resolve().parent.parent / "data/exposure_maps"
        t_start = t_range[0].strftime("%Y%m%d_%H%M%S")
        t_stop = t_range[1].strftime("%Y%m%d_%H%M%S")
        ra_start = ra_range[0]
        ra_stop = ra_range[1]
        dec_start = dec_range[0]
        dec_stop = dec_range[1]
        ra_res = ra_res
        dec_res = dec_res
        t_integrate = int(t_integrate)
        exposure_maps_file_name = (
            f"{save_folder}/lexi_exposure_map_Tstart_{t_start}_Tstop_{t_stop}_RAstart_{ra_start}"
            f"_RAstop_{ra_stop}_RAres_{ra_res}_DECstart_{dec_start}_DECstop_{dec_stop}_DECres_"
            f"{dec_res}_Tint_{t_integrate}.npy"
        )
        # Read the exposure map from the pickle file
        exposure_maps_dict = pickle.load(open(exposure_maps_file_name, "rb"))
        exposure_maps = exposure_maps_dict["exposure_maps"]
        print(
            f"Exposure map loaded from file \033[92m {exposure_maps_file_name} \033[0m\n"
        )
    except FileNotFoundError:
        print("Exposure map not found, computing now. This may take a while \n")

        # Slice to relevant time range; make groups of rows spanning t_integration
        integ_groups = spc_df[t_range[0] : t_range[1]].resample(
            pd.Timedelta(t_integrate, unit="s"), origin="start"
        )
        # Make as many empty exposure maps as there are integration groups
        exposure_maps = np.zeros((len(integ_groups), len(ra_arr), len(dec_arr)))

        # Loop through each pointing step and add the exposure to the map
        # Wrap-proofing: First make everything [0,360)...
        ra_grid_mod = ra_grid % 360
        dec_grid_mod = dec_grid % 180
        for map_idx, (_, group) in enumerate(integ_groups):
            for row in group.itertuples():
                # Get distance in degrees to the pointing step
                # Wrap-proofing: First make everything [0,360), then +-360 on second operand
                # TODO: Change the dec wrap-proofing to +-90. Check if this is right
                row_ra_mod = row.RA % 360
                row_dec_mod = row.DEC % 90

                ra_diff = np.minimum(
                    abs(ra_grid_mod - row_ra_mod),
                    abs(ra_grid_mod - (row_ra_mod - 360)),
                    abs(ra_grid_mod - (row_ra_mod + 360)),
                )
                dec_diff = np.minimum(
                    abs(dec_grid_mod - row_dec_mod),
                    abs(dec_grid_mod - (row_dec_mod - 90)),
                    abs(dec_grid_mod - (row_dec_mod + 90)),
                )
                r = np.sqrt(ra_diff**2 + dec_diff**2)
                # Make an exposure delta for this span
                exposure_delt = np.where((r < LEXI_FOV * 0.5), vignette(r) * t_step, 0)
                # Add the delta to the full map
                exposure_maps[map_idx] += exposure_delt
            print(
                f"Computing exposure map ==> \x1b[1;32;255m {np.round(map_idx/len(integ_groups)*100, 6)}\x1b[0m % complete",
                end="\r",
            )
        if save_exposure_map_file:
            # Define the folder to save the exposure maps to
            save_folder = Path(__file__).resolve().parent.parent / "data/exposure_maps"
            Path(save_folder).mkdir(parents=True, exist_ok=True)
            t_start = t_range[0].strftime("%Y%m%d_%H%M%S")
            t_stop = t_range[1].strftime("%Y%m%d_%H%M%S")
            ra_start = ra_range[0]
            ra_stop = ra_range[1]
            dec_start = dec_range[0]
            dec_stop = dec_range[1]
            ra_res = ra_res
            dec_res = dec_res
            t_integrate = int(t_integrate)
            exposure_maps_file_name = (
                f"{save_folder}/lexi_exposure_map_Tstart_{t_start}_Tstop_{t_stop}_RAstart_{ra_start}"
                f"_RAstop_{ra_stop}_RAres_{ra_res}_DECstart_{dec_start}_DECstop_{dec_stop}_DECres_"
                f"{dec_res}_Tint_{t_integrate}.npy"
            )
            # Define a dictoinary to store the exposure maps, ra_arr, and dec_arr, t_range, and t_integrate,
            # ra_range, and dec_range, ra_res, and dec_res
            exposure_maps_dict = {
                "exposure_maps": exposure_maps,
                "ra_arr": ra_arr,
                "dec_arr": dec_arr,
                "t_range": t_range,
                "t_integrate": t_integrate,
                "ra_range": ra_range,
                "dec_range": dec_range,
                "ra_res": ra_res,
                "dec_res": dec_res,
            }
            # Save the exposure map array to a pickle file
            with open(exposure_maps_file_name, "wb") as f:
                pickle.dump(exposure_maps_dict, f)
            print(
                f"Exposure map saved to file: \033[92m {exposure_maps_file_name} \033[0m \n"
            )
    if save_exposure_map_image:
        print("Saving exposure maps as images")
        for i, exposure in enumerate(exposure_maps):
            array_to_image(
                exposure,
                x_range=ra_range,
                y_range=dec_range,
                cmap="jet",
                norm=None,
                norm_type="log",
                aspect="auto",
                figure_title=f"Exposure Map {i}",
                show_colorbar=True,
                cbar_label="Seconds",
                cbar_orientation="vertical",
                show_axes=True,
                display=True,
                figure_size=(10, 10),
                figure_format="png",
                figure_font_size=12,
                save=True,
                save_path="figures/exposure_maps",
                save_name=f"exposure_map_{i}",
                dpi=300,
                dark_mode=True,
            )
    # If the first element of exposure_maps shape is 1, then remove the first dimension
    # if np.shape(exposure_maps)[0] == 1:
    #     exposure_maps = exposure_maps[0]

    return exposure_maps_dict


def get_sky_backgrounds(
    t_range=None,
    time_zone="UTC",
    interp_method=None,
    t_step=5,
    t_integrate=None,
    ra_range=[0, 360],
    dec_range=[-90, 90],
    ra_res=0.1,
    dec_res=0.1,
    save_exposure_map_file=False,
    save_exposure_map_image=False,
    save_sky_backgrounds_file=False,
    save_sky_backgrounds_image=False,
    verbose=True,
):

    # Get exposure maps
    exposure_maps_dict = get_exposure_maps(
        t_range=t_range,
        time_zone=time_zone,
        interp_method=interp_method,
        t_step=t_step,
        ra_range=ra_range,
        dec_range=dec_range,
        ra_res=ra_res,
        dec_res=dec_res,
        t_integrate=t_integrate,
        save_exposure_map_file=save_exposure_map_file,
        save_exposure_map_image=save_exposure_map_image,
        verbose=verbose,
    )
    exposure_maps = exposure_maps_dict["exposure_maps"]
    # If exposure_maps is a 2D array, then add a dimension to it
    if len(np.shape(exposure_maps)) == 2:
        exposure_maps = np.array([exposure_maps])

    try:
        # Read the sky background from a pickle file, if it exists
        # Define the folder where the sky backgrounds are saved
        save_folder = Path(__file__).resolve().parent.parent / "data/sky_backgrounds"
        t_start = exposure_maps_dict["t_range"][0].strftime("%Y%m%d_%H%M%S")
        t_stop = exposure_maps_dict["t_range"][1].strftime("%Y%m%d_%H%M%S")
        ra_start = exposure_maps_dict["ra_range"][0]
        ra_stop = exposure_maps_dict["ra_range"][1]
        dec_start = exposure_maps_dict["dec_range"][0]
        dec_stop = exposure_maps_dict["dec_range"][1]
        ra_res = exposure_maps_dict["ra_res"]
        dec_res = exposure_maps_dict["dec_res"]
        t_integrate = int(exposure_maps_dict["t_integrate"])
        sky_backgrounds_file_name = (
            f"{save_folder}/lexi_sky_background_Tstart_{t_start}_Tstop_{t_stop}_RAstart_{ra_start}"
            f"_RAstop_{ra_stop}_RAres_{ra_res}_DECstart_{dec_start}_DECstop_{dec_stop}_DECres_"
            f"{dec_res}_Tint_{t_integrate}.npy"
        )
        # Read the sky background from the pickle file
        sky_backgrounds_dict = pickle.load(open(sky_backgrounds_file_name, "rb"))
        print(
            f"Sky background loaded from file \033[92m {sky_backgrounds_file_name} \033[0m\n"
        )
    except FileNotFoundError:
        print("Sky background not found, computing now. This may take a while \n")

        # Get ROSAT background
        # Ultimately KKip is supposed to provide this file and we will have it saved somewhere static.
        # For now, this is Cadin's sample xray data:
        rosat_df = pd.read_csv("../data/sample_xray_background.csv", header=None)
        # Slice to RA/DEC range, interpolate to RA/DEC res
        # For now just interpolate Cadin data:
        # TODO: when using actual data, check that axes are correct (index/column to ra/dec)
        rosat_df.index = np.linspace(ra_range[0], ra_range[1], 100)
        rosat_df.columns = np.linspace(dec_range[0], dec_range[1], 100)

        # Reindex to include desired RA/DEC indices (but don't throw out old indices yet; need for
        # interpolation)
        desired_ra_idx = np.arange(ra_range[0], ra_range[1], ra_res)
        desired_dec_idx = np.arange(dec_range[0], dec_range[1], dec_res)
        rosat_enlarged_idx = rosat_df.reindex(
            index=np.union1d(rosat_df.index, desired_ra_idx),
            columns=np.union1d(rosat_df.columns, desired_dec_idx),
        )
        # Interpolate and then throw out the old indices to get correct dimensions
        rosat_interpolated = rosat_enlarged_idx.interpolate(
            method=interp_method
        ).interpolate(method=interp_method, axis=1)
        rosat_resampled = rosat_interpolated.reindex(
            index=desired_ra_idx, columns=desired_dec_idx
        )

        # Multiply each exposure map (seconds) with the ROSAT background (counts/sec)
        sky_backgrounds = [
            exposure_map * rosat_resampled for exposure_map in exposure_maps
        ]

        # Make a dictionary to store the sky backgrounds, ra_arr, and dec_arr, t_range, and
        # t_integrate, ra_range, and dec_range, ra_res, and dec_res, and save it to a pickle file
        sky_backgrounds_dict = {
            "sky_backgrounds": sky_backgrounds,
            "ra_arr": exposure_maps_dict["ra_arr"],
            "dec_arr": exposure_maps_dict["dec_arr"],
            "t_range": t_range,
            "t_integrate": t_integrate,
            "ra_range": ra_range,
            "dec_range": dec_range,
            "ra_res": ra_res,
            "dec_res": dec_res,
        }
        if save_sky_backgrounds_file:
            # Define the folder to save the sky backgrounds to
            save_folder = (
                Path(__file__).resolve().parent.parent / "data/sky_backgrounds"
            )
            Path(save_folder).mkdir(parents=True, exist_ok=True)
            t_start = exposure_maps_dict["t_range"][0].strftime("%Y%m%d_%H%M%S")
            t_stop = exposure_maps_dict["t_range"][1].strftime("%Y%m%d_%H%M%S")
            ra_start = exposure_maps_dict["ra_range"][0]
            ra_stop = exposure_maps_dict["ra_range"][1]
            dec_start = exposure_maps_dict["dec_range"][0]
            dec_stop = exposure_maps_dict["dec_range"][1]
            ra_res = exposure_maps_dict["ra_res"]
            dec_res = exposure_maps_dict["dec_res"]
            t_integrate = int(exposure_maps_dict["t_integrate"])
            sky_backgrounds_file_name = (
                f"{save_folder}/lexi_sky_background_Tstart_{t_start}_Tstop_{t_stop}_RAstart_{ra_start}"
                f"_RAstop_{ra_stop}_RAres_{ra_res}_DECstart_{dec_start}_DECstop_{dec_stop}_DECres_"
                f"{dec_res}_Tint_{t_integrate}.npy"
            )
            # Save the sky background array to a pickle file
            with open(sky_backgrounds_file_name, "wb") as f:
                pickle.dump(sky_backgrounds_dict, f)
            print(
                f"Sky background saved to file: \033[92m {sky_backgrounds_file_name} \033[0m \n"
            )

    # If requested, save the sky background as an image
    if save_sky_backgrounds_image:  # NOT WORKING UNTIL ARRAY_TO_IMAGE IS FIXED
        for i, sky_background in enumerate(sky_backgrounds):
            array_to_image(
                sky_background,
                x_range=ra_range,
                y_range=dec_range,
                cmap="viridis",
                norm=None,
                norm_type="linear",
                aspect="auto",
                figure_title="Sky Background",
                show_colorbar=True,
                cbar_label="Counts/sec",
                cbar_orientation="vertical",
                show_axes=True,
                display=True,
                figure_size=(10, 10),
                figure_format="png",
                figure_font_size=12,
                save=True,
                save_path="figures/sky_background",
                save_name=f"sky_background_{i}",
                dpi=300,
                dark_mode=True,
            )
    # If the first element of sky_backgrounds shape is 1, then remove the first dimension
    # if np.shape(sky_backgrounds)[0] == 1:
    #     sky_backgrounds = sky_backgrounds[0]
    return sky_backgrounds_dict


def get_lexi_images(
    t_range=None,
    time_zone="UTC",
    interp_method=None,
    t_step=5,
    t_integrate=None,
    ra_range=[0, 360],
    dec_range=[-90, 90],
    ra_res=0.1,
    dec_res=0.1,
    save_exposure_map_file=False,
    save_exposure_map_image=False,
    save_sky_backgrounds_file=False,
    save_sky_backgrounds_image=False,
    save_lexi_images_file=False,
    save_lexi_images_image=False,
    background_correction_on=False,
    verbose=True,
):

    # Validate each of the inputs
    t_range_validated = validate_input("t_range", t_range)
    if t_range_validated:
        # Check if each element of t_range is a datetime object, if not then convert it to a datetime
        # object
        if isinstance(t_range[0], str):
            t_range[0] = pd.to_datetime(t_range[0])
        if isinstance(t_range[1], str):
            t_range[1] = pd.to_datetime(t_range[1])
    time_zone_validated = validate_input("time_zone", time_zone)
    if time_zone_validated:
        # Set the timezone to the t_range
        t_range[0] = t_range[0].tz_localize(time_zone)
        t_range[1] = t_range[1].tz_localize(time_zone)
        if verbose:
            print(f"Timezone set to \033[1;92m {time_zone} \033[0m \n")
    else:
        t_range[0] = t_range[0].tz_localize("UTC")
        t_range[1] = t_range[1].tz_localize("UTC")
        if verbose:
            print("Timezone set to \033[1;92m UTC \033[0m \n")
    interp_method_validated = validate_input("interp_method", interp_method)
    t_step_validated = validate_input("t_step", t_step)
    t_integrate_validated = validate_input("t_integrate", t_integrate)
    if not t_integrate_validated:
        t_integrate = (t_range[1] - t_range[0]).total_seconds()
        if verbose:
            print(
                f"\033[1;91m Integration time \033[1;92m (t_integrate) \033[1;91m not provided. Setting integration time to the time span of the spacecraft ephemeris data: \033[1;92m {t_integrate} seconds \033[0m\n"
            )
    ra_range_validated = validate_input("ra_range", ra_range)
    dec_range_validated = validate_input("dec_range", dec_range)
    ra_res_validated = validate_input("ra_res", ra_res)
    dec_res_validated = validate_input("dec_res", dec_res)
    save_exposure_map_file_validated = validate_input(
        "save_exposure_map_file", save_exposure_map_file
    )
    save_exposure_map_image_validated = validate_input(
        "save_exposure_map_image", save_exposure_map_image
    )
    save_sky_backgrounds_file_validated = validate_input(
        "save_sky_backgrounds_file", save_sky_backgrounds_file
    )
    save_sky_backgrounds_image_validated = validate_input(
        "save_sky_backgrounds_image", save_sky_backgrounds_image
    )
    save_lexi_images_file_validated = validate_input(
        "save_lexi_images_file", save_lexi_images_file
    )
    save_lexi_images_image_validated = validate_input(
        "save_lexi_images_image", save_lexi_images_image
    )
    background_correction_on_validated = validate_input(
        "background_correction_on", background_correction_on
    )
    verbose_validated = validate_input("verbose", verbose)

    # TODO: Get the actual timeseries data from the spacecraft
    # For now, try reading in sample CDF file
    photons_cdf = CDF("../data/PIT_shifted_jul08.cdf")
    key_list = photons_cdf.cdf_info().zVariables
    print(key_list)
    photons_data = {}
    for key in key_list:
        photons_data[key] = photons_cdf.varget(key)
    # Convert to dataframe
    photons = pd.DataFrame({key: photons_data[key] for key in photons_data.keys()})
    # Convert the time to a datetime objecta from UNIX time (in nanoseconds)
    photons["Epoch"] = pd.to_datetime(photons["Epoch"])

    # Set index to the time column
    photons = photons.set_index("Epoch", inplace=False)

    # Set the timezone to UTC
    photons = photons.tz_localize("UTC")

    # Check if the photons dataframe has duplicate indices
    # NOTE: Refer to the GitHub issue for more information on why we are doing this:
    # https://github.com/Lexi-BU/lexi/issues/38

    if photons.index.duplicated().any():
        # Remove the duplicate indices
        photons = photons[~photons.index.duplicated(keep="first")]
    print(
        f"Extrema: RA min {photons.ra_J2000_deg.min()}, RA max {photons.ra_J2000_deg.max()}, "
        f"DEC min {photons.dec_J2000_deg.min()}, DEC max {photons.dec_J2000_deg.max()}"
    )

    # Set up coordinate grid for lexi histograms
    ra_grid = np.arange(ra_range[0], ra_range[1], ra_res)
    dec_grid = np.arange(dec_range[0], dec_range[1], dec_res)

    # Insert one row per integration window with NaN data.
    # This ensures that even if there are periods in the data longer than t_integrate
    # in which "nothing happens", this function will still return the appropriate
    # number of lexi images, some of which empty.
    # (Besides it being more correct to return also the empty lexi images, this is
    # required in order for the images to align with the correct sky backgrounds when combined.)
    integration_filler_idcs = pd.date_range(
        t_range[0],
        t_range[1],
        freq=pd.Timedelta(t_integrate, unit="s"),
    )
    photons = photons.reindex(
        index=np.union1d(integration_filler_idcs, photons.index), method=None
    )

    # Slice to relevant time range; make groups of rows spanning t_integration
    integ_groups = photons[t_range[0] : t_range[1]].resample(
        pd.Timedelta(t_integrate, unit="s"), origin="start"
    )

    # Make as many empty lexi histograms as there are integration groups
    histograms = np.zeros((len(integ_groups), len(ra_grid), len(dec_grid)))

    for hist_idx, (_, group) in enumerate(integ_groups):
        # Loop through each photon strike and add it to the map
        for row in group.itertuples():
            try:
                ra_idx = np.nanargmin(
                    np.where(ra_grid % 360 >= row.ra_J2000_deg % 360, 1, np.nan)
                )
                dec_idx = np.nanargmin(
                    np.where(dec_grid % 360 >= row.dec_J2000_deg % 360, 1, np.nan)
                )
                histograms[hist_idx][ra_idx][dec_idx] += 1
            except ValueError:
                # photon was out of bounds on one or both axes,
                # or the row was an integration filler
                pass

    # Do background correction if requested
    if background_correction_on:
        # Get sky backgrounds
        sky_backgrounds_dict = get_sky_backgrounds(
            t_range=t_range,
            time_zone=time_zone,
            interp_method=interp_method,
            t_step=t_step,
            t_integrate=t_integrate,
            ra_range=ra_range,
            dec_range=dec_range,
            ra_res=ra_res,
            dec_res=dec_res,
            save_exposure_map_file=save_exposure_map_file,
            save_exposure_map_image=save_exposure_map_image,
            save_sky_backgrounds_file=save_sky_backgrounds_file,
            save_sky_backgrounds_image=save_sky_backgrounds_image,
            verbose=verbose,
        )
        sky_backgrounds = sky_backgrounds_dict["sky_backgrounds"]

        histograms = np.maximum(histograms - sky_backgrounds, 0)
    else:
        if ra_range_validated:
            ra_arr = np.arange(ra_range[0], ra_range[1], ra_res)
        else:
            ra_range = np.array(
                [np.nanmin(photons.ra_J2000_deg), np.nanmax(photons.ra_J2000_deg)]
            )
            ra_arr = np.arange(ra_range[0], ra_range[1], ra_res)
            if verbose:
                print(
                    f"\033[1;91m RA range \033[1;92m (ra_range) \033[1;91m not provided. Setting RA range to the range of the spacecraft ephemeris data: \033[1;92m {ra_range} \033[0m\n"
                )

        if dec_range_validated:
            dec_arr = np.arange(dec_range[0], dec_range[1], dec_res)
        else:
            dec_range = np.array(
                [np.nanmin(photons.dec_J2000_deg), np.nanmax(photons.dec_J2000_deg)]
            )
            dec_arr = np.arange(dec_range[0], dec_range[1], dec_res)
            if verbose:
                print(
                    f"\033[1;91m DEC range \033[1;92m (dec_range) \033[1;91m not provided. Setting DEC range to the range of the spacecraft ephemeris data: \033[1;92m {dec_range} \033[0m\n"
                )

    # If requested, save the histograms as images
    if save_lexi_images_file:
        for i, histogram in enumerate(histograms):
            array_to_image(
                histogram,
                x_range=ra_range,
                y_range=dec_range,
                cmap="viridis",
                norm=None,
                norm_type="log",
                aspect="auto",
                figure_title=(
                    "Background Corrected LEXI Image"
                    if background_correction_on
                    else "LEXI Image (no background correction)"
                ),
                show_colorbar=True,
                cbar_label="Counts/sec",
                cbar_orientation="vertical",
                show_axes=True,
                display=True,
                figure_size=(10, 10),
                figure_format="png",
                figure_font_size=12,
                save=True,
                save_path="figures/lexi_images",
                save_name=f"lexi_image_{i}",
                dpi=300,
                dark_mode=True,
            )

    # Define a dictionary to store the histograms, ra_arr, and dec_arr, t_range, and t_integrate,
    # ra_range, and dec_range, ra_res, and dec_res, and save it to a pickle file
    lexi_images_dict = {
        "lexi_images": histograms,
        "ra_arr": ra_arr,
        "dec_arr": dec_arr,
        "t_range": t_range,
        "t_integrate": t_integrate,
        "ra_range": ra_range,
        "dec_range": dec_range,
        "ra_res": ra_res,
        "dec_res": dec_res,
    }

    return lexi_images_dict
