import numpy as np
import pandas as pd
import urllib.request
from pathlib import Path
from spacepy import pycdf
import matplotlib as mpl
import matplotlib.pyplot as plt

import warnings


def validate_input(
        *,
        t_range,
        t_step,
        ra_range,
        dec_range,
        ra_res,
        dec_res,
        interp_method,
        background_correction_on,
        save_df,
        filename,
        filetype,
        save_exposure_maps,
        save_sky_backgrounds,
        save_lexi_images,
):
    # Validate t_range
    if not isinstance(t_range, list):
        raise ValueError("t_range must be a list")
    if len(t_range) != 2:
        raise ValueError("t_range must have two elements")
    if not all(isinstance(x, str) for x in t_range):
        raise ValueError("t_range elements must be strings")

    # Validate t_step
    if not isinstance(t_step, int):
        raise ValueError("t_step must be an integer")
    if t_step <= 0:
        raise ValueError("t_step must be greater than 0")

    # Validate ra_range
    if not isinstance(ra_range, list):
        raise ValueError("ra_range must be a list")
    if len(ra_range) != 2:
        raise ValueError("ra_range must have two elements")
    if not all(isinstance(x, (int, float)) for x in ra_range):
        raise ValueError("ra_range elements must be integers or floats")

    # Validate dec_range
    if not isinstance(dec_range, list):
        raise ValueError("dec_range must be a list")
    if len(dec_range) != 2:
        raise ValueError("dec_range must have two elements")
    if not all(isinstance(x, (int, float)) for x in dec_range):
        raise ValueError("dec_range elements must be integers or floats")

    # Validate ra_res
    if not isinstance(ra_res, (int, float)):
        raise ValueError("ra_res must be an integer or float")
    if ra_res <= 0:
        raise ValueError("ra_res must be greater than 0")

    # Validate dec_res
    if not isinstance(dec_res, (int, float)):
        raise ValueError("dec_res must be an integer or float")
    if dec_res <= 0:
        raise ValueError("dec_res must be greater than 0")

    # Validate interp_method
    if not isinstance(interp_method, str):
        raise ValueError("interp_method must be a string")
    if interp_method not in ["linear", "nearest", "zero", "slinear", "quadratic", "cubic"]:
        raise ValueError("interp_method must be one of 'linear', 'nearest', 'zero', 'slinear', 'quadratic', or 'cubic'")

    # Validate background_correction_on
    if not isinstance(background_correction_on, bool):
        raise ValueError("background_correction_on must be a boolean")

    # Validate save_df
    if not isinstance(save_df, bool):
        raise ValueError("save_df must be a boolean")
    
    # Validate filename 
    if not isinstance(filename, str):
        raise ValueError("filename must be a string")
    if len(filename) == 0:
        raise ValueError("filename must not be an empty string")

    # Validate filetype
    if not isinstance(filetype, str):
        raise ValueError("filetype must be a string")
    if len(filetype) == 0:
        raise ValueError("filetype must not be an empty string")
    if filetype not in ["pkl", "csv"]:
        raise ValueError("filetype must be one of 'pkl' or 'csv'")
    
    # Validate save_exposure_maps
    if not isinstance(save_exposure_maps, bool):
        raise ValueError("save_exposure_maps must be a boolean")
    
    # Validate save_sky_backgrounds
    if not isinstance(save_sky_backgrounds, bool):
        raise ValueError("save_sky_backgrounds must be a boolean")

    # Validate save_lexi_images
    if not isinstance(save_lexi_images, bool):
        raise ValueError("save_lexi_images must be a boolean")

    return True


def get_spc_prams(
        t_range,
        t_step,
        ra_range,
        dec_range,
        ra_res,
        dec_res,
        interp_method,
        background_correction_on,
        save_df,
        filename,
        filetype,
        save_exposure_maps,
        save_sky_backgrounds,
        save_lexi_images,
):
    # TODO: REMOVE ME once we start using real ephemeris data
    df = pd.read_csv("data/sample_lexi_pointing_ephem_edited.csv")
    # Convert the epoch_utc column to a datetime object
    df["epoch_utc"] = pd.to_datetime(df["epoch_utc"])
    # Set the index to be the epoch_utc column and remove the epoch_utc column
    df = df.set_index("epoch_utc", inplace=False)

    if df.index[0] > t_range[0] or df.index[-1] < t_range[1]:
        warnings.warn(
            "Ephemeris data do not cover the full time range requested."
            "End regions will be forward/backfilled."
        )
        # Add the just the two endpoints to the index
        df = df.reindex(
            index=np.union1d(
                pd.date_range(t_range[0], t_range[1], periods=2), df.index
            )
        )

    dfslice = df[t_range[0] : t_range[1]]
    dfresamp = dfslice.resample(pd.Timedelta(t_step, unit="s"))
    dfinterp = dfresamp.interpolate(
        method=interp_method, limit_direction="both"
    )
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
                elif (
                    (year == stop_year)
                    and (month == stop_month)
                    and (day > stop_day)
                ):
                    continue
                else:
                    file_list.append(filename)

    # Download the files in the file list to the data/ephemeris directory
    data_dir = Path(__file__).resolve().parent.parent / "data/ephemeris"
    # If the data directory doesn't exist, then create it
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # Download the files in the file list to the data/ephemeris directory
    for file in file_list:
        urllib.request.urlretrieve(CDA_LINK + file, data_dir / file)

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

    # If the ephemeris data do not span the t_range, send warning
    if df.index[0] > t_range[0] or df.index[-1] < t_range[1]:
        warnings.warn(
            "Ephemeris data do not cover the full time range requested."
            "End regions will be forward/backfilled."
        )
        # Add the just the two endpoints to the index
        df = df.reindex(
            index=np.union1d(
                pd.date_range(t_range[0], t_range[1], periods=2), df.index
            )
        )

    # Slice, resample, interpolate
    dfslice = df[t_range[0] : t_range[1]]
    dfresamp = dfslice.resample(pd.Timedelta(t_step, unit="s"))
    dfinterp = dfresamp.interpolate(
        method=interp_method, limit_direction="both"
    )

    return dfinterp


df = get_spc_prams(
    t_range=["2024-07-08T21:43:41", "2024-07-08T21:47:48"],
    t_step=5,
    ra_range=[290, 360],
    dec_range=[290, 360],
    ra_res=4,
    dec_res=3,
    interp_method="linear",
    background_correction_on=False,
    save_df=True,
    filename="test_data/LEXI_pointing_ephem_highres",
    filetype="pkl",
    save_exposure_maps=True,
    save_sky_backgrounds=True,
    save_lexi_images=True,
)