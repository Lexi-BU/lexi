import numbers
import numpy as np
import pandas as pd
import pytz
import pickle
import requests
import urllib.request
from pathlib import Path
from cdflib import CDF
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import datetime

# Define a list of global variables
# Define the field of view of LEXI in degrees
LEXI_FOV = 9.1


def validate_input(key, value):
    """
    Function to validate the input parameters for LEXI functions

    Parameters
    ----------
    key : str
        The name of the input parameter

    value : any
        The value of the input parameter

    Returns
    -------
    bool
        True if the input parameter is valid, False otherwise

    Raises
    ------
    ValueError
        If the input parameter is not valid

    """
    if key == "time_range":
        if not isinstance(value, list):
            raise ValueError("time_range must be a list")
        if len(value) != 2:
            raise ValueError("time_range must have two elements")
        # Check that all elements are either one of these types: str, datetime, float, int, float
        allowed_types = (str, datetime.datetime, int, float)
        for item in value:
            if not isinstance(item, allowed_types):
                raise ValueError(
                    f"Invalid type: {type(item)} for value {item} in time_range"
                )
        # Check if the start time is less than the end time (if they are both numbers)
        if isinstance(value[0], numbers.Number) and isinstance(
            value[1], numbers.Number
        ):
            if value[0] >= value[1]:
                raise ValueError("start time must be less than end time")
        # Check if the start time is less than the end time (if they are both strings)
        if isinstance(value[0], str) and isinstance(value[1], str):
            if value[0] >= value[1]:
                raise ValueError("start time must be less than end time")

    if key == "time_zone":
        if not isinstance(value, str):
            raise ValueError("time_zone must be a string")
        if len(value) == 0:
            raise ValueError("time_zone must not be an empty string")
        # Check that the timezone is valid
        if value not in pytz.all_timezones:
            # Print a warning that the provided timezone is not valid and set it to UTC
            warnings.warn(
                f"\n \033[1;92m Timezone '{value}' \033[1;91m is not valid. Setting timezone to UTC \033[0m \n"
            )
            return False

    if key == "time_step":
        if not isinstance(value, (int, float)) or value < 0:
            warnings.warn(
                "\n \033[1;92m time_step \033[1;91m must be a positive integer or float.\033[0m \n"
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

    if key == "time_integrate":
        if not isinstance(value, numbers.Number):
            # warnings.warn(
            #     "\n \033[1;92m time_integrate \033[1;91m must be a positive number. Setting time_integrate to default value \033[0m \n"
            # )
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


def download_files_from_github(
    file_name_list,
    repo,
    folder_path,
    branch="main",
    save_dir="downloaded_data",
    verbose=False,
):
    """
    Function to download files from a GitHub repository. Eventually, this function will be removed
    and we will be able to use the `get_lexi_data` function to download the files directly from the
    CDAweb website. For now, we will use this function to download the files from the GitHub to be
    used as a placeholder until we have the real data hosted on the appropriate website.

    NOTE: In this function, we are using two folders to store and download the files. The first
    folder contains the first 950 files, and the second folder contains the remaining files. The
    reason for this is that the GitHub API only returns a maximum of 1000 files per request. If the
    folder contains more than 1000 files, then the files are split into multiple folders. The folder
    names are as follows: files_0_to_950, files_950_to_1917. The folder names are hard-coded in the
    function.

    Parameters
    ----------
    file_name_list : list
        List of file names to download

    repo : str
        Name of the GitHub repository

    folder_path : str
        Path to the folder in the GitHub repository

    branch : str, optional
        Name of the branch in the GitHub repository. Default is "main"

    save_dir : str, optional
        Directory to save the downloaded files. Default is "downloaded_data"

    verbose : bool, optional
        If True, print messages. Default is False

    Returns
    -------
    local_file_list : list
        List of local file paths

    Raises
    ------
    ValueError
        If the status code of the response is not 200

    """
    # GitHub API URL for the folder
    # NOTE: The GitHub API only returns a maximum of 1000 files per request. If the folder contains
    # more than 1000 files, then the files are split into multiple folders. The first folder contains
    # the first 950 files, and the second folder contains the remaining files. The folder names are
    # as follows: files_0_to_950, files_950_to_1917
    api_url = f"https://api.github.com/repos/{repo}/contents/{folder_path}"
    api_url_1 = api_url + "/files_0_to_950" + f"?ref={branch}"
    api_url_2 = api_url + "/files_950_to_1917" + f"?ref={branch}"

    # Fetch folder contents
    response_1 = requests.get(api_url_1)
    response_2 = requests.get(api_url_2)
    if response_1.status_code != 200:
        print(
            f"Error: Unable to access {api_url} (Status code: {response_1.status_code})"
        )
        # return
    if response_2.status_code != 200:
        print(
            f"Error: Unable to access {api_url} (Status code: {response_2.status_code})"
        )
        return

    # Parse response JSON
    files_1 = response_1.json()
    files_2 = response_2.json()
    files = files_1 + files_2

    # Ensure the save directory exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(
        f"Downloading files from \033[95m{folder_path}\033[00m on branch \033[92m{branch}\033[00m:"
    )
    print(f"A total of \033[1;92m{len(files)}\033[0m files found\n")
    print(f"Files to download: \033[1;92m{len(file_name_list)}\033[0m\n")
    local_file_list = []
    for file in files:
        if file["name"] in file_name_list:
            # Check if the file exists in the data directory, if it does then skip to the next file
            if (Path(save_dir) / file["name"]).exists():
                if verbose:
                    print(f"File already exists ==> \033[92m{file['name']}\033[00m\n")
                local_file_list.append(Path(save_dir) / file["name"])
                continue
            # Construct the raw file URL
            raw_url = file["download_url"]

            # Download the file
            print(f"Downloading \033[96m{file['name']}\033[00m...\n")
            file_response = requests.get(raw_url, stream=True)
            if file_response.status_code == 200:
                local_path = Path(save_dir) / file["name"]
                with open(local_path, "wb") as f:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(
                    f"Saved \033[96m{file['name']}\033[00m to \033[92m{local_path}\033[00m\n"
                )
                local_file_list.append(local_path)
            else:
                print(
                    f"Failed to download \033[91m{file['name']}\033[00m (Status code: {file_response.status_code})"
                )
        else:
            # print(f"Skipping {file['name']} (not in file_name_list)")
            pass
    return local_file_list


def get_lexi_data(
    time_range: list = None,
    time_zone: str = "UTC",
    time_pad: float = 300,
    data_clip: bool = True,
    verbose: bool = True,
    spc_prams: bool = False,
    return_data_type: str = "merged",
    spc_prams_kwargs: dict = None,
):
    """
    Function to get LEXI data from the CDAweb website (eventually). Currently the code is set up to
    download the data from the GitHub repository. This function will be updated to download the data
    from the CDAweb website once the data is available and hosted on the website.

    Parameters
    ----------
    time_range : list
        Time range to consider. [start time, end time]. Times can be expressed in the following
    formats:
            1. A string in the format 'YYYY-MM-DDTHH:MM:SS' (e.g. '2022-01-01T00:00:00')
            2. A datetime object
            3. A float in the format of a UNIX timestamp (e.g. 1640995200.0)

    This time range defines the time range of the ephemeris data and the time range of the LEXI data.

    Note that endpoints are inclusive (the end time is a closed interval); this is because
    the time range slicing is done with pandas, and label slicing in pandas is inclusive.

    time_zone : str, optional
        The timezone of the time range of interest. Default is "UTC"
    verbose : bool, optional
        If True, print messages. Default is True

    time_pad : float, optional
        Time padding in seconds to add to the time range value. Default is 300 seconds

    data_clip : bool, optional
        If True, clip the data to the original time range specified, else, keep the entire dataframe.
        Default is True

    spc_prams : bool, optional
        If True, get the spacecraft parameters for the same time range as LEXI data. Default is False

    return_data_type : str, optional
        Type of data to return. This parameter is only used when spc_prams is True. This defines what
        kind of dataframes to return. Valid options are:
            - 'merged': Merged LEXI and spacecraft parameters dataframes using the 'pd.merge_asof'
              function with a tolerance of 1 minute and direction of 'nearest'. Default option.
            - 'lexi': LEXI data only
            - 'spc_prams': Spacecraft parameters data only
            - 'both': Both LEXI and spacecraft parameters dataframes
            - 'all': All three dataframes
        Default is 'merged'

    spc_prams_kwargs : dict, optional
        Keyword arguments to pass to the get_spc_prams function. Default is None. If None, then the
        default values of the get_spc_prams function are used.

    Returns
    -------
    df : pandas DataFrame
        LEXI data

    df_spc_prams : pandas DataFrame
        Spacecraft parameters data

    df_merged : pandas DataFrame
        Merged LEXI and spacecraft parameters data

    Example Usage
    -------------

    >>> from lexi_bu.lexi import get_lexi_data

    >>> df_lexi = get_lexi_data(
            time_range=["2025-03-02 08:50:00", "2025-03-02 09:23:00"],
            verbose=True
        )

    Jupyter Notebook Usage:
    -----------------------

    .. jupyter-execute::

        from lexi_bu.lexi import get_lexi_data

        df_lexi = get_lexi_data(
            time_range=["2025-03-02 08:50:00", "2025-03-02 09:23:00"],
            verbose=False
        )

        print(df_lexi.head())

    """

    # Validate time_range
    time_range_validated = validate_input("time_range", time_range)

    if time_range_validated:
        # If time_range elements are strings, convert them to datetime objects
        if isinstance(time_range[0], str):
            time_range[0] = pd.to_datetime(time_range[0])
        if isinstance(time_range[1], str):
            time_range[1] = pd.to_datetime(time_range[1])
        if isinstance(time_range[0], numbers.Number):
            time_range[0] = pd.to_datetime(time_range[0], unit="s", utc=True)
        if isinstance(time_range[1], numbers.Number):
            time_range[1] = pd.to_datetime(time_range[1], unit="s", utc=True)
        # Validate time_zone, if it is not valid, set it to UTC
        if time_zone is not None:
            time_zone_validated = validate_input("time_zone", time_zone)
            if time_zone_validated:
                # Check if time_range elements are timezone aware
                if time_range[0].tzinfo is None:
                    # Set the timezone to the time_range
                    time_range[0] = time_range[0].tz_localize(time_zone)
                    time_range[1] = time_range[1].tz_localize(time_zone)
                elif time_range[0].tzinfo != time_zone:
                    # Convert the timezone to the time_range
                    time_range[0] = time_range[0].tz_convert(time_zone)
                    time_range[1] = time_range[1].tz_convert(time_zone)
                if verbose:
                    print(f"Timezone set to \033[1;92m {time_zone} \033[0m \n")
            else:
                time_range[0] = time_range[0].tz_localize("UTC")
                time_range[1] = time_range[1].tz_localize("UTC")
                if verbose:
                    print(
                        "Timezone of input time range set to \033[1;92m UTC \033[0m \n"
                    )
    # Modify the time_range based on the time_pad value
    if time_pad is not None:
        new_time_range = [
            time_range[0] - pd.Timedelta(seconds=time_pad),
            time_range[1] + pd.Timedelta(seconds=time_pad),
        ]
    else:
        new_time_range = time_range

    # Read the file_list data
    lexi_file_list_name = (
        Path(__file__).resolve().parent / ".lexi_data/all_lexi_file_list.csv"
    ).expanduser()
    df = pd.read_csv(str(lexi_file_list_name))

    # Change the time column to datetime format
    df["epoch_utc"] = pd.to_datetime(df["epoch_utc"], unit="s", utc=True)
    # Set the index to the epoch_utc column
    df.set_index("epoch_utc", inplace=True)

    # Get the file name list based on the start and end time
    file_name_list = df.loc[new_time_range[0] : new_time_range[1], "file_name"].tolist()

    repo_name = "Lexi-BU/lexi_data_analysis"
    folder_path = "data/level_1c/cdf/1.0.0"
    branch_name = "stable"
    local_file_list = download_files_from_github(
        file_name_list, repo_name, folder_path, branch_name
    )

    # For each file in the local_file_list, read the cdf file and save it to a dictionary
    lexi_data_dict_list = []
    for file in local_file_list:
        # Read the cdf file
        cdf_file = CDF(file)
        # Try to get the data from the cdf file using either of the following methods
        try:
            key_list = cdf_file.cdf_info().zVariables
            # if verbose:
            #     print(
            #         "Getting the keys from the CDF file using the \033[1;92m .zVariables \033[0m method"
            #     )
        except Exception:
            key_list = cdf_file.cdf_info()["zVariables"]
            # if verbose:
            #     print(
            #         "Getting the keys from the CDF file using the \033[1;92m ['zVariables'] \033[0m method"
            #     )

        # Create a dictionary to store the data
        lexi_data_dict = {}
        for key in key_list:
            lexi_data_dict[key] = cdf_file.varget(key)

        # Add the dictionary to the list of dictionaries
        lexi_data_dict_list.append(lexi_data_dict)

    # Loop through the list of dictionaries and save the data to a single dictionary
    lexi_data_dict = {}
    for key in lexi_data_dict_list[0].keys():
        lexi_data_dict[key] = np.concatenate(
            [d[key] for d in lexi_data_dict_list], axis=0
        )

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(lexi_data_dict)

    # Convert the Epoch_utc column to a datetime object
    df["Epoch_utc"] = pd.to_datetime(df["Epoch_unix"], unit="s", utc=True)

    # Drop the Epoch column
    df = df.drop(columns=["Epoch"])

    # Set the index to the Epoch column
    df = df.set_index("Epoch_utc", inplace=False)

    # If spc_prams is True, then get the spacecraft parameters
    if spc_prams:
        df_spc_prams = get_spc_prams(
            time_range=new_time_range,
            time_zone=time_zone,
            verbose=verbose,
            **(spc_prams_kwargs if spc_prams_kwargs else {}),
        )

        valid_return_data_types = ["merged", "lexi", "spc_prams", "both", "all"]
        if return_data_type not in valid_return_data_types:
            if verbose:
                warnings.warn(
                    f"Invalid \033[1;91m return_data_type = {return_data_type}\033[0m. Setting return_data_type to \033[1;32m 'merged' \033[0m \n"
                )
            return_data_type = "merged"
        if return_data_type in ["merged", "all"]:
            print("Merging the LEXI data with the spacecraft parameters")
            df_merged = pd.merge_asof(
                df,
                df_spc_prams,
                left_index=True,
                right_index=True,
                tolerance=pd.Timedelta("1min"),
                direction="nearest",
            )
            if return_data_type == "merged":
                if verbose:
                    print("Returning merged data")
                if data_clip:
                    df_merged = df_merged.loc[time_range[0] : time_range[1]]
                return df_merged
            elif return_data_type == "all":
                if verbose:
                    print("Returning all data")
                if data_clip:
                    df = df.loc[time_range[0] : time_range[1]]
                    df_spc_prams = df_spc_prams.loc[time_range[0] : time_range[1]]
                    df_merged = df_merged.loc[time_range[0] : time_range[1]]
                return df, df_spc_prams, df_merged
        elif return_data_type == "both":
            if verbose:
                print("Returning both LEXI and spacecraft parameters dataframes")
            if data_clip:
                df = df.loc[time_range[0] : time_range[1]]
                df_spc_prams = df_spc_prams.loc[time_range[0] : time_range[1]]
            return df, df_spc_prams
        elif return_data_type == "lexi":
            if verbose:
                print("Returning LEXI data only")
            if data_clip:
                df = df.loc[time_range[0] : time_range[1]]
            return df
        elif return_data_type == "spc_prams":
            if verbose:
                print("Returning spacecraft parameters data only")
            if data_clip:
                df_spc_prams = df_spc_prams.loc[time_range[0] : time_range[1]]
            return df_spc_prams
    else:
        if verbose:
            print("Returning LEXI data only")
        if data_clip:
            df = df.loc[time_range[0] : time_range[1]]
        return df


def get_spc_prams(
    time_range: list = None,
    time_zone: str = "UTC",
    time_step: float = 5,
    time_pad: float = 300,
    data_clip: bool = True,
    interp_method: str = "linear",
    verbose: bool = True,
    lexi_data: bool = False,
    return_data_type: str = "merged",
    lexi_data_kwargs: dict = None,
):
    """
    Function to get spacecraft ephemeris data

    Parameters
    ----------
    time_range : list
        Time range to consider. [start time, end time]. Times can be expressed in the following
    formats:
            1. A string in the format 'YYYY-MM-DDTHH:MM:SS' (e.g. '2022-01-01T00:00:00')
            2. A datetime object
            3. A float in the format of a UNIX timestamp (e.g. 1640995200.0)

    This time range defines the time range of the ephemeris data and the time range of he LEXI data.
    Note that endpoints are inclusive (the end time is a closed interval); this is because he time
    range slicing is done with pandas, and label slicing in pandas is inclusive.

    time_zone : str, optional
        The timezone of the time range of interest. Default is "UTC"

    time_step : int or float, optional
        Time step in seconds for time resolution of the look direction datum.

    time_pad : float, optional
        Time padding in seconds to add to the time range value. Default is 300 seconds

    data_clip : bool, optional
        If True, clip the data to the original time range specified, else, keep the entire dataframe.
        Default is True

    interp_method : str, optional
        Interpolation method used when upsampling/resampling ephemeris data, ROSAT data. Options:
        'linear', 'index', 'values', 'pad'. See pandas.DataFrame.interpolate documentation for
        more information. Default is 'linear'.

    verbose : bool, optional
        If True, print messages. Default is True

    lexi_data : bool, optional
        If True, get the LEXI data for the same time range as the spacecraft parameters. Default is
        False.

    return_data_type : str, optional
        Type of data to return. This parameter is only used when lexi_data is True. This defines what
        kind of dataframes to return. Valid options are:
            - 'merged': Merged LEXI and spacecraft parameters dataframes using the 'pd.merge_asof'
              function with a tolerance of 1 minute and direction of 'nearest'. Default option.
            - 'lexi': LEXI data only
            - 'spc_prams': Spacecraft parameters data only
            - 'both': Both LEXI and spacecraft parameters dataframes
            - 'all': All three dataframes
        Default is 'merged'

    lexi_data_kwargs : dict, optional
        Keyword arguments to pass to the get_lexi_data function. Default is None. If None, then the
        default values of the get_lexi_data function are used.

    Returns
    -------
    df : pandas DataFrame
        Spacecraft parameters data

    df_lexi : pandas DataFrame
        LEXI data

    df_merged : pandas DataFrame
        Merged LEXI and spacecraft parameters data

    Example Usage
    -------------

    >>> from lexi_bu.lexi import get_spc_prams

    >>> df_spc = get_spc_prams(
            time_range=["2025-03-02 08:50:00", "2025-03-02 09:23:00"],
            verbose=True
        )

    Jupyter Notebook Usage:
    -----------------------

    .. jupyter-execute::

        from lexi_bu.lexi import get_spc_prams

        df_spc = get_spc_prams(
            time_range=["2025-03-02 08:50:00", "2025-03-02 09:23:00"],
            verbose=False
        )

        print(df_spc.head())

    """
    # Validate time_range
    time_range_validated = validate_input("time_range", time_range)

    if time_range_validated:
        # If time_range elements are strings, convert them to datetime objects
        if isinstance(time_range[0], str):
            time_range[0] = pd.to_datetime(time_range[0])
        if isinstance(time_range[1], str):
            time_range[1] = pd.to_datetime(time_range[1])
        if isinstance(time_range[0], numbers.Number):
            time_range[0] = pd.to_datetime(time_range[0], unit="s", utc=True)
        if isinstance(time_range[1], numbers.Number):
            time_range[1] = pd.to_datetime(time_range[1], unit="s", utc=True)
        # Validate time_zone, if it is not valid, set it to UTC
        if time_zone is not None:
            time_zone_validated = validate_input("time_zone", time_zone)
            if time_zone_validated:
                # Check if time_range elements are timezone aware
                if time_range[0].tzinfo is None:
                    # Set the timezone to the time_range
                    time_range[0] = time_range[0].tz_localize(time_zone)
                    time_range[1] = time_range[1].tz_localize(time_zone)
                elif time_range[0].tzinfo != time_zone:
                    # Convert the timezone to the time_range
                    time_range[0] = time_range[0].tz_convert(time_zone)
                    time_range[1] = time_range[1].tz_convert(time_zone)
                if verbose:
                    print(f"Timezone set to \033[1;92m {time_zone} \033[0m \n")
            else:
                time_range[0] = time_range[0].tz_localize("UTC")
                time_range[1] = time_range[1].tz_localize("UTC")
                if verbose:
                    print(
                        "Timezone of input time range set to \033[1;92m UTC \033[0m \n"
                    )
    # Modify the time_range based on the time_pad value
    if time_pad is not None:
        new_time_range = [
            time_range[0] - pd.Timedelta(seconds=time_pad),
            time_range[1] + pd.Timedelta(seconds=time_pad),
        ]
    else:
        new_time_range = time_range

    # Validate time_step
    time_step_validated = validate_input("time_step", time_step)
    if not time_step_validated:
        time_step = 5

    # Validate interp_method
    interp_method_validated = validate_input("interp_method", interp_method)
    if not interp_method_validated:
        interp_method = "linear"

    # TODO: REMOVE ME once we start using real ephemeris data (start of chunk)
    # Get the folder location of where the current file is located
    eph_file_path = (
        Path(__file__).resolve().parent
        / ".lexi_data/20241114_LEXIAngleData_20250302Landing_rad.csv"
    )
    df = pd.read_csv(eph_file_path)
    # Convert the time coloumn from UNIX timestamp to datetime object and set the timezone to UTC
    df["epoch_utc"] = pd.to_datetime(df["epoch_utc"], unit="s")
    # Check if the time_zone is UTC, if not then set it to UTC
    if df["epoch_utc"].dt.tz is None:
        df["epoch_utc"] = df["epoch_utc"].dt.tz_localize("UTC")
        if verbose:
            print("Timezone of ephemeris file set to \033[1;92m UTC \033[0m \n")

    # Set the index to be the epoch_utc column and remove the epoch_utc column
    df = df.set_index("epoch_utc", inplace=False)

    # Rename the columns, so that they are called just "RA" and "DEC"
    try:
        for key in df.keys():
            if "ra_" in key.lower():
                df = df.rename(columns={key: "RA"})
            if "dec_" in key.lower():
                df = df.rename(columns={key: "DEC"})
    except Exception as e:
        print(e)

    # If the ephemeris data do not span the time_range, send warning
    if df.index[0] > new_time_range[0] or df.index[-1] < new_time_range[1]:
        warnings.warn(
            "Ephemeris data do not cover the full time range requested."
            "End regions will be forward/backfilled."
        )
        # Add the just the two endpoints to the index
        df = df.reindex(
            index=np.union1d(
                pd.date_range(new_time_range[0], new_time_range[1], periods=2), df.index
            )
        )

    # While slicing the dataframe, we need to make sure that the start and stop times are rounded
    # to the nearest minute.
    t_start = new_time_range[0].floor("min")
    t_stop = new_time_range[1].ceil("min")
    dfslice = df[t_start:t_stop]
    dfresamp = dfslice.resample(pd.Timedelta(time_step, unit="s"))
    dfinterp = dfresamp.interpolate(method=interp_method, limit_direction="both")

    # If lexi_data is True, then get the LEXI data
    if lexi_data:
        df_lexi = get_lexi_data(
            time_range=new_time_range,
            time_zone=time_zone,
            verbose=verbose,
            **(lexi_data_kwargs if lexi_data_kwargs else {}),
        )

        valid_return_data_types = ["merged", "lexi", "spc_prams", "both", "all"]
        if return_data_type not in valid_return_data_types:
            if verbose:
                warnings.warn(
                    f"Invalid \033[1;91m return_data_type = {return_data_type}\033[0m. Setting return_data_type to \033[1;32m 'merged' \033[0m \n"
                )
            return_data_type = "merged"
        if return_data_type in ["merged", "all"]:
            print("Merging the LEXI data with the spacecraft parameters")
            df_merged = pd.merge_asof(
                df_lexi,
                dfinterp,
                left_index=True,
                right_index=True,
                tolerance=pd.Timedelta("1min"),
                direction="nearest",
            )
            if return_data_type == "merged":
                if verbose:
                    print("Returning merged data")
                if data_clip:
                    df_merged = df_merged.loc[time_range[0] : time_range[1]]
                return df_merged
            elif return_data_type == "all":
                if verbose:
                    print("Returning all data")
                if data_clip:
                    df_lexi = df_lexi.loc[time_range[0] : time_range[1]]
                    dfinterp = dfinterp.loc[time_range[0] : time_range[1]]
                    df_merged = df_merged.loc[time_range[0] : time_range[1]]
                return df_lexi, dfinterp, df_merged
        elif return_data_type == "both":
            if verbose:
                print("Returning both LEXI and spacecraft parameters data")
            if data_clip:
                df_lexi = df_lexi.loc[time_range[0] : time_range[1]]
                dfinterp = dfinterp.loc[time_range[0] : time_range[1]]
            return df_lexi, dfinterp
        elif return_data_type == "lexi":
            if verbose:
                print("Returning LEXI data")
            if data_clip:
                df_lexi = df_lexi.loc[time_range[0] : time_range[1]]
            return df_lexi
        elif return_data_type == "spc_prams":
            if verbose:
                print("Returning spacecraft parameters data only")
            if data_clip:
                dfinterp = dfinterp.loc[time_range[0] : time_range[1]]
            return dfinterp
    else:
        if verbose:
            print("Returning spacecraft parameters data only")
        if data_clip:
            dfinterp = dfinterp.loc[time_range[0] : time_range[1]]
        return dfinterp

    # NOTE: (end of chunk that must be removed once we start using real ephemeris data) However, do
    # move the merged data part to the end of the function

    # Get the year, month, and day of the start and stop times
    start_time = time_range[0]
    stop_time = time_range[1]

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

    # If the ephemeris data do not span the time_range, send warning
    if df.index[0] > time_range[0] or df.index[-1] < time_range[1]:
        warnings.warn(
            "Ephemeris data do not cover the full time range requested."
            "End regions will be forward/backfilled."
        )
        # Add the just the two endpoints to the index
        df = df.reindex(
            index=np.union1d(
                pd.date_range(time_range[0], time_range[1], periods=2), df.index
            )
        )

    # While slicing the dataframe, we need to make sure that the start and stop times are rounded
    # to the nearest minute.
    t_start = time_range[0].floor("T")
    t_stop = time_range[1].ceil("T")
    dfslice = df[t_start:t_stop]
    dfresamp = dfslice.resample(pd.Timedelta(time_step, unit="s"))
    # Validate interp_method
    interp_method_validated = validate_input("interp_method", interp_method)
    if interp_method_validated:
        dfinterp = dfresamp.interpolate(method=interp_method, limit_direction="both")

    return dfinterp


def vignette(d: float = 0.0):
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
    time_range: list = None,
    time_zone: str = "UTC",
    interp_method: str = "linear",
    time_step: float = 5,
    ra_range: list = [0, 360],
    dec_range: list = [-90, 90],
    ra_res: float = 0.1,
    dec_res: float = 0.1,
    time_integrate: float = None,
    save_exposure_map_file: bool = False,
    save_exposure_map_image: bool = False,
    verbose: bool = True,
    force_compute: bool = False,
    array_to_image_kwargs: dict = {},
):
    """
    Function to get exposure maps

    Parameters
    ----------
    time_range : list
        Time range to consider. [start time, end time]. Times can be expressed in the following
    formats:
            1. A string in the format 'YYYY-MM-DDTHH:MM:SS' (e.g. '2022-01-01T00:00:00')
            2. A datetime object
            3. A float in the format of a UNIX timestamp (e.g. 1640995200.0)

    This time range defines the time range of the ephemeris data and the time range of he LEXI data.

    Note that endpoints are inclusive (the end time is a closed interval); this is because the time
    range slicing is done with pandas, and label slicing in pandas is inclusive.

    time_zone : str, optional
        The timezone of the time range of interest. Default is "UTC"

    interp_method : str, optional
        Interpolation method used when upsampling/resampling ephemeris data, ROSAT data.
    Options:
        'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'.

    See pandas.DataFrame.interpolate documentation for more information. Default is 'linear'.

    time_step : int or float, optional
        Time step in seconds for time resolution of the look direction datum.

    ra_range : list, optional
        Range of right ascension in degrees. If no range is provided, the range of the spacecraft
        ephemeris data is used.

    dec_range : list, optional
        Range of declination in degrees. If no range is provided, the range of the spacecraft
        ephemeris data is used.

    ra_res : float, optional
        Right ascension resolution in degrees. Default is 0.1 degrees.

    dec_res : float, optional
        Declination resolution in degrees. Default is 0.1 degrees.

    time_integrate : int or float, optional
        Integration time in seconds. If no integration time is provided, the time span of the
        `time_range` is used.

    save_exposure_map_file : bool, optional
        If True, save the exposure maps to a binary file. Default is False.

    save_exposure_map_image : bool, optional
        If True, save the exposure maps to a PNG image. Default is False.

    verbose : bool, optional
        If True, print messages. Default is True
    force_compute : bool, optional
        If True, force the computation of the exposure maps. Default is False.

    force_compute : bool, optional
        If True, force the computation of the exposure maps even if an exposure map is present in the
        default folder. Default is False.
        If True, force the computation of the exposure maps even if an exposure map is present in the
        default folder. Default is False.

    array_to_image_kwargs : dict, optional
        Keyword arguments to pass to the array_to_image function. Default is None. If None, then the
        default values of the array_to_image function are used.


    Returns
    -------
    exposure_maps_dict : dict
        Dictionary containing the following keys:
            - exposure_maps : numpy array
                Exposure maps
            - ra_arr : numpy array
                Right ascension array
            - dec_arr : numpy array
                Declination array
            - time_range : list
                Time range of the exposure maps
            - time_integrate : int or float
                Integration time in seconds of the exposure maps
            - ra_range : list
                Right ascension range of the exposure maps in degrees
            - dec_range : list
                Declination range of the exposure maps in degrees
            - ra_res : float
                Right ascension resolution of the exposure maps in degrees
            - dec_res : float
                Declination resolution of the exposure maps in degrees
            - start_time_arr : numpy array
                Start time of each exposure map
            - stop_time_arr : numpy array
                Stop time of each exposure map

    Example Usage
    -------------
    The following example shows how to get the exposure maps for a given time range:

    >>> from lexi_bu.lexi import get_exposure_maps

    >>> exposure_maps_dict = get_exposure_maps(
        time_range=["2025-03-02 08:50:00", "2025-03-02 09:23:00"],
            ra_range=[160, 230],
            dec_range=[-20, 5],
            ra_res=0.25,
            dec_res=0.25,
            time_integrate=500,
            save_exposure_map_file=True,
            save_exposure_map_image=True,
            verbose=True
        )

    Jupyter Notebook Usage:
    -----------------------

    .. jupyter-execute::

        from lexi_bu.lexi import get_exposure_maps

        exposure_maps_dict = get_exposure_maps(
            time_range=["2025-03-02 08:04:00", "2025-03-08 23:43:00"],
            ra_range=[190, 310],
            dec_range=[-33, 3],
            ra_res=0.5,
            dec_res=0.5,
            save_exposure_map_file=False,
            save_exposure_map_image=True,
            verbose=False,
            array_to_image_kwargs={"display": True}
        )

        print(exposure_maps_dict.keys())

    """

    # Validate time_step
    time_step_validated = validate_input("time_step", time_step)
    if not time_step_validated:
        time_step = 5
        if verbose:
            print(
                f"\033[1;91m Time step \033[1;92m (time_step) \033[1;91m not provided. Setting time step to \033[1;92m {time_step} seconds \033[0m\n"
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

    # Get spacecraft ephemeris data
    spc_df = get_spc_prams(
        time_range=time_range,
        time_zone=time_zone,
        interp_method=interp_method,
        verbose=verbose,
    )

    # Convert the RA and DEC columns to degrees
    spc_df["RA"] = np.degrees(spc_df["RA"])
    spc_df["DEC"] = np.degrees(spc_df["DEC"])

    # Validate time_integrate
    if time_integrate is None:
        # If time_integrate is not provided, set it to the timedelta of the provided time_range
        time_integrate = (time_range[1] - time_range[0]).total_seconds()
        if verbose:
            print(
                f"\033[1;91m Integration time \033[1;92m (time_integrate) \033[1;91m not provided. Setting integration time to the time span of the spacecraft ephemeris data: \033[1;92m {time_integrate} seconds \033[0m\n"
            )
    else:
        time_integrate_validated = validate_input("time_integrate", time_integrate)
        if not time_integrate_validated:
            time_integrate = (time_range[1] - time_range[0]).total_seconds()
            if verbose:
                print(
                    f"\033[1;91m Integration time \033[1;92m (time_integrate) \033[1;91m not provided. Setting integration time to the time span of the spacecraft ephemeris data: \033[1;92m {time_integrate} seconds \033[0m\n"
                )

    # TODO: REMOVE ME once we start using real ephemeris data
    # The sample ephemeris data uses column names "mp_ra" and "mp_dec" for look direction;
    # in the final lexi ephemeris files on CDAweb, this will be called just "RA" and "DEC".
    # Therefore...
    # spc_df["RA"] = spc_df.mp_ra
    # spc_df["DEC"] = spc_df.mp_dec
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
        # If force_compute is set to True, then go to the except block
        if force_compute:
            raise FileNotFoundError
        # Read the exposure map from a pickle file, if it exists
        # Define the folder where the exposure maps are saved
        save_folder = Path.cwd() / "data/exposure_maps"
        t_start = time_range[0].strftime("%Y%m%d_%H%M%S")
        t_stop = time_range[1].strftime("%Y%m%d_%H%M%S")
        ra_start = ra_range[0]
        ra_stop = ra_range[1]
        dec_start = dec_range[0]
        dec_stop = dec_range[1]
        ra_res = ra_res
        dec_res = dec_res
        time_integrate = int(time_integrate)
        exposure_maps_file_name = (
            f"{save_folder}/lexi_exposure_map_Tstart_{t_start}_Tstop_{t_stop}_RAstart_{ra_start}"
            f"_RAstop_{ra_stop}_RAres_{ra_res}_DECstart_{dec_start}_DECstop_{dec_stop}_DECres_"
            f"{dec_res}_Tint_{time_integrate}.npy"
        )
        # Read the exposure map from the pickle file
        exposure_maps_dict = pickle.load(open(exposure_maps_file_name, "rb"))
        if verbose:
            exposure_maps_file_dir = Path(exposure_maps_file_name).parent
            exposure_maps_file_name = Path(exposure_maps_file_name).name
            print(
                f"Exposure map loaded from file \033[1;94m {exposure_maps_file_dir}/\033[1;92m{exposure_maps_file_name} \033[0m\n"
            )
    except FileNotFoundError:
        print("Exposure map not found, computing now. This may take a while \n")

        # Slice to relevant time range; make groups of rows spanning time_integratetion
        integ_groups = spc_df[time_range[0] : time_range[1]].resample(
            pd.Timedelta(time_integrate, unit="s"), origin="start"
        )

        # Get the min and max times of each group
        start_time_arr = []
        stop_time_arr = []
        for _, group in integ_groups:
            start_time_arr.append(group.index.min())
            stop_time_arr.append(group.index.max())
        # Make as many empty exposure maps as there are integration groups
        exposure_maps = np.zeros((len(integ_groups), len(ra_arr), len(dec_arr)))

        # Loop through each pointing step and add the exposure to the map
        # Wrap-proofing: First make everything [0,360)...
        ra_grid_mod = ra_grid  # % 360
        dec_grid_mod = dec_grid  # % 90

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
                exposure_delt = np.where(
                    (r < LEXI_FOV * 0.5), vignette(r) * time_step, 0
                )
                # Add the delta to the full map
                exposure_maps[map_idx] += exposure_delt
                if verbose:
                    print(
                        f"Computing exposure map ==> \x1b[1;32;255m {np.round(map_idx/len(integ_groups)*100, 6)}\x1b[0m % complete",
                        end="\r",
                    )
        t_start = time_range[0].strftime("%Y%m%d_%H%M%S")
        t_stop = time_range[1].strftime("%Y%m%d_%H%M%S")
        ra_start = ra_range[0]
        ra_stop = ra_range[1]
        dec_start = dec_range[0]
        dec_stop = dec_range[1]
        ra_res = ra_res
        dec_res = dec_res
        time_integrate = int(time_integrate)

        # Define a dictionary to store the exposure maps, ra_arr, and dec_arr, time_range, and time_integrate,
        # ra_range, and dec_range, ra_res, and dec_res
        exposure_maps_dict = {
            "exposure_maps": exposure_maps,
            "ra_arr": ra_arr,
            "dec_arr": dec_arr,
            "time_range": time_range,
            "time_integrate": time_integrate,
            "ra_range": ra_range,
            "dec_range": dec_range,
            "ra_res": ra_res,
            "dec_res": dec_res,
            "start_time_arr": start_time_arr,
            "stop_time_arr": stop_time_arr,
        }
        if save_exposure_map_file:
            # Define the folder to save the exposure maps to
            save_folder = Path.cwd() / "data/exposure_maps"
            Path(save_folder).mkdir(parents=True, exist_ok=True)

            exposure_maps_file_name = (
                f"{save_folder}/lexi_exposure_map_Tstart_{t_start}_Tstop_{t_stop}_RAstart_{ra_start}"
                f"_RAstop_{ra_stop}_RAres_{ra_res}_DECstart_{dec_start}_DECstop_{dec_stop}_DECres_"
                f"{dec_res}_Tint_{time_integrate}.npy"
            )

            # Save the exposure map array to a pickle file
            with open(exposure_maps_file_name, "wb") as f:
                pickle.dump(exposure_maps_dict, f)
            if verbose:
                exposure_maps_file_dir = Path(exposure_maps_file_name).parent
                exposure_maps_file_name = Path(exposure_maps_file_name).name
                print(
                    f"Exposure map saved to file \033[1;94m {exposure_maps_file_dir}/\033[1;92m{exposure_maps_file_name} \033[0m\n"
                )

    # If requested, save the exposure maps as images
    if save_exposure_map_image:
        if verbose:
            print("Saving exposure maps as images")
        # Check if the following keys are present in the array_to_image_kwargs dictionary, if not
        # then add them:
        # - x_range
        # - y_range
        if "x_range" not in array_to_image_kwargs:
            array_to_image_kwargs["x_range"] = ra_range
        if "y_range" not in array_to_image_kwargs:
            array_to_image_kwargs["y_range"] = dec_range
        for i, exposure in enumerate(exposure_maps_dict["exposure_maps"]):
            array_to_image(
                input_array=exposure,
                key="exposure_maps",
                start_time=exposure_maps_dict["start_time_arr"][i],
                stop_time=exposure_maps_dict["stop_time_arr"][i],
                ra_res=ra_res,
                dec_res=dec_res,
                time_integrate=exposure_maps_dict["time_integrate"],
                **(array_to_image_kwargs if array_to_image_kwargs else {}),
            )

    return exposure_maps_dict


def get_sky_backgrounds(
    time_range: list = None,
    time_zone: str = "UTC",
    interp_method: str = "linear",
    time_step: float = 5,
    time_integrate: float = None,
    ra_range: list = [0, 360],
    dec_range: list = [-90, 90],
    ra_res: float = 0.1,
    dec_res: float = 0.1,
    save_exposure_map_file: bool = False,
    save_exposure_map_image: bool = False,
    save_sky_backgrounds_file: bool = False,
    save_sky_backgrounds_image: bool = False,
    verbose: bool = True,
    force_compute: bool = False,
    array_to_image_kwargs: dict = {},
):
    """
    Function to get sky backgrounds for a given time range and RA/DEC range and resolution using
    ROSAT data and exposure maps

    Parameters
    ----------
    time_range : list
        Time range to consider. [start time, end time]. Times can be expressed in the following
    formats:
            1. A string in the format 'YYYY-MM-DDTHH:MM:SS' (e.g. '2022-01-01T00:00:00')
            2. A datetime object
            3. A float in the format of a UNIX timestamp (e.g. 1640995200.0)

    This time range defines the time range of the ephemeris data and the time range of the LEXI data.

    Note that endpoints are inclusive (the end time is a closed interval); this is because the time
    range slicing is done with pandas, and label slicing in pandas is inclusive.

    time_zone : str, optional
        The timezone of the time range of interest. Default is "UTC"

    interp_method : str, optional
        Interpolation method used when upsampling/resampling ephemeris data, ROSAT data.
    Options:
        'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'.

    See pandas.DataFrame.interpolate documentation for more information. Default is 'linear'.

    time_step : int or float, optional
        Time step in seconds for time resolution of the look direction datum.

    time_integrate : int or float, optional
        Integration time in seconds. If no integration time is provided, the time span of the
        `time_range` is used.

    ra_range : list, optional
        Range of right ascension in degrees. If no range is provided, the range of the spacecraft
        ephemeris data is used.

    dec_range : list, optional
        Range of declination in degrees. If no range is provided, the range of the spacecraft
        ephemeris data is used.

    ra_res : float, optional
        Right ascension resolution in degrees. Default is 0.1 degrees.

    dec_res : float, optional
        Declination resolution in degrees. Default is 0.1 degrees.

    save_exposure_map_file : bool, optional
        If True, save the exposure maps to a binary file. Default is False.

    save_exposure_map_image : bool, optional
        If True, save the exposure maps to a PNG image. Default is False.

    save_sky_backgrounds_file : bool, optional
        If True, save the sky backgrounds to a binary file. Default is False.

    save_sky_backgrounds_image : bool, optional
        If True, save the sky backgrounds to a PNG image. Default is False.

    verbose : bool, optional
        If True, print messages. Default is True
    force_compute : bool, optional
        If True, force the computation of the sky backgrounds. Default is False.

    force_compute : bool, optional
        If True, force the computation of the sky backgrounds even if a skybackground data is present
        in the default folder. Default is False.

    array_to_image_kwargs : dict, optional
        Keyword arguments to pass to the array_to_image function. Default is None. If None, then the
        default values of the array_to_image function are used.

    Returns
    -------
    sky_backgrounds_dict : dict
        Dictionary containing the following keys:
            - sky_backgrounds : numpy array
                Sky backgrounds
            - ra_arr : numpy array
                Right ascension array
            - dec_arr : numpy array
                Declination array
            - time_range : list
                Time range of the sky backgrounds
            - time_integrate : int or float
                Integration time in seconds of the sky backgrounds
            - ra_range : list
                Right ascension range of the sky backgrounds in degrees
            - dec_range : list
                Declination range of the sky backgrounds in degrees
            - ra_res : float
                Right ascension resolution of the sky backgrounds in degrees
            - dec_res : float
                Declination resolution of the sky backgrounds in degrees
            - start_time_arr : numpy array
                Start time of each sky background
            - stop_time_arr : numpy array
                Stop time of each sky background

    Example Usage
    -------------
    The following example demonstrates how to get sky backgrounds for a given time range and RA/DEC
    range and resolution using ROSAT data and exposure maps:

    >>> from lexi_bu.lexi import get_sky_backgrounds

    >>> sky_background_dict = get_sky_backgrounds(
            time_range=["2025-03-02 08:50:00", "2025-03-02 09:23:00"],
            ra_range=[160, 230],
            dec_range=[-20, 5],
            ra_res=0.5,
            dec_res=0.5,
            time_integrate=500,
            save_exposure_map_file=True,
            save_sky_backgrounds_file=True,
            save_exposure_map_image=True,
            save_sky_backgrounds_image=True,
            verbose=True
        )

    Jupyter Notebook Usage:
    -----------------------

    .. jupyter-execute::

        from lexi_bu.lexi import get_sky_backgrounds

        sky_background_dict = get_sky_backgrounds(
            time_range=["2025-03-02 08:04:00", "2025-03-08 23:43:00"],
            ra_range=[190, 310],
            dec_range=[-33, 3],
            ra_res=0.5,
            dec_res=0.5,
            save_exposure_map_file=False,
            save_sky_backgrounds_file=False,
            save_exposure_map_image=True,
            save_sky_backgrounds_image=False,
            verbose=False,
            array_to_image_kwargs={"display": True}
        )

        print(sky_background_dict.keys())

    """

    # Get exposure maps
    exposure_maps_dict = get_exposure_maps(
        time_range=time_range,
        time_zone=time_zone,
        interp_method=interp_method,
        time_step=time_step,
        ra_range=ra_range,
        dec_range=dec_range,
        ra_res=ra_res,
        dec_res=dec_res,
        time_integrate=time_integrate,
        save_exposure_map_file=save_exposure_map_file,
        save_exposure_map_image=save_exposure_map_image,
        verbose=verbose,
        array_to_image_kwargs=array_to_image_kwargs,
    )
    exposure_maps = exposure_maps_dict["exposure_maps"]

    try:
        # If force_compute is set to True, then go to the except block
        if force_compute:
            raise FileNotFoundError
        # Read the sky background from a pickle file, if it exists
        # Define the folder where the sky backgrounds are saved
        save_folder = Path.cwd() / "data/sky_backgrounds"
        t_start = exposure_maps_dict["time_range"][0].strftime("%Y%m%d_%H%M%S")
        t_stop = exposure_maps_dict["time_range"][1].strftime("%Y%m%d_%H%M%S")
        ra_start = exposure_maps_dict["ra_range"][0]
        ra_stop = exposure_maps_dict["ra_range"][1]
        dec_start = exposure_maps_dict["dec_range"][0]
        dec_stop = exposure_maps_dict["dec_range"][1]
        ra_res = exposure_maps_dict["ra_res"]
        dec_res = exposure_maps_dict["dec_res"]
        time_integrate = int(exposure_maps_dict["time_integrate"])
        start_time_arr = exposure_maps_dict["start_time_arr"]
        stop_time_arr = exposure_maps_dict["stop_time_arr"]

        sky_backgrounds_file_name = (
            f"{save_folder}/lexi_sky_background_Tstart_{t_start}_Tstop_{t_stop}_RAstart_{ra_start}"
            f"_RAstop_{ra_stop}_RAres_{ra_res}_DECstart_{dec_start}_DECstop_{dec_stop}_DECres_"
            f"{dec_res}_Tint_{time_integrate}.npy"
        )
        # Read the sky background from the pickle file
        sky_backgrounds_dict = pickle.load(open(sky_backgrounds_file_name, "rb"))
        if verbose:
            sky_backgrounds_file_dir = Path(sky_backgrounds_file_name).parent
            sky_backgrounds_file_name = Path(sky_backgrounds_file_name).name
            print(
                f"Sky background loaded from file \033[1;94m {sky_backgrounds_file_dir}/\033[1;92m{sky_backgrounds_file_name} \033[0m\n"
            )
    except FileNotFoundError:
        print("Sky background not found, computing now. This may take a while \n")

        # Get ROSAT background
        # NOTE: Ultimately KKip is supposed to provide this file and we will have it saved somewhere static.
        # For now, this is Cadin's sample xray data:
        rosat_data = (
            Path(__file__).resolve().parent / ".lexi_data/sample_xray_background.csv"
        )
        rosat_df = pd.read_csv(rosat_data, header=None)
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

        # Convert the sky_backgrounds to a numpy array
        sky_backgrounds = np.array(
            [np.array(sky_background) for sky_background in sky_backgrounds]
        )

        # Make a dictionary to store the sky backgrounds, ra_arr, and dec_arr, time_range, and
        # time_integrate, ra_range, and dec_range, ra_res, and dec_res, and save it to a pickle file
        sky_backgrounds_dict = {
            "sky_backgrounds": sky_backgrounds,
            "ra_arr": exposure_maps_dict["ra_arr"],
            "dec_arr": exposure_maps_dict["dec_arr"],
            "time_range": time_range,
            "time_integrate": time_integrate,
            "ra_range": ra_range,
            "dec_range": dec_range,
            "ra_res": ra_res,
            "dec_res": dec_res,
            "start_time_arr": start_time_arr,
            "stop_time_arr": stop_time_arr,
        }
        if save_sky_backgrounds_file:
            # Define the folder to save the sky backgrounds to
            save_folder = Path.cwd() / "data/sky_backgrounds"
            Path(save_folder).mkdir(parents=True, exist_ok=True)
            t_start = exposure_maps_dict["time_range"][0].strftime("%Y%m%d_%H%M%S")
            t_stop = exposure_maps_dict["time_range"][1].strftime("%Y%m%d_%H%M%S")
            ra_start = exposure_maps_dict["ra_range"][0]
            ra_stop = exposure_maps_dict["ra_range"][1]
            dec_start = exposure_maps_dict["dec_range"][0]
            dec_stop = exposure_maps_dict["dec_range"][1]
            ra_res = exposure_maps_dict["ra_res"]
            dec_res = exposure_maps_dict["dec_res"]
            time_integrate = int(exposure_maps_dict["time_integrate"])
            sky_backgrounds_file_name = (
                f"{save_folder}/lexi_sky_background_Tstart_{t_start}_Tstop_{t_stop}_RAstart_{ra_start}"
                f"_RAstop_{ra_stop}_RAres_{ra_res}_DECstart_{dec_start}_DECstop_{dec_stop}_DECres_"
                f"{dec_res}_Tint_{time_integrate}.npy"
            )
            # Save the sky background array to a pickle file
            with open(sky_backgrounds_file_name, "wb") as f:
                pickle.dump(sky_backgrounds_dict, f)
            if verbose:
                sky_backgrounds_file_dir = Path(sky_backgrounds_file_name).parent
                sky_backgrounds_file_name = Path(sky_backgrounds_file_name).name
                print(
                    f"Sky background saved to file: \033[1;94m {sky_backgrounds_file_dir}/\033[1;92m{sky_backgrounds_file_name} \033[0m\n"
                )

    # If requested, save the sky background as an image
    if save_sky_backgrounds_image:
        if verbose:
            print("Saving sky backgrounds as images")
        # Check if the following keys are present in the array_to_image_kwargs dictionary, if not
        # then add them:
        # - x_range
        # - y_range
        if "x_range" not in array_to_image_kwargs:
            array_to_image_kwargs["x_range"] = ra_range
        if "y_range" not in array_to_image_kwargs:
            array_to_image_kwargs["y_range"] = dec_range
        for i, sky_background in enumerate(sky_backgrounds_dict["sky_backgrounds"]):
            array_to_image(
                input_array=sky_background,
                key="sky_backgrounds",
                start_time=sky_backgrounds_dict["start_time_arr"][i],
                stop_time=sky_backgrounds_dict["stop_time_arr"][i],
                ra_res=ra_res,
                dec_res=dec_res,
                time_integrate=sky_backgrounds_dict["time_integrate"],
                **(array_to_image_kwargs if array_to_image_kwargs else {}),
            )
    # If the first element of sky_backgrounds shape is 1, then remove the first dimension
    # if np.shape(sky_backgrounds)[0] == 1:
    #     sky_backgrounds = sky_backgrounds[0]
    return sky_backgrounds_dict


def get_lexi_images(
    time_range: list = None,
    time_zone: str = "UTC",
    interp_method: str = "linear",
    time_step: float = 5,
    ra_range: list = [0, 360],
    dec_range: list = [-90, 90],
    ra_res: float = 0.1,
    dec_res: float = 0.1,
    time_integrate: float = None,
    background_correction_on: bool = True,
    save_exposure_map_file: bool = False,
    save_exposure_map_image: bool = False,
    save_sky_backgrounds_file: bool = False,
    save_sky_backgrounds_image: bool = False,
    save_lexi_images: bool = False,
    verbose: bool = True,
    array_to_image_kwargs: dict = {},
):
    """
    Function to get LEXI images for a given time range and RA/DEC range and resolution using
    ROSAT data and exposure maps

    Parameters
    ----------
    time_range : list
        Time range to consider. [start time, end time]. Times can be expressed in the following
    formats:
            1. A string in the format 'YYYY-MM-DDTHH:MM:SS' (e.g. '2022-01-01T00:00:00')
            2. A datetime object
            3. A float in the format of a UNIX timestamp (e.g. 1640995200.0)

    This time range defines the time range of the ephemeris data and the time range of the LEXI data.

    Note that endpoints are inclusive (the end time is a closed interval); this is because the time
    range slicing is done with pandas, and label slicing in pandas is inclusive.

    time_zone : str, optional
        The timezone of the time range of interest. Default is "UTC"

    interp_method : str, optional
        Interpolation method used when upsampling/resampling ephemeris data, ROSAT data.
    Options:
        'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'.

    See pandas.DataFrame.interpolate documentation for more information. Default is 'linear'.

    time_step : int or float, optional
        Time step in seconds for time resolution of the look direction datum.

    time_integrate : int or float, optional
        Integration time in seconds. If no integration time is provided, the time span of the
        `time_range` is used.

    ra_range : list, optional
        Range of right ascension in degrees. If no range is provided, the range of the spacecraft
        ephemeris data is used.

    dec_range : list, optional
        Range of declination in degrees. If no range is provided, the range of the spacecraft
        ephemeris data is used.

    ra_res : float, optional
        Right ascension resolution in degrees. Default is 0.1 degrees.

    dec_res : float, optional
        Declination resolution in degrees. Default is 0.1 degrees.

    background_correction_on : bool, optional
        If True, apply the background correction to the LEXI images. Default is True.

    save_exposure_map_file : bool, optional
        If True, save the exposure maps to a binary file. Default is False.

    save_exposure_map_image : bool, optional
        If True, save the exposure maps to a PNG image. Default is False.

    save_sky_backgrounds_file : bool, optional
        If True, save the sky backgrounds to a binary file. Default is False.

    save_sky_backgrounds_image : bool, optional
        If True, save the sky backgrounds to a PNG image. Default is False.

    save_lexi_images : bool, optional
        If True, save the LEXI images to a PNG file. Default is False.

    verbose : bool, optional
        If True, print messages. Default is True

    array_to_image_kwargs : dict, optional
        Keyword arguments to pass to the array_to_image function. Default is None. If None, then the
        default values of the array_to_image function are used.

    Returns
    -------
    lexi_images_dict : dict
        Dictionary containing the following keys:
            - lexi_images : numpy array
                LEXI images
            - ra_arr : numpy array
                Right ascension array
            - dec_arr : numpy array
                Declination array
            - time_range : list
                Time range of the LEXI images
            - time_integrate : int or float
                Integration time in seconds of the LEXI images
            - ra_range : list
                Right ascension range of the LEXI images in degrees
            - dec_range : list
                Declination range of the LEXI images in degrees
            - ra_res : float
                Right ascension resolution of the LEXI images in degrees
            - dec_res : float
                Declination resolution of the LEXI images in degrees

    Example Usage
    -------------
    The following example shows how to get LEXI images for a given time range and RA/DEC range and
    resolution

    >>> from lexi_bu.lexi import get_lexi_images

    >>> lexi_images_dict = get_lexi_images(
            time_range=["2025-03-02 08:04:00", "2025-03-08 23:43:00"],
            ra_range=[190, 310],
            dec_range=[-33, 3],
            ra_res=0.5,
            dec_res=0.5,
            # time_integrate=500,
            background_correction_on=True,
            save_exposure_map_file=True,
            save_sky_backgrounds_file=True,
            save_exposure_map_image=True,
            save_sky_backgrounds_image=True,
            save_lexi_images=True,
            verbose=True
        )

    Jupyter Notebook Usage:
    -----------------------

    .. jupyter-execute::

        from lexi_bu.lexi import get_lexi_images

        lexi_images_dict = get_lexi_images(
            time_range=["2025-03-04 08:53:41", "2025-03-04 09:23:41"],
            ra_range=[220, 240],
            dec_range=[-30, -15],
            ra_res=1,
            dec_res=1,
            background_correction_on=False,
            save_exposure_map_file=False,
            save_sky_backgrounds_file=False,
            save_exposure_map_image=False,
            save_sky_backgrounds_image=False,
            save_lexi_images=True,
            verbose=False,
            array_to_image_kwargs={"display": True}
        )

        print(lexi_images_dict.keys())
    """

    # Validate each of the inputs
    time_range_validated = validate_input("time_range", time_range)

    if time_range_validated:
        # If time_range elements are strings, convert them to datetime objects
        if isinstance(time_range[0], str):
            time_range[0] = pd.to_datetime(time_range[0])
        if isinstance(time_range[1], str):
            time_range[1] = pd.to_datetime(time_range[1])
        if isinstance(time_range[0], numbers.Number):
            time_range[0] = pd.to_datetime(time_range[0], unit="s", utc=True)
        if isinstance(time_range[1], numbers.Number):
            time_range[1] = pd.to_datetime(time_range[1], unit="s", utc=True)
        # Validate time_zone, if it is not valid, set it to UTC
        if time_zone is not None:
            time_zone_validated = validate_input("time_zone", time_zone)
            if time_zone_validated:
                # Check if time_range elements are timezone aware
                if time_range[0].tzinfo is None:
                    # Set the timezone to the time_range
                    time_range[0] = time_range[0].tz_localize(time_zone)
                    time_range[1] = time_range[1].tz_localize(time_zone)
                elif time_range[0].tzinfo != time_zone:
                    # Convert the timezone to the time_range
                    time_range[0] = time_range[0].tz_convert(time_zone)
                    time_range[1] = time_range[1].tz_convert(time_zone)
                if verbose:
                    print(f"Timezone set to \033[1;92m {time_zone} \033[0m \n")
            else:
                time_range[0] = time_range[0].tz_localize("UTC")
                time_range[1] = time_range[1].tz_localize("UTC")
                if verbose:
                    print(
                        "Timezone of input time range set to \033[1;92m UTC \033[0m \n"
                    )

    interp_method_validated = validate_input("interp_method", interp_method)
    if not interp_method_validated:
        interp_method = "linear"
        if verbose:
            print(
                f"\033[1;91m Interpolation method \033[1;92m (interp_method) \033[1;91m not provided. Setting interpolation method to \033[1;92m {interp_method} \033[0m\n"
            )

    _ = validate_input("time_step", time_step)
    time_integrate_validated = validate_input("time_integrate", time_integrate)
    if not time_integrate_validated:
        time_integrate = (time_range[1] - time_range[0]).total_seconds()
        if verbose:
            print(
                f"\033[1;91m Integration time \033[1;92m (time_integrate) \033[1;91m not provided. Setting integration time to the time span of the spacecraft ephemeris data: \033[1;92m {time_integrate} seconds \033[0m\n"
            )
    ra_range_validated = validate_input("ra_range", ra_range)
    dec_range_validated = validate_input("dec_range", dec_range)
    _ = validate_input("ra_res", ra_res)
    _ = validate_input("dec_res", dec_res)

    # TODO: Get the actual timeseries data from the spacecraft
    # NOTE: This will require a function that will take the limits of the time range and return the
    # data in the time range as a dataframe. Potentially, that will add a keyword to the main
    # function called `data_dir` or something similar. This function will be implemented in the future.
    # For now, try reading in sample CDF file
    # Get the location of the LEXI data

    # Download and read the LEXI data in a pandas dataframe
    # NOTE: This is a sample LEXI data file. The actual LEXI data will be downloaded from the LEXI
    # database.
    photons = get_lexi_data(time_range=time_range, verbose=verbose)

    # Check if the photons dataframe has duplicate indices
    # NOTE: Refer to the GitHub issue for more information on why we are doing this:
    # https://github.com/Lexi-BU/lexi/issues/38

    if photons.index.duplicated().any():
        # Remove the duplicate indices
        photons = photons[~photons.index.duplicated(keep="first")]

    # Set up coordinate grid for lexi histograms
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

    # Insert one row per integration window with NaN data.
    # This ensures that even if there are periods in the data longer than time_integrate
    # in which "nothing happens", this function will still return the appropriate
    # number of lexi images, some of which empty.
    # (Besides it being more correct to return also the empty lexi images, this is
    # required in order for the images to align with the correct sky backgrounds when combined.)
    integration_filler_idcs = pd.date_range(
        time_range[0],
        time_range[1],
        freq=pd.Timedelta(time_integrate, unit="s"),
    )
    photons = photons.reindex(
        index=np.union1d(integration_filler_idcs, photons.index), method=None
    )

    # Slice to relevant time range; make groups of rows spanning time_integratetion
    integ_groups = photons[time_range[0] : time_range[1]].resample(
        pd.Timedelta(time_integrate, unit="s"), origin="start"
    )

    start_time_arr = []
    stop_time_arr = []
    for _, group in integ_groups:
        start_time_val = group.index.min()
        stop_time_val = group.index.max()
        # If start and stop times are the same, then skip this group
        if start_time_val == stop_time_val:
            continue
        else:
            start_time_arr.append(group.index.min())
            stop_time_arr.append(group.index.max())

    # Make as many empty lexi histograms as there are integration groups
    histograms = np.zeros((len(start_time_arr), len(ra_arr), len(dec_arr)))

    for hist_idx, (_, group) in enumerate(integ_groups):
        # Loop through each photon strike and add it to the map
        for row in group.itertuples():
            try:
                ra_idx = np.nanargmin(
                    np.where(ra_arr % 360 >= row.ra_J2000_deg % 360, 1, np.nan)
                )
                dec_idx = np.nanargmin(
                    np.where(dec_arr % 90 >= row.dec_J2000_deg % 90, 1, np.nan)
                )
                histograms[hist_idx][ra_idx][dec_idx] += 1
            except Exception:
                # photon was out of bounds on one or both axes,
                # or the row was an integration filler
                pass

    # Do background correction if requested
    if background_correction_on:
        # Get sky backgrounds
        sky_backgrounds_dict = get_sky_backgrounds(
            time_range=time_range,
            time_zone=time_zone,
            interp_method=interp_method,
            time_step=time_step,
            time_integrate=time_integrate,
            ra_range=ra_range,
            dec_range=dec_range,
            ra_res=ra_res,
            dec_res=dec_res,
            save_exposure_map_file=save_exposure_map_file,
            save_exposure_map_image=save_exposure_map_image,
            save_sky_backgrounds_file=save_sky_backgrounds_file,
            save_sky_backgrounds_image=save_sky_backgrounds_image,
            verbose=verbose,
            array_to_image_kwargs=array_to_image_kwargs,
        )
        # NOTE: Chnage the factor of 0.001 in the line below to the actual factor that should be
        # (ideallly 1)
        sky_backgrounds = 0.01 * sky_backgrounds_dict["sky_backgrounds"]
        histograms = np.maximum(histograms - sky_backgrounds, 0)

    # Define a dictionary to store the histograms, ra_arr, and dec_arr, time_range, and time_integrate,
    # ra_range, and dec_range, ra_res, and dec_res, and save it to a pickle file
    lexi_images_dict = {
        "lexi_images": histograms,
        "ra_arr": ra_arr,
        "dec_arr": dec_arr,
        "time_range": time_range,
        "time_integrate": time_integrate,
        "ra_range": ra_range,
        "dec_range": dec_range,
        "ra_res": ra_res,
        "dec_res": dec_res,
        "start_time_arr": start_time_arr,
        "stop_time_arr": stop_time_arr,
    }
    print(start_time_arr)
    # If requested, save the histograms as images
    if save_lexi_images:
        if verbose:
            print("Saving LEXI images as images")
        # Check if the following keys are present in the array_to_image_kwargs dictionary, if not
        # then add them:
        # - x_range
        # - y_range
        if "x_range" not in array_to_image_kwargs:
            array_to_image_kwargs["x_range"] = ra_range
        if "y_range" not in array_to_image_kwargs:
            array_to_image_kwargs["y_range"] = dec_range
        for i, histogram in enumerate(lexi_images_dict["lexi_images"]):
            array_to_image(
                input_array=histogram,
                key=f"lexi_images/background_corrected_{background_correction_on}",
                start_time=start_time_arr[i],
                stop_time=stop_time_arr[i],
                ra_res=ra_res,
                dec_res=dec_res,
                time_integrate=lexi_images_dict["time_integrate"],
                figure_title=(
                    "Background Corrected LEXI Image"
                    if background_correction_on
                    else "LEXI Image (no background correction)"
                ),
                **(array_to_image_kwargs if array_to_image_kwargs else {}),
            )

    return lexi_images_dict


def array_to_image(
    input_array: np.ndarray = None,
    key: str = None,
    x_range: list = None,
    y_range: list = None,
    x_lim: list = None,
    y_lim: list = None,
    start_time: pd.Timestamp = None,
    stop_time: pd.Timestamp = None,
    ra_res: float = None,
    dec_res: float = None,
    time_integrate: float = None,
    cmap: str = "viridis",
    cmin: float = None,
    v_min: float = None,
    v_max: float = None,
    norm: mpl.colors.LogNorm = mpl.colors.LogNorm(),
    norm_type: str = "log",
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
    verbose: bool = False,
):
    """
    Convert a 2D array to an image.

    Parameters
    ----------
    ra_res : float, optional
        Right ascension resolution in degrees. Default is None.

    dec_res : float, optional
        Declination resolution in degrees. Default is None.

    time_integrate : int or float, optional
        Integration time in seconds. Default is None.

    input_array : np.ndarray
        2D array to convert to an image.

    x_range : list, optional
        Range of the x-axis.  Default is None.

    y_range : list, optional
        Range of the y-axis.  Default is None.

    x_lim : list, optional
        Limits of the x-axis.  Default is None.

    y_lim : list, optional
        Limits of the y-axis.  Default is None.

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

    Example Usage
    -------------
    The following demonstrates how to use the `array_to_image` function:

    .. jupyter-execute::

        from lexi_bu.lexi import array_to_image
        import numpy as np
        import matplotlib.pyplot as plt

        # Create a 2D array
        input_array = np.random.rand(100, 100)

        # Print the shape of the input array
        # The shape should be (100, 100)
        print(input_array.shape)

    """
    # Try to use latex rendering
    # plt.rc("text", usetex=False)
    # try:
    #     plt.rc("text", usetex=True)
    #     plt.rc("font", family="serif")
    #     plt.rc("font", size=figure_font_size)
    # except Exception:
    #     pass

    # Check whether input_array is a 2D array
    if len(input_array.shape) != 2:
        raise ValueError("input_array must be a 2D array")

    # Mask the input array if cmin is specified
    if cmin is not None:
        input_array = np.ma.masked_less(input_array, cmin)

    # Check whether x_range is a list
    if x_range is not None:
        if not isinstance(x_range, (list, tuple, np.ndarray)):
            raise ValueError("x_range must be a list, tuple, or numpy array")
        if len(x_range) != 2:
            raise ValueError("x_range must be a list of length 2")
    else:
        x_range = x_range

    # Check whether y_range is a list
    if y_range is not None:
        if not isinstance(y_range, (list, tuple, np.ndarray)):
            raise ValueError("y_range must be a list, tuple, or numpy array")
        if len(y_range) != 2:
            raise ValueError("y_range must be a list of length 2")
    else:
        y_range = y_range

    if dark_mode:
        plt.style.use("dark_background")
        facecolor = "k"
        edgecolor = "w"
        textcolor = "w"
    else:
        plt.style.use("default")
        facecolor = "w"
        edgecolor = "k"
        textcolor = "k"

    if v_min is None and v_max is None:
        array_min = np.nanmin(input_array)
        array_max = np.nanmax(input_array)
        if array_min == array_max:
            # In theory, could be a real instance of a perfectly flat map;
            # probably, just an integration window with no photons.
            print(
                f"Encountered map where array min {array_min} == array max {array_max}. "
                "Plotting a range of \u00B1 1."
            )
            array_min -= 1
            array_max += 1

        if norm_type == "linear":
            v_min = 0.9 * array_min
            v_max = 1.1 * array_max
            norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
        elif norm_type == "log":
            if array_min <= 0:
                v_min = 1e-5
            else:
                v_min = array_min
            if array_max <= 0:
                v_max = 1e-1
            else:
                v_max = array_max
            norm = mpl.colors.LogNorm(vmin=v_min, vmax=v_max)
    elif v_min is not None and v_max is not None:
        if norm_type == "linear":
            norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
        elif norm_type == "log":
            if v_min <= 0:
                v_min = 1e-5
            if v_max <= 0:
                v_max = 1e-1
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
        interpolation=None,
    )

    # Set the x and y limits
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    # Turn on the grid
    ax.grid(True, color="k", alpha=0.5, linestyle="-")
    # Turn on minor grid
    ax.minorticks_on()
    # Set the tick label size
    ax.tick_params(labelsize=0.8 * figure_font_size)
    # Add start and stop time as text to the plot
    ax.text(
        0.05,
        0.93,
        f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=0.8 * figure_font_size,
        color=textcolor,
    )
    ax.text(
        0.05,
        0.92,
        f"Stop Time: {stop_time.strftime('%Y-%m-%d %H:%M:%S')}",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=0.8 * figure_font_size,
        color=textcolor,
    )
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
            save_path = Path.cwd() / f"figures/{key}"
            if verbose:
                print("save_path not provided. Saving figure to default lcoation \n")
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if save_name is None or save_name == "default":
            start_time_str = start_time.strftime("%Y%m%d_%H%M%S")
            stop_time_str = stop_time.strftime("%Y%m%d_%H%M%S")
            save_name = (
                f"{key.split('/')[0]}_Tstart_{start_time_str}_Tstop_{stop_time_str}_RAstart_{x_range[0]}"
                f"_RAstop_{x_range[1]}_RAres_{ra_res}_DECstart_{y_range[0]}_DECstop_{y_range[1]}_DECres_"
                f"{dec_res}_Tint_{time_integrate}"
            )

        save_name = save_name + "." + figure_format
        plt.savefig(
            f"{save_path}/{save_name}",
            format=figure_format,
            dpi=dpi,
            bbox_inches="tight",
        )
        if verbose:
            print(
                f"Saved figure to ==> \033[1;94m {save_path}/\033[1;92m{save_name} \033[0m \n"
            )

    if display:
        plt.show()

    # Close the figure
    # plt.close()

    return fig, ax
