import csv
import datetime
import os
import logging
import struct
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytz
import pickle

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")

# Check if the log folder exists, if not then create it
Path("../log").mkdir(parents=True, exist_ok=True)

file_handler = logging.FileHandler("../log/lxi_read_binary_data.log")
file_handler.setFormatter(formatter)

# stream_handler = logging.StreamHandler()

logger.addHandler(file_handler)

# Tha packet format of the science and housekeeping packets
packet_format_sci = ">II4H"

# double precision format for time stamp from pit
packet_format_pit = ">d"


sync_lxi = b"\xfe\x6b\x28\x40"

sync_pit = b"\x54\x53"

volts_per_count = 4.5126 / 65536  # volts per increment of digitization


class sci_packet_cls(NamedTuple):
    """
    Class for the science packet.
    The code unpacks the science packet into a named tuple. Based on the packet format, each packet
    is unpacked into following parameters:
    - Date: time of the packet as received from the PIT
    - timestamp: int (32 bit)
    - IsCommanded: bool (1 bit)
    - voltage channel1: float (16 bit)
    - voltage channel2: float (16 bit)
    - voltage channel3: float (16 bit)
    - voltage channel4: float (16 bit)

    TimeStamp is the time stamp of the packet in seconds.
    IsCommand tells you if the packet was commanded.
    Voltages 1 to 4 are the voltages of corresponding different channels.
    """

    Date: float
    is_commanded: bool
    timestamp: int
    channel1: float
    channel2: float
    channel3: float
    channel4: float

    @classmethod
    def from_bytes(cls, bytes_: bytes):
        structure_time = struct.unpack(">d", bytes_[2:10])
        structure = struct.unpack(packet_format_sci, bytes_[12:])
        return cls(
            Date=structure_time[0],
            is_commanded=bool(
                structure[1] & 0x40000000
            ),  # mask to test for commanded event type
            timestamp=structure[1] & 0x3FFFFFFF,  # mask for getting all timestamp bits
            channel1=structure[2] * volts_per_count,
            channel2=structure[3] * volts_per_count,
            channel3=structure[4] * volts_per_count,
            channel4=structure[5] * volts_per_count,
        )


def read_binary_data_sci(
    in_file_name=None,
    save_file_name="../data/processed/sci/output_sci.csv",
    number_of_decimals=6,
):
    """
    Reads science packet of the binary data from a file and saves it to a csv file.

    Parameters
    ----------
    in_file_name : str
        Name of the input file. Default is None.
    save_file_name : str
        Name of the output file. Default is "output_sci.csv".
    number_of_decimals : int
        Number of decimals to save. Default is 6.

    Raises
    ------
    FileNotFoundError :
        If the input file does not exist.
    TypeError :
        If the name of the input file or input directory is not a string. Or if the number of
        decimals is not an integer.
    Returns
    -------
        df : pandas.DataFrame
            DataFrame of the science packet.
        save_file_name : str
            Name of the output file.
    """
    if in_file_name is None:
        in_file_name = (
            "../data/raw_data/2022_03_03_1030_LEXI_raw_2100_newMCP_copper.txt"
        )

    # Check if the file exists, if does not exist raise an error
    if not Path(in_file_name).is_file():
        raise FileNotFoundError("The file " + in_file_name + " does not exist.")
    # Check if the file name and folder name are strings, if not then raise an error
    if not isinstance(in_file_name, str):
        raise TypeError("The file name must be a string.")

    # Check the number of decimals to save
    if not isinstance(number_of_decimals, int):
        raise TypeError("The number of decimals to save must be an integer.")

    input_file_name = in_file_name

    # Get the creation date of the file in UTC and local time
    creation_date_utc = datetime.datetime.utcfromtimestamp(
        os.path.getctime(input_file_name)
    )
    creation_date_local = datetime.datetime.fromtimestamp(
        os.path.getctime(input_file_name)
    )

    with open(input_file_name, "rb") as file:
        raw = file.read()

    index = 0
    packets = []

    # Check if the "file_name" has payload in its name or not. If it has payload in its name, then
    # use the sci_packet_cls else use sci_packet_cls_gsfc
    if "payload" in in_file_name:
        while index < len(raw) - 28:
            if (
                raw[index : index + 2] == sync_pit
                and raw[index + 12 : index + 16] == sync_lxi
            ):
                packets.append(sci_packet_cls.from_bytes(raw[index : index + 28]))
                index += 28
                continue
            elif (raw[index : index + 2] == sync_pit) and (
                raw[index + 12 : index + 16] != sync_lxi
            ):
                # Ignore the last packet
                if index >= len(raw) - 28 - 16:
                    # NOTE: This is a temporary fix. The last packet is ignored because the last
                    # packet often isn't complete. Need to find a better solution.
                    index += 28
                    continue
                # Check if sync_lxi is present in the next 16 bytes
                if sync_lxi in raw[index + 12 : index + 28] and index + 28 < len(raw):
                    # Find the index of sync_lxi
                    index_sync = (
                        index + 12 + raw[index + 12 : index + 28].index(sync_lxi)
                    )
                    # Reorder the packet
                    new_packet = (
                        raw[index + 28 : index + 12 + 28]
                        + raw[index_sync : index + 28]
                        + raw[index + 12 + 28 : index_sync + 28]
                    )
                    # Check if the packet length is 28
                    if len(new_packet) != 28:
                        # If the index + 28 is greater than the length of the raw data, then break
                        if index + 28 > len(raw):
                            break
                    packets.append(sci_packet_cls.from_bytes(new_packet))
                    index += 28
                    continue
                # Check if raw[index - 3:index] + raw[index+12:index+13] == sync_lxi
                elif raw[index - 3 : index] + raw[index + 12 : index + 13] == sync_lxi:
                    # Reorder the packet
                    new_packet = (
                        raw[index : index + 12]
                        + raw[index - 3 : index]
                        + raw[index + 12 : index + 25]
                    )
                    packets.append(sci_packet_cls.from_bytes(new_packet))
                    index += 28
                    continue
                # Check if raw[index - 2:index] + raw[index+12:index+14] == sync_lxi
                elif raw[index - 2 : index] + raw[index + 12 : index + 14] == sync_lxi:
                    # Reorder the packet
                    new_packet = (
                        raw[index : index + 12]
                        + raw[index - 2 : index]
                        + raw[index + 13 : index + 26]
                    )
                    packets.append(sci_packet_cls.from_bytes(new_packet))
                    index += 28
                    continue
                # Check if raw[index - 1:index] + raw[index+12:index+15] == sync_lxi
                elif raw[index - 1 : index] + raw[index + 12 : index + 15] == sync_lxi:
                    # Reorder the packet
                    new_packet = (
                        raw[index : index + 12]
                        + raw[index - 1 : index]
                        + raw[index + 14 : index + 27]
                    )
                    packets.append(sci_packet_cls.from_bytes(new_packet))
                    index += 28
                    continue
                index += 28
                continue
            index += 28
    else:
        # Raise FileNameError mentioning that the file name does not contain proper keywords
        raise FileNotFoundError("The file name does not contain the keyword 'payload'.")

    # Split the file name in a folder and a file name
    # Format filenames and folder names for the different operating systems
    output_file_name = (
        os.path.basename(os.path.normpath(in_file_name)).split(".")[0]
        + "_sci_output.csv"
    )
    output_folder_name = (
        os.path.dirname(os.path.normpath(in_file_name)) + "/processed_data/sci"
    )
    save_file_name = output_folder_name + "/" + output_file_name

    # Check if the save folder exists, if not then create it
    if not Path(output_folder_name).exists():
        Path(output_folder_name).mkdir(parents=True, exist_ok=True)

    if "payload" in in_file_name:
        with open(save_file_name, "w", newline="") as file:
            dict_writer = csv.DictWriter(
                file,
                fieldnames=(
                    "Date",
                    "TimeStamp",
                    "IsCommanded",
                    "Channel1",
                    "Channel2",
                    "Channel3",
                    "Channel4",
                ),
            )
            dict_writer.writeheader()
            try:
                dict_writer.writerows(
                    {
                        "Date": datetime.datetime.utcfromtimestamp(sci_packet_cls.Date),
                        "TimeStamp": sci_packet_cls.timestamp / 1e3,
                        "IsCommanded": sci_packet_cls.is_commanded,
                        "Channel1": np.round(
                            sci_packet_cls.channel1, decimals=number_of_decimals
                        ),
                        "Channel2": np.round(
                            sci_packet_cls.channel2, decimals=number_of_decimals
                        ),
                        "Channel3": np.round(
                            sci_packet_cls.channel3, decimals=number_of_decimals
                        ),
                        "Channel4": np.round(
                            sci_packet_cls.channel4, decimals=number_of_decimals
                        ),
                    }
                    for sci_packet_cls in packets
                )
            except Exception as e:
                # Print the exception in red color
                print(f"\n\033[91m{e}\033[00m\n")
                print(
                    f"Number of science packets found in the file \033[96m {in_file_name}\033[0m "
                    f"is just \033[91m {len(packets)}\033[0m. \n \033[96m Check the datafile to "
                    "see if the datafile has proper data.\033[0m \n "
                )
    else:
        # Raise FileNameError mentioning that the file name does not contain proper keyword
        raise FileNotFoundError("The file name does not contain the keyword 'payload'.")

    # Read the saved file data in a dataframe
    df = pd.read_csv(save_file_name)

    # Convert the date column to datetime
    print(df["Date"])
    df["Date"] = pd.to_datetime(df["Date"])

    # Set index to the date
    df.set_index("Date", inplace=False)

    # For each row, get the time difference between the current row and the last row
    try:
        time_diff = df["Date"].iloc[:] - df["Date"].iloc[-1]
    except Exception:
        # Set time difference to 0
        time_diff = datetime.timedelta(seconds=0)
        logger.warning(
            f"For the science data, the time difference between the current row and the last row is 0 for {input_file_name}."
        )
    try:
        # For each time difference, get the total number of seconds as an array
        time_diff_seconds = time_diff.dt.total_seconds().values
    except Exception:
        # Set time difference to 0 seconds
        time_diff_seconds = 0
        logger.warning(
            f"For the scicence data, the time difference between the current row and the last row is 0 for {input_file_name}."
        )

    # Add utc_time and local_time column to the dataframe as NaNs
    df["utc_time"] = np.nan
    df["local_time"] = np.nan
    # For each row, set the utc_time and local_time as sum of created_date_utc and time_diff_seconds
    df["utc_time"] = creation_date_utc + pd.to_timedelta(time_diff_seconds, unit="s")
    df["local_time"] = creation_date_local + pd.to_timedelta(
        time_diff_seconds, unit="s"
    )

    # Save the dataframe to a csv file
    df.to_csv(save_file_name, index=False)

    return df, save_file_name


def lin_correction(
    x,
    y,
    M_inv=np.array([[0.98678, 0.16204], [0.11385, 0.993497]]),
    b=np.array([0.00195, 0.56355]),
):
    """
    Function to apply nonlinearity correction to MCP position x/y data
    # TODO: Add correct M_inv matrix and the offsets
    """
    x_lin = (x * M_inv[0, 0] + y * M_inv[0, 1]) - b[0]
    y_lin = x * M_inv[1, 0] + y * M_inv[1, 1]

    return x_lin, y_lin


def non_lin_correction(
    x,
    y,
):
    """
    Function to apply nonlinearity correction to MCP position x/y data. The model to apply the
    nonlinearity correction is a Gaussian Process model trained on the data from the LEXI massk
    testing. The kernel used is Matern with length scale = 5 and nu = 2.5.

    Parameters
    ----------
    x : numpy.ndarray
        x position data.
    y : numpy.ndarray
        y position data.

    Returns
    -------
    x_nln : numpy.ndarray
        x position data after applying nonlinearity correction.
    y_nln : numpy.ndarray
        y position data after applying nonlinearity correction.
    """
    gp_model_file_name = "../data/gp_models/gp_data_3.0_10_0.0_0.8_4_Matern(length_scale=5, nu=2.5).pickle"

    # Get the gp_model from the pickle file
    with open(gp_model_file_name, "rb") as f:
        gp_model = pickle.load(f)

    # Close the pickle file
    f.close()

    xy_coord = np.array([x, y]).T
    delta_xy, sigma = gp_model.predict(xy_coord, return_std=True)

    corrected_xy = xy_coord - delta_xy
    x_nln = corrected_xy[:, 0]
    y_nln = corrected_xy[:, 1]

    return x_nln, y_nln


def volt_to_mcp(x, y):
    """
    Function to convert voltage coordinates to MCP coordinates
    """
    x_mcp = (x - 0.544) * 78.55
    y_mcp = (y - 0.564) * 78.55

    return x_mcp, y_mcp


def compute_position(v1=None, v2=None, n_bins=401, bin_min=0, bin_max=4):
    """
    The function computes the position of the particle in the xy-plane. The ratios to compute
    both the x and y position are taken from Dennis' code. The code computes the offset of the
    four voltages as measured by LEXI. We then subtract the offset from the four voltages to get
    the shifted voltages and then compute the position of the particle based on the shifted
    voltages.

    Parameters
    ----------
    v1 : float
        Voltage of the first channel. Default is None.
    v2 : float
        Voltage of the second channel. Default is None.
    n_bins : int
        Number of bins to compute the position. Default is 401.
    bin_min : float
        Minimum value of the bin. Default is 0.
    bin_max : float
        Maximum value of the bin. Default is 4.

    Returns
    -------
    particle_pos : float
        position of the particle along one of the axis. Whether it gives x or y position depends
        on which voltages were provided. For example, if v1 and v3 were provided, then the x
        position is returned. Else if v4 and v2 were provided, then the y position is returned.
        It is important to note that the order of the voltages is important.
    v1_shift: float
        Offset corrected voltage of the first channel.
    v2_shift: float
        Offset corrected voltage of the second channel.
    """
    bin_size = (bin_max - bin_min) / (n_bins - 1)

    # make 1-D histogram of all 4 channels
    hist_v1 = np.histogram(v1, bins=n_bins, range=(bin_min, bin_max))
    hist_v2 = np.histogram(v2, bins=n_bins, range=(bin_min, bin_max))

    xx = bin_min + bin_size * np.arange(n_bins)

    # Find the index where the histogram is the maximum
    # NOTE/TODO: I don't quite understand why the offset is computed this way. Need to talk to
    # Dennis about this and get an engineering/physics reason for it.
    max_index_v1 = np.argmax(hist_v1[0][0 : int(n_bins / 2)])
    max_index_v2 = np.argmax(hist_v2[0][0 : int(n_bins / 2)])

    z1_min = 1000 * xx[max_index_v1]

    z2_min = 1000 * xx[max_index_v2]

    n1_z = z1_min / 1000
    n2_z = z2_min / 1000

    v1_shift = v1 - n1_z
    v2_shift = v2 - n2_z

    particle_pos = v2_shift / (v2_shift + v1_shift)

    return particle_pos, v1_shift, v2_shift


def read_csv_sci(file_val=None, t_start=None, t_end=None):
    """
    Reads a csv file and returns a pandas dataframe for the selected time range along with x and
    y-coordinates.

    Parameters
    ----------
    file_val:str
        Path to the input file. Default is None.
    t_start:float
        Start time of the data. Default is None.
    t_end : float
        End time of the data. Default is None.
    """

    df = pd.read_csv(file_val, index_col=False)

    # Check all the keys and find out which one has the word "time" in it
    for key in df.keys():
        if "time" in key.lower():
            time_col = key
            break
    # Rename the time column to TimeStamp
    df.rename(columns={time_col: "TimeStamp"}, inplace=True)

    # Convert the Date column from string to datetime in utc
    try:
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
    except Exception:
        # Convert timestamp to datetime and set it to Date
        df["Date"] = pd.to_datetime(df["TimeStamp"], unit="s", utc=True)

    # Set the index to the time column
    df.set_index("Date", inplace=True)
    # Sort the dataframe by timestamp
    df = df.sort_index()

    if t_start is None:
        t_start = df.index.min()
    else:
        # Check if t_start and t_end are datetime objects. If not, convert them to datetime
        if not isinstance(t_start, datetime.datetime):
            t_start = datetime.datetime.strptime(t_start, "%Y-%m-%d %H:%M:%S")
        if not isinstance(t_end, datetime.datetime):
            t_end = datetime.datetime.strptime(t_end, "%Y-%m-%d %H:%M:%S")
        # Check if t_start is time-zone aware. If not, make it time-zone aware
        if t_start.tzinfo is None:
            t_start = t_start.replace(tzinfo=pytz.utc)
    if t_end is None:
        t_end = df.index.max()
    else:
        # Check if t_end is time-zone aware. If not, make it time-zone aware
        if t_end.tzinfo is None:
            t_end = t_end.replace(tzinfo=pytz.utc)

    x, v1_shift, v3_shift = compute_position(
        v1=df["Channel1"], v2=df["Channel3"], n_bins=401, bin_min=0, bin_max=4
    )

    y, v4_shift, v2_shift = compute_position(
        v1=df["Channel4"], v2=df["Channel2"], n_bins=401, bin_min=0, bin_max=4
    )

    # Correct for the non-linearity in the positions
    x_lin, y_lin = lin_correction(x, y)

    # Get the x,y value in mcp units
    x_mcp, y_mcp = volt_to_mcp(x, y)
    x_mcp_lin, y_mcp_lin = volt_to_mcp(x_lin, y_lin)

    # Add the x-coordinate to the dataframe
    df.loc[:, "x_val"] = x
    df.loc[:, "x_val_lin"] = x_lin
    df.loc[:, "x_mcp"] = x_mcp
    df.loc[:, "x_mcp_lin"] = x_mcp_lin
    df.loc[:, "v1_shift"] = v1_shift
    df.loc[:, "v3_shift"] = v3_shift

    # Add the y-coordinate to the dataframe
    df.loc[:, "y_val"] = y
    df.loc[:, "y_val_lin"] = y_lin
    df.loc[:, "y_mcp"] = y_mcp
    df.loc[:, "y_mcp_lin"] = y_mcp_lin
    df.loc[:, "v4_shift"] = v4_shift
    df.loc[:, "v2_shift"] = v2_shift

    return df


def read_binary_file(file_val=None, t_start=None, t_end=None, multiple_files=False):
    """
    Reads the binary file using functions saved in the file "lxi_read_binary_data.py" and returns
    a pandas dataframe for the selected time range along with x and y-coordinates.

    Parameters
    ----------
    file_val : str
        Path to the input file. Default is None.
    t_start : float
        Start time of the data. Default is None.
    t_end : float
        End time of the data. Default is None.

    Returns
    -------
    df_sci : pandas.DataFrame
        The Science dataframe for the entire time range in the file.
    file_name_sci : str
        The name of the Science file.
    """

    if multiple_files is False:
        # Read the science data
        df_sci, file_name_sci = read_binary_data_sci(
            in_file_name=file_val, save_file_name=None, number_of_decimals=6
        )

    else:
        # If only one of t_start and t_end is None, raise an error
        if (t_start is None and t_end is not None) or (
            t_start is not None and t_end is None
        ):
            raise ValueError(
                "when multiple_files is True, both t_start and t_end must either be"
                f"None or a valid time value. The values provided are t_start ="
                f"{t_start} and t_end = {t_end}."
            )
        # If both t_start and t_end are None, raise a warning stating that the times are set to none
        if t_start is None and t_end is None:
            print(
                "\n \x1b[1;31;255m WARNING: Both the start and end time values provided were None"
                "setting both of them to None \x1b[0m"
            )
            t_start = None
            t_end = None

        if t_start is not None and t_end is not None:
            # Convert t_start and t_end from string to datetime in UTC timezone
            t_start = pd.to_datetime(t_start, utc=True)
            t_end = pd.to_datetime(t_end, utc=True)
            try:
                # Convert t_start and t_end from string to unix time in seconds in UTC timezone
                t_start_unix = t_start.timestamp()
                t_end_unix = t_end.timestamp()
            except Exception:
                t_start_unix = None
                t_end_unix = None

        # Define a list in which the dataframes will be stored
        df_sci_list = []
        file_name_sci_list = []

        # Make sure that file_val is a directory
        if not os.path.isdir(file_val):
            raise ValueError("file_val should be a directory.")

        # Get the names of all the files in the directory with*.dat or *.txt extension
        file_list = np.sort(
            [
                os.path.join(file_val, f)
                for f in os.listdir(file_val)
                if f.endswith((".dat", ".txt"))
            ]
        )

        # If file list is empty, raise an error and exit
        if len(file_list) == 0:
            raise ValueError("No files found in the directory.")
        else:
            print(
                f"Found total \x1b[1;32;255m {len(file_list)} \x1b[0m files in the directory."
            )

        if t_start_unix is not None and t_end_unix is not None:
            # In file_list, select only those files which are within the time range
            file_list = [
                file_name
                for file_name in file_list
                if t_start_unix
                <= float(os.path.basename(file_name).split("_")[2])
                <= t_end_unix
            ]
            print(
                f"Found \x1b[1;32;255m {len(file_list)} \x1b[0m files in the time range "
                f"\x1b[1;32;255m {t_start.strftime('%Y-%m-%d %H:%M:%S')} \x1b[0m to "
                f"\x1b[1;32;255m {t_end.strftime('%Y-%m-%d %H:%M:%S')}\x1b[0m"
            )

        # Loop through all the files
        for file_name in file_list:
            # Print in cyan color that file number is being read from the directory conatining total
            # number of files
            print(
                f"\n Reading file \x1b[1;36;255m {file_list.index(file_name) + 1} \x1b[0m of "
                f"total \x1b[1;36;255m {len(file_list)} \x1b[0m files."
            )
            # Read the science data
            df_sci, file_name_sci = read_binary_data_sci(
                in_file_name=file_name, save_file_name=None, number_of_decimals=6
            )

            # Append the dataframes to the list
            df_sci_list.append(df_sci)
            file_name_sci_list.append(file_name_sci)

        # Concatenate all the dataframes
        df_sci = pd.concat(df_sci_list)

        # Set file_names_sci to dates of first and last files
        save_dir = os.path.dirname(file_val)
        # If save_dir does not exist, create it
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Get the file name
        file_name_sci = (
            save_dir
            + "/processed_data/sci/"
            + file_name_sci_list[0].split("/")[-1].split(".")[0].split("_")[1]
            + "_"
            + file_name_sci_list[0].split("/")[-1].split(".")[0].split("_")[0]
            + "_"
            + file_name_sci_list[0].split("/")[-1].split(".")[0].split("_")[2]
            + "_"
            + file_name_sci_list[0].split("/")[-1].split(".")[0].split("_")[3]
            + "_"
            + file_name_sci_list[-1].split("/")[-1].split(".")[0].split("_")[-4]
            + "_"
            + file_name_sci_list[-1].split("/")[-1].split(".")[0].split("_")[-3]
            + "_sci_output.csv"
        )

        print(f"The Science File name =\x1b[1;32;255m{file_name_sci} \x1b[0m \n")
        # Save the dataframe to a csv file
        df_sci.to_csv(file_name_sci, index=False)

        print(
            f"Saved the dataframes to csv files. \n"
            f"The Science File name =\x1b[1;32;255m{file_name_sci} \x1b[0m \n"
        )
    # Replace index with timestamp
    df_sci.set_index("Date", inplace=True)

    # Sort the dataframe by timestamp
    df_sci = df_sci.sort_index()

    if t_start is None:
        t_start = df_sci.index.min()
        print(f"t_start is None. Setting t_start = {t_start}")
    if t_end is None:
        t_end = df_sci.index.max()

    # Select only those where "IsCommanded" is False
    df_sci = df_sci[df_sci["IsCommanded"] == False]

    # Select only rows where all channels are greater than 0
    df_sci = df_sci[
        (df_sci["Channel1"] > 0)
        & (df_sci["Channel2"] > 0)
        & (df_sci["Channel3"] > 0)
        & (df_sci["Channel4"] > 0)
    ]

    # For the entire dataframes, compute the x and y-coordinates and the shift in the voltages
    x, v1_shift, v3_shift = compute_position(
        v1=df_sci["Channel1"], v2=df_sci["Channel3"], n_bins=401, bin_min=0, bin_max=4
    )

    df_sci.loc[:, "x_val"] = x
    df_sci.loc[:, "v1_shift"] = v1_shift
    df_sci.loc[:, "v3_shift"] = v3_shift

    y, v4_shift, v2_shift = compute_position(
        v1=df_sci["Channel4"], v2=df_sci["Channel2"], n_bins=401, bin_min=0, bin_max=4
    )

    # Add the y-coordinate to the dataframe
    df_sci.loc[:, "y_val"] = y
    df_sci.loc[:, "v4_shift"] = v4_shift
    df_sci.loc[:, "v2_shift"] = v2_shift

    # Correct for the non-linearity in the positions using linear correction
    # NOTE: Linear correction must be applied to the data when the data is in the
    # voltage/dimensionless units.
    x_lin, y_lin = lin_correction(x, y)

    # Get the x,y value in mcp units
    x_mcp, y_mcp = volt_to_mcp(x, y)
    x_mcp_lin, y_mcp_lin = volt_to_mcp(x_lin, y_lin)

    # Correct for the non-linearity in the positions using non-linear correction model
    # NOTE: The non-linear correction is only applied on the mcp coordinates after linear correction
    # has been applied.
    x_mcp_nln, y_mcp_nln = non_lin_correction(x_mcp_lin, y_mcp_lin)

    # Add the x-coordinate to the dataframe
    df_sci.loc[:, "x_val_lin"] = x_lin
    df_sci.loc[:, "x_mcp"] = x_mcp
    df_sci.loc[:, "x_mcp_lin"] = x_mcp_lin
    df_sci.loc[:, "x_mcp_nln"] = x_mcp_nln

    # Add the y-coordinate to the dataframe
    df_sci.loc[:, "y_val_lin"] = y_lin
    df_sci.loc[:, "y_mcp"] = y_mcp
    df_sci.loc[:, "y_mcp_lin"] = y_mcp_lin
    df_sci.loc[:, "y_mcp_nln"] = y_mcp_nln

    # Get the file name
    file_name_sci = (
        save_dir
        + "/processed_data/sci/l1b/"
        + file_name_sci_list[0].split("/")[-1].split(".")[0].split("_")[1]
        + "_"
        + file_name_sci_list[0].split("/")[-1].split(".")[0].split("_")[0]
        + "_"
        + file_name_sci_list[0].split("/")[-1].split(".")[0].split("_")[2]
        + "_"
        + file_name_sci_list[0].split("/")[-1].split(".")[0].split("_")[3]
        + "_"
        + file_name_sci_list[-1].split("/")[-1].split(".")[0].split("_")[-4]
        + "_"
        + file_name_sci_list[-1].split("/")[-1].split(".")[0].split("_")[-3]
        + "_sci_output.csv"
    )

    # Save the dataframe to a csv file
    df_sci.to_csv(file_name_sci, index=False)

    return file_name_sci, df_sci
