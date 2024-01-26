# Write a fuunction which takes time as an input and gets the spacecraft ephemeris data by
# downloading the approapriate file from the NASA CDAweb website.

import urllib.request
import pandas as pd
from spacepy import pycdf
from pathlib import Path


def get_spc_prams(self):
    """ """

    # Define the link to the CDAweb website
    cda_link = "https://cdaweb.gsfc.nasa.gov/pub/data/lexi/ephemeris/"

    start_time = self.t_range[0]
    stop_time = self.t_range[1]

    # Get the year, month, and day of the start and stop times
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
                link = cda_link + filename

                # Try to open the link, if it doesn't exist then skip to the next date
                try:
                    urllib.request.urlopen(link)
                except urllib.error.HTTPError:
                    # Print in that the file doesn't exist or is unavailable for download from the CDAweb website
                    print(
                        f"Warning: Following file is unavailable for downloading oor doesn't exits. Skipping the file: \033[93m {filename}\033[0m"
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
        urllib.request.urlretrieve(cda_link + file, data_dir / file)

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

    return df
