# Define class called LEXI with a list of different functions that can be called


class LEXI:
    # Define the first function called "get_spc_prams" that takes time as an argument and returns the
    # time, look direction (gamma) and the roll angle (phi) of the lander
    def get_spc_prams(self, time, time_resolution):
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

        time = [0, 0]  # The start and end time of the simulation
        gamma = 0
        alpha = 0
        time_array = 0
        return time_array, gamma, alpha

    # Define a second function which takes the following list of arguments:
    #    --time
    #    --RA
    #    --DEC
    #    --binsize
    #    --nbins
    #    --integration_time
    # The function then computes the sky background and returns the sky background
    def get_sky_background(self, time, RA, DEC, binsize, nbins, integration_time):
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

        time_array, gamma, alpha = self.get_spc_prams(time, 0)
        sky_background = 0

        return sky_background

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
