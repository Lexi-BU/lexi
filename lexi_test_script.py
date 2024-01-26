# Import LEXI
from lexi.lxi_code_plan import LEXI

# Set up a LEXI instance
lexi = LEXI(
    {
        "t_range": [
            "2024-07-08T15:01:00",
            "2024-07-09T15:01:00",
        ],  # Ephemeris data timerange
        # "t_range": ["2024-05-23T21:43:41","2024-05-23T21:48:41"], # Ramiz's PIT data timerange
        "t_integrate": 60 * 60 * 24,  # 12 hours # usually couple secs to (10m) to hour
        "t_step": 3,  # 1, #30
        "ra_range": [325.0, 365.0],
        "dec_range": [-21.0, 6.0],
        # "ra_range": [0, 360],
        # "dec_range": [0, 360],
        "ra_res": 3,
        "dec_res": 3,
        # "background_correction_on": False, # Dates do not align
        "save_exposure_maps": False,
        "save_sky_backgrounds": False,
        "save_lexi_images": False,
    }
)

# Get space params
# spaceparams = lexi.get_spc_prams()
# Get exposure maps
expmaps = lexi.get_exposure_maps()
# Get sky backgrounds
# skybgs = lexi.get_sky_backgrounds()
# Get background corrected images
# hists = lexi.get_lexi_images()
