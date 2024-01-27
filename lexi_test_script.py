# Import LEXI
from lexi.lexi import LEXI

# Set up a LEXI instance
lexi = LEXI(
    {
        "t_range": [
            "2024-07-08T21:43:41",
            "2024-07-08T21:47:48",
        ],
        "t_integrate": 90,  # 10 minutes - spans whole range
        "t_step": 60,  # 1, #30 # NOTE: If we set t_step to 3 or 30, the exposure maps keep getting saved. There is a bug in the code specifically for this case.
        # "ra_range": [325.0, 365.0],
        # "dec_range": [-21.0, 6.0],
        "ra_range": [0, 360],
        "dec_range": [0, 360],
        "ra_res": 3,
        "dec_res": 3,
        # "background_correction_on": False, # Dates do not align
        "save_exposure_maps": True,
        "save_sky_backgrounds": True,
        "save_lexi_images": True,
    }
)

# Get space params
# spaceparams = lexi.get_spc_prams()
# Get exposure maps
# expmaps = lexi.get_exposure_maps()
# Get sky backgrounds
# skybgs = lexi.get_sky_backgrounds()
# Get background corrected images
hists = lexi.get_lexi_images()
