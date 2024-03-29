from lexi import lexi as lexi
import importlib
import numpy as np

importlib.reload(lexi)

input_params = {
    "time_range": ["2024-07-08T21:43:41", "2024-07-08T21:47:41"],
    "time_zone": "UTC",
    # "time_integrate": 120,
    # "time_step": "5",
    "ra_res": 2,
    "dec_res": 5,
    "ra_range": [330, 360],
    "dec_range": [60, 90],
    "interp_method": "linear",
    # save_df=True,
    # filename="test_data/LEXI_pointing_ephem_highres",
    # filetype="pkl",
    "save_exposure_map_file": True,
    "save_exposure_map_image": True,
    "save_sky_backgrounds_file": True,
    "save_sky_backgrounds_image": True,
    "background_correction_on": True,
    "save_lexi_images": True,
    "verbose": True,
}

# df = lexi.get_spc_prams(**input_params)

# exposure_maps_dict = lexi.get_exposure_maps(**input_params)

# print(np.shape(exposure_maps_dict["exposure_maps"]))
# sky_backgrounds_dict = lexi.get_sky_backgrounds(**input_params)
# print(np.shape(sky_backgrounds_dict["sky_backgrounds"]))
# print(df.head())

lexi_images_dict = lexi.get_lexi_images(**input_params)
# print(np.shape(lexi_images_dict["lexi_images"]))
