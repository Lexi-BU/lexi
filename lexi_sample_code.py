from lexi import lexi as lexi
import importlib
import numpy as np

importlib.reload(lexi)

input_params = {
    "time_range": ["2025-03-04T08:53:41", "2025-03-08T08:53:41"],
    # "time_zone": "UTC",
    # "time_integrate": 120,
    # "time_step": "5",
    # "ra_res": 1,
    # "dec_res": 1,
    # "ra_range": [190, 240],
    # "dec_range": [-30, -10],
    # "interp_method": "linear",
    # save_df=True,
    # filename="test_data/LEXI_pointing_ephem_highres",
    # filetype="pkl",
    # "save_exposure_map_file": True,
    # "save_exposure_map_image": True,
    # "save_sky_backgrounds_file": True,
    # "save_sky_backgrounds_image": True,
    # "background_correction_on": False,
    # "save_lexi_images": True,
    "verbose": False,
    # "force_compute": False,
}

df = lexi.get_lexi_data(**input_params)
# df = lexi.get_spc_prams(**input_params)

# exposure_maps_dict = lexi.get_exposure_maps(**input_params)

# exposure_maps = exposure_maps_dict["exposure_maps"]

# print(np.shape(exposure_maps))
# print(np.shape(exposure_maps[0]))
# print(np.nanmin(exposure_maps[0]), np.nanmax(exposure_maps[0]))
# print(np.shape(exposure_maps_dict["exposure_maps"]))
# sky_backgrounds_dict = lexi.get_sky_backgrounds(**input_params)
# print(np.shape(sky_backgrounds_dict["sky_backgrounds"]))
# print(df.head())

# lexi_images_dict = lexi.get_lexi_images(**input_params)
# print(np.shape(lexi_images_dict["lexi_images"]))
