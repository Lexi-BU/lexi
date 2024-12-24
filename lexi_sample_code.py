from lexi import lexi as lexi
import importlib
import numpy as np

importlib.reload(lexi)

input_params = {
    "time_range": ["2025-03-04 08:53:41", "2025-03-04 09:23:41"],
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
    "verbose": True,
    "spc_prams": True,
    # "lexi_data": True,
    "return_data_type": "all",
    "time_pad": 600,
    "data_clip": False,
    # "spc_prams_kwargs": {"time_step": "60", "interp_method": "index"},
    # "lexi_data_kwargs": {
    # "force_compute": False,
}

# df1, df2, df3 = lexi.get_lexi_data(**input_params)
# df3 = lexi.get_spc_prams(**input_params)
df1, df2, df3 = lexi.get_lexi_data(**input_params)
print(df1, df2, df3)
print(input_params["time_range"])
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
