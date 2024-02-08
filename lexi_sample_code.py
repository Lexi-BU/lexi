import lexi.lexi as lfs
import importlib
import numpy as np

importlib.reload(lfs)

input_params = {
    "time_range": ["2024-07-08T21:43:41", "2024-07-08T21:47:41"],
    "time_zone": "UTC",
    # "t_integrate": 120,
    # "t_step": "5",
    "ra_res": 1,
    "dec_res": 1,
    "ra_range": np.array([330, 360]),
    # ra_range=[290, 360],
    "dec_range": np.array([60, 90]),
    # ra_res=4,
    # dec_res=3,
    "interp_method": "linear",
    # "t_integrate": 20,
    # save_df=True,
    # filename="test_data/LEXI_pointing_ephem_highres",
    # filetype="pkl",
    "save_exposure_map_file": True,
    "save_exposure_map_image": True,
    "save_sky_backgrounds_file": True,
    "save_sky_backgrounds_image": True,
    "background_correction_on": True,
    "save_lexi_images": True,
    "verbose": False,
}

# df = lnc.get_spc_prams(**input_params)

# exposure_maps_dict = lfs.get_exposure_maps(**input_params)

# print(np.shape(exposure_maps_dict["exposure_maps"]))
# sky_backgrounds_dict = lfs.get_sky_backgrounds(**input_params)
# print(np.shape(sky_backgrounds_dict["sky_backgrounds"]))
# print(df.head())

lexi_images_dict = lfs.get_lexi_images(**input_params)
# print(np.shape(lexi_images_dict["lexi_images"]))
