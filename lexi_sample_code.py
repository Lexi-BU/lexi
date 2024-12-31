from lexi_bu import lexi as lexi
import importlib
import pandas as pd

# import numpy as np
#
importlib.reload(lexi)

array_to_image_kwargs_exp = {
    "x_range": [190, 240],
    "y_range": [-30, -10],
    "x_lim": [220, 240],
    "y_lim": [-30, -15],
    "cmap": "plasma",
    "cmin": 1,
    "norm": None,
    "norm_type": "log",
    "aspect": "equal",
    "figure_title": "LEXI Exposure Map",
    "show_colorbar": True,
    "cbar_label": "Seconds",
    "cbar_orientation": "vertical",
    "show_axes": True,
    "display": False,
    "figure_size": (5, 5),
    "figure_format": "png",
    "figure_font_size": 12,
    "save": True,
    "save_name": "default",
    "save_path": None,
    "dpi": 300,
    "dark_mode": True,
    "verbose": True,
}

array_to_image_kwargs_sbg = {
    "x_range": [190, 240],
    "y_range": [-30, -10],
    "x_lim": [220, 240],
    "y_lim": [-30, -15],
    "cmap": "plasma",
    "cmin": 1,
    "norm": None,
    "norm_type": "log",
    "aspect": "equal",
    "figure_title": "Sky Backgrounds",
    "show_colorbar": True,
    "cbar_label": "Counts",
    "cbar_orientation": "vertical",
    "show_axes": True,
    "display": False,
    "figure_size": (5, 5),
    "figure_format": "png",
    "figure_font_size": 12,
    "save": True,
    "save_name": "default",
    "save_path": None,
    "dpi": 300,
    "dark_mode": True,
    "verbose": True,
}

array_to_image_kwargs_lex = {
    "x_range": [190, 240],
    "y_range": [-30, -10],
    "x_lim": [220, 240],
    "y_lim": [-30, -15],
    "cmap": "plasma",
    "cmin": 1,
    "norm": None,
    "norm_type": "log",
    "aspect": "equal",
    # "figure_title": "LEXI Image",
    "show_colorbar": True,
    "cbar_label": "Counts",
    "cbar_orientation": "vertical",
    "show_axes": True,
    "display": False,
    "figure_size": (5, 5),
    "figure_format": "png",
    "figure_font_size": 12,
    "save": True,
    "save_name": "default",
    "save_path": None,
    "dpi": 300,
    "dark_mode": True,
    "verbose": True,
}

input_params = {
    "time_range": ["2025-03-04 08:53:41", "2025-03-04 09:23:41"],
    "ra_res": 1,
    "dec_res": 1,
    "ra_range": [190, 240],
    "dec_range": [-30, -10],
    "save_exposure_map_file": True,
    "save_exposure_map_image": False,
    "save_sky_backgrounds_file": True,
    "save_sky_backgrounds_image": False,
    "save_lexi_images": True,
    "verbose": True,
    "background_correction_on": True,
    "array_to_image_kwargs": array_to_image_kwargs_lex,
}


# df1, df2, df3 = lexi.get_lexi_data(**input_params)
# df3 = lexi.get_spc_prams(**input_params)
# df1, df2, df3 = lexi.get_spc_prams(**input_params)
# print(df1, df2, df3)
# print(input_params["time_range"])
# exposure_maps_dict = lexi.get_lexi_images(**input_params)

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


lexi.get_lexi_data(
    time_range=[
        pd.to_datetime("2025-03-04 08:53:41"),
        pd.to_datetime("2025-03-04 09:23:41"),
    ]
)
