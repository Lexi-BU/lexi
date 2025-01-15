from lexi_xray import lexi as lexi
import importlib
import pandas as pd
from pathlib import Path
import shutil

import numpy as np

importlib.reload(lexi)
#
# array_to_image_kwargs_exp = {
#     "x_range": [190, 240],
#     "y_range": [-30, -10],
#     "x_lim": [220, 240],
#     "y_lim": [-30, -15],
#     "cmap": "plasma",
#     "cmin": 1,
#     "norm": None,
#     "norm_type": "log",
#     "aspect": "equal",
#     "figure_title": "LEXI Exposure Map",
#     "show_colorbar": True,
#     "cbar_label": "Seconds",
#     "cbar_orientation": "vertical",
#     "show_axes": True,
#     "display": False,
#     "figure_size": (5, 5),
#     "figure_format": "png",
#     "figure_font_size": 12,
#     "save": True,
#     "save_name": "default",
#     "save_path": None,
#     "dpi": 300,
#     "dark_mode": True,
#     "verbose": True,
# }
#
# array_to_image_kwargs_sbg = {
#     "x_range": [190, 240],
#     "y_range": [-30, -10],
#     "x_lim": [220, 240],
#     "y_lim": [-30, -15],
#     "cmap": "plasma",
#     "cmin": 1,
#     "norm": None,
#     "norm_type": "log",
#     "aspect": "equal",
#     "figure_title": "Sky Backgrounds",
#     "show_colorbar": True,
#     "cbar_label": "Counts",
#     "cbar_orientation": "vertical",
#     "show_axes": True,
#     "display": False,
#     "figure_size": (5, 5),
#     "figure_format": "png",
#     "figure_font_size": 12,
#     "save": True,
#     "save_name": "default",
#     "save_path": None,
#     "dpi": 300,
#     "dark_mode": True,
#     "verbose": True,
# }
#
# array_to_image_kwargs_lex = {
#     "x_range": [190, 240],
#     "y_range": [-30, -10],
#     "x_lim": [220, 240],
#     "y_lim": [-30, -15],
#     "cmap": "plasma",
#     "cmin": 1,
#     "norm": None,
#     "norm_type": "log",
#     "aspect": "equal",
#     # "figure_title": "LEXI Image",
#     "show_colorbar": True,
#     "cbar_label": "Counts",
#     "cbar_orientation": "vertical",
#     "show_axes": True,
#     "display": True,
#     "figure_size": (5, 5),
#     "figure_format": "png",
#     "figure_font_size": 12,
#     "save": True,
#     "save_name": "default",
#     "save_path": None,
#     "dpi": 300,
#     "dark_mode": True,
#     "verbose": True,
# }
#
# input_params = {
#     "time_range": ["2025-03-04 08:53:41", "2025-03-04 09:23:41"],
#     "ra_res": 1,
#     "dec_res": 1,
#     "ra_range": [190, 240],
#     "dec_range": [-30, -10],
#     "save_exposure_map_file": True,
#     "save_exposure_map_image": True,
#     # "save_sky_backgrounds_file": True,
#     # "save_sky_backgrounds_image": False,
#     # "save_lexi_images": True,
#     "verbose": True,
#     # "background_correction_on": True,
#     "array_to_image_kwargs": array_to_image_kwargs_lex,
# }
#
#
# # df1, df2, df3 = lexi.get_lexi_data(**input_params)
# # df3 = lexi.get_spc_prams(**input_params)
# # df1, df2, df3 = lexi.get_spc_prams(**input_params)
# # print(df1, df2, df3)
# # print(input_params["time_range"])
# # exposure_maps_dict = lexi.calc_exposure_maps(**input_params)
# # Delete the `data` folder and its contents and `figures` folder and its contents before running this
# # code using Path even if the fodlers are not empty
# # shutil.rmtree(Path("data"), ignore_errors=True)
# # shutil.rmtree(Path("figures"), ignore_errors=True)
# #
# lexi_images_dict = lexi.make_lexi_images(
#     time_range=["2025-03-04 08:53:41", "2025-03-05 09:23:46"],
#     ra_range=[190, 310],
#     dec_range=[-33, 3],
#     ra_res=0.5,
#     dec_res=0.5,
#     time_step=0.5,
#     # time_integrate=600.23,
#     background_correction_on=True,
#     save_exposure_map_file=True,
#     save_sky_backgrounds_file=True,
#     save_exposure_map_image=True,
#     save_sky_backgrounds_image=True,
#     save_lexi_images=True,
#     verbose=False,
#     array_to_image_kwargs={
#         "norm_type": "log",
#     },
# )
# # exposure_maps = exposure_maps_dict["exposure_maps"]
# print(f"the shape of exposure_maps is {np.shape(lexi_images_dict['lexi_images'])}")
# # print(np.shape(exposure_maps))
# # print(np.shape(exposure_maps[0]))
# # print(np.nanmin(exposure_maps[0]), np.nanmax(exposure_maps[0]))
# # print(np.shape(exposure_maps_dict["exposure_maps"]))
# # sky_backgrounds_dict = lexi.get_sky_backgrounds(**input_params)
# # print(np.shape(sky_backgrounds_dict["sky_backgrounds"]))
# # print(df.head())
#
# # lexi_images_dict = lexi.get_lexi_images(**input_params)
# # print(np.shape(lexi_images_dict["lexi_images"]))
#
#
# # lexi.get_lexi_data(
# #     time_range=[
# #         pd.to_datetime("2025-03-04 08:53:41"),
# #         pd.to_datetime("2025-03-04 09:23:41"),
# #     ]
# # )
#

input_run_val = input("Enter 1 to run the code: ")

if input_run_val == "1" or input_run_val == "3":
    print("Running the code for the first time")
    lexi_images_dict_2 = lexi.make_lexi_images(
        time_range=["2025-03-02 08:04:00", "2025-03-02 08:14:00"],
        ra_range=[190, 210],
        dec_range=[-40, 0],
        ra_res=0.5,
        dec_res=0.5,
        # time_integrate=500,
        save_exposure_map_image=True,
        save_sky_backgrounds_image=True,
        background_correction_on=True,
        save_lexi_images=True,
        verbose=False,
        array_to_image_kwargs={"norm_type": "log"},
    )
if input_run_val == "2" or input_run_val == "3":
    print("Running the code for the second time")
    lexi_images_dict_1 = lexi.make_lexi_images(
        time_range=["2025-03-02 08:04:00", "2025-03-02 09:14:00"],
        ra_range=[170, 260],
        dec_range=[-25, -6],
        ra_res=0.5,
        dec_res=0.5,
        # time_integrate=500,
        save_exposure_map_image=True,
        save_sky_backgrounds_image=True,
        background_correction_on=True,
        save_lexi_images=True,
        verbose=False,
        array_to_image_kwargs={"norm_type": "log", "display_time": True},
    )

# sky_backgrounds_dict = lexi.calc_sky_backgrounds(
#     time_range=["2025-03-02 08:04:00", "2025-03-09 08:14:00"],
#     # ra_range=[190, 210],
#     # dec_range=[-40, 0],
#     ra_res=0.5,
#     dec_res=0.5,
#     verbose=False,
#     save_sky_backgrounds_image=True,
# )
# input_params = {
#     "input_array": lexi_images_dict_2["lexi_images"][0],
#     "x_range": [190, 210],
#     "y_range": [-40, 0],
#     "x_lim": [190, 210],
# }
# fig, ax = lexi.array_to_image(**input_params)
