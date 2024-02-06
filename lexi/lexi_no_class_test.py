import lexi_no_class as lnc
import importlib
import numpy as np

importlib.reload(lnc)


input_params = {
    # t_range=["2024-07-08T21:43:41", "2024-07-08T21:47:48"],
    "t_range": ["2024-07-08T21:43:41", "2024-07-09T21:47:48"],
    "time_zone": "UTC",
    "t_step": "5",
    "ra_res": 1,
    "dec_res": 1,
    "ra_range": np.array([330, 360]),
    # ra_range=[290, 360],
    "dec_range": np.array([-80, 80]),
    # ra_res=4,
    # dec_res=3,
    "interp_method": "linear",
    # "t_integrate": 20,
    # save_df=True,
    # filename="test_data/LEXI_pointing_ephem_highres",
    # filetype="pkl",
    "save_exposure_map_file": True,
    "verbose": True,
}

# df = lnc.get_spc_prams(**input_params)

exposure_maps, ra_arr, dec_arr = lnc.get_exposure_maps(**input_params)
# print(df.head())
