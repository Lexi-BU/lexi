import lexi_no_class as lnc
import importlib

importlib.reload(lnc)


df = lnc.get_spc_prams(
    # t_range=["2024-07-08T21:43:41", "2024-07-08T21:47:48"],
    t_range=["2001-01-08T21:43:41", "2001-12-08T21:47:48"],
    time_zone="IST",
    t_step="a",
    # ra_range=[290, 360],
    # dec_range=[290, 360],
    # ra_res=4,
    # dec_res=3,
    interp_method="linearly",
    # save_df=True,
    # filename="test_data/LEXI_pointing_ephem_highres",
    # filetype="pkl",
    verbose=True,
)

# print(df.head())
