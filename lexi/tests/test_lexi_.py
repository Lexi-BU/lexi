import pytest
from lexi.lexi import LEXI


@pytest.fixture
def lexi_instances():
    instances = []
    # Define some sample input parameters for LEXI class
    input_params_save_map = {
        "save_df": True,
        "filename": "test_data/LEXI_pointing_ephem_highres",
        "filetype": "pkl",
        "interp_method": "linear",
        "background_correction_on": False,
        "t_range": ("2024-01-01", "2024-01-02"),
        "t_step": 10,
        "t_integrate": 600,
        "ra_range": [0.0, 360.0],
        "dec_range": [-90.0, 90.0],
        "ra_res": 0.1,
        "dec_res": 0.1,
        "save_exposure_maps": True,
        "save_sky_backgrounds": True,
        "save_lexi_images": True
    }
    input_params_no_save_map = {
        "save_df": False,
        "filename": "test_data/LEXI_pointing_ephem_highres",
        "filetype": "pkl",
        "interp_method": "linear",
        "background_correction_on": False,
        "t_range": ("2024-01-01", "2024-01-02"),
        "t_step": 10,
        "t_integrate": 600,
        "ra_range": [0.0, 360.0],
        "dec_range": [-90.0, 90.0],
        "ra_res": 0.1,
        "dec_res": 0.1,
        "save_exposure_maps": False,
        "save_sky_backgrounds": False,
        "save_lexi_images": False
    }

    instances.append(LEXI(input_params_save_map))
    instances.append(LEXI(input_params_no_save_map))
    return instances


def test_get_spc_prams(lexi_instances):
    for lexi_instance in lexi_instances:
        spaceparams = lexi_instance.get_spc_prams()
        assert spaceparams is not None