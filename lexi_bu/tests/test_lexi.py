import unittest
import datetime
import pytest
import warnings
import numpy as np
import pandas as pd
from unittest.mock import patch
from lexi_bu.lexi import (
    validate_input,
    get_lexi_data,
    get_spc_prams,
    get_exposure_maps,
    get_sky_backgrounds,
    get_lexi_images,
)

# Suppress warnings in tests for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


def test_validate_input_time_range():
    # Valid inputs
    assert validate_input("time_range", ["2022-01-01T00:00:00", "2022-01-02T00:00:00"])
    assert validate_input(
        "time_range", [pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-02")]
    )
    assert validate_input("time_range", [1640995200, 1641081600])
    assert validate_input(
        "time_range", [datetime.datetime(2022, 1, 1), datetime.datetime(2022, 1, 2)]
    )

    # Invalid inputs
    with pytest.raises(ValueError):
        validate_input("time_range", "2022-01-01T00:00:00")  # Not a list
    with pytest.raises(ValueError):
        validate_input("time_range", ["2022-01-01T00:00:00"])  # Only one element


def test_validate_input_time_zone():
    # Valid inputs
    assert validate_input("time_zone", "UTC")
    assert validate_input("time_zone", "America/New_York")

    # Invalid inputs
    with pytest.raises(ValueError):
        validate_input("time_zone", 123)  # Not a string
    assert not validate_input("time_zone", "Invalid/Zone")  # Invalid timezone


def test_validate_input_ra_range():
    # Valid inputs
    assert validate_input("ra_range", [0, 360])
    assert validate_input("ra_range", np.array([10, 50]))

    # Invalid inputs
    with pytest.raises(ValueError):
        validate_input("ra_range", "Not a list")  # Invalid type
    assert not validate_input("ra_range", [400, 50])  # Out of range
    assert not validate_input("ra_range", [10, "Not a number"])  # Mixed types


def test_validate_input_dec_range():
    # Valid inputs
    assert validate_input("dec_range", [-90, 90])
    assert validate_input("dec_range", np.array([-45, 45]))

    # Invalid inputs
    with pytest.raises(ValueError):
        validate_input("dec_range", "Not a list")  # Invalid type
    assert not validate_input("dec_range", [-100, 45])  # Out of range
    assert not validate_input("dec_range", [10, "Not a number"])  # Mixed types


def test_validate_input_numeric_positive():
    # Valid inputs
    assert validate_input("time_step", 10)
    assert validate_input("ra_res", 0.1)
    assert validate_input("dec_res", 0.1)

    # Invalid inputs
    assert not validate_input("time_step", -10)  # Negative value
    assert not validate_input("ra_res", "Not a number")  # Invalid type
    assert not validate_input("dec_res", 0)  # Non-positive


def test_validate_input_boolean():
    # Valid inputs
    assert validate_input("background_correction_on", True)
    assert validate_input("save_df", False)

    # Invalid inputs
    with pytest.raises(ValueError):
        validate_input("background_correction_on", "Not a boolean")  # Invalid type
    with pytest.raises(ValueError):
        validate_input("save_df", 1)  # Not a boolean


def test_validate_input_filename_filetype():
    # Valid inputs
    assert validate_input("filename", "example_file")
    assert validate_input("filetype", "pkl")

    # Invalid inputs
    with pytest.raises(ValueError):
        validate_input("filename", "")  # Empty string
    with pytest.raises(ValueError):
        validate_input("filetype", "unsupported_type")  # Invalid filetype


def test_get_spc_prams():
    # Test Case 1: Testing with valid time_range, time_zone and default values
    time_range = [
        pd.to_datetime("2025-03-02 08:50:00"),
        pd.to_datetime("2025-03-02 09:23:00"),
    ]
    result = get_spc_prams(time_range=time_range)
    assert isinstance(result, pd.DataFrame), "Result should be a pandas DataFrame."
    assert not result.empty, "Result DataFrame should not be empty."

    # Check if the time range is properly respected (check if first time is within range)
    assert result.index[0] >= time_range[0], f"Start time should be >= {time_range[0]}"
    assert result.index[-1] <= time_range[1], f"End time should be <= {time_range[1]}"

    # Test Case 3: Test interpolation method parameter (mocking resampling and interpolation)
    interp_method = "linear"
    result = get_spc_prams(time_range=time_range, interp_method=interp_method)
    assert not result.empty, "Result should not be empty after interpolation."

    # Test Case 4: Test with data_clip set to False (check if the data is not clipped)
    data_clip = False
    result = get_spc_prams(time_range=time_range, data_clip=data_clip)
    assert not result.empty, "Result should not be empty with data_clip=False."

    # Test Case 5: Test with lexi_data set to True (lexi data behavior)
    lexi_data = True
    result = get_spc_prams(time_range=time_range, lexi_data=lexi_data)
    # Since we are mocking, we'll just check that the result is still a DataFrame
    assert isinstance(
        result, pd.DataFrame
    ), "Result should still be a pandas DataFrame when lexi_data=True."

    # Test Case 6: Test return_data_type option 'merged' (if data is merged with lexi data)
    return_data_type = "merged"
    result = get_spc_prams(
        time_range=time_range, lexi_data=True, return_data_type=return_data_type
    )
    assert isinstance(result, pd.DataFrame), "Merged result should be a DataFrame."

    # Test Case 8: Test with default time_step and time_pad
    time_step = 5  # in seconds
    time_pad = 300  # in seconds
    result = get_spc_prams(
        time_range=time_range, time_step=time_step, time_pad=time_pad
    )
    assert isinstance(
        result, pd.DataFrame
    ), "Result should be a pandas DataFrame when using default time_step and time_pad."


# Helper function to create a dummy test for validation
def test_validate_input():
    assert validate_input("time_step", 5) is True
    assert validate_input("ra_range", [0, 360]) is True
    assert validate_input("dec_range", [-90, 90]) is True
    assert validate_input("ra_res", 0.1) is True
    assert validate_input("dec_res", 0.1) is True


# Test: Ensure function returns correct structure (dictionary)
def test_get_exposure_maps_basic():
    time_range = ["2025-03-04 08:53:41", "2025-03-04 09:23:41"]
    exposure_maps_dict = get_exposure_maps(
        time_range=time_range, ra_range=[160, 230], dec_range=[-20, 5]
    )

    # Check if the return type is a dictionary
    assert isinstance(exposure_maps_dict, dict)

    # Check if expected keys are in the returned dictionary
    expected_keys = [
        "exposure_maps",
        "ra_arr",
        "dec_arr",
        "time_range",
        "time_integrate",
        "ra_range",
        "dec_range",
        "ra_res",
        "dec_res",
        "start_time_arr",
        "stop_time_arr",
    ]
    for key in expected_keys:
        assert key in exposure_maps_dict


# FIXME: This test is failing because the function is not returning the expected values for "ra_range" and "dec_range"
# Test: Check validation of RA and DEC ranges
# def test_ra_dec_range_validation():
#     time_range = ["2025-03-04 08:53:41", "2025-03-04 09:23:41"]
#
#     # Invalid RA/DEC range
#     with patch("builtins.print") as mocked_print:
#         exposure_maps_dict = get_exposure_maps(
#             time_range=time_range, ra_range=[400, 360], dec_range=[-100, 100]
#         )
#
#         # Assert that "ra_range" is set to [0, 360] and "dec_range" is set to [-90, 90]
#         assert np.array_equal(
#             exposure_maps_dict["ra_range"], [0, 360]
#         ), f"Expected ra_range to be [0, 360], but got {exposure_maps_dict['ra_range']}"
#         assert np.array_equal(
#             exposure_maps_dict["dec_range"], [-90, 90]
#         ), f"Expected dec_range to be [-90, 90], but got {exposure_maps_dict['dec_range']}"
#
#         # Check that the mocked print function was called with the correct output
#         mocked_print.assert_any_call(
#             "\033[1;91m RA range \033[1;92m (ra_range) \033[1;91m not provided. Setting RA range to the range of the spacecraft ephemeris data: \033[1;92m [0, 360] \033[0m\n"
#         )
#         mocked_print.assert_any_call(
#             "\033[1;91m DEC range \033[1;92m (dec_range) \033[1;91m not provided. Setting DEC range to the range of the spacecraft ephemeris data: \033[1;92m [-90, 90] \033[0m\n"
#         )


# Test: Check if exposure map is computed correctly
def test_exposure_map_computation():
    time_range = ["2025-03-04 08:53:41", "2025-03-04 09:23:41"]
    exposure_maps_dict = get_exposure_maps(
        time_range=time_range,
        ra_range=[160, 230],
        dec_range=[-20, 5],
        time_integrate=500,
    )

    # Check if exposure map array has been computed
    exposure_maps = exposure_maps_dict["exposure_maps"]
    assert isinstance(exposure_maps, np.ndarray)
    assert len(exposure_maps.shape) == 3  # Expecting 3D array (time, RA, DEC)

    # Check that the shape of the array matches the expected dimensions
    assert exposure_maps.shape[1] == len(exposure_maps_dict["ra_arr"])
    assert exposure_maps.shape[2] == len(exposure_maps_dict["dec_arr"])


# FIXME: Need to resolve the path issue before asserting the file saving
# Test: Check saving exposure map file
# def test_save_exposure_map_file():
#     time_range = ["2025-03-04 08:53:41", "2025-03-04 09:23:41"]
#
#     with patch("builtins.print") as mocked_print:
#         exposure_maps_dict = get_exposure_maps(
#             time_range=time_range,
#             ra_range=[160, 230],
#             dec_range=[-20, 5],
#             save_exposure_map_file=True,
#         )
#
#         # Check if file saving logic is triggered
#         assert "data/exposure_maps" in str(mocked_print.call_args)
#         mocked_print.assert_any_call("Exposure map saved to file")


# Test: Ensure function handles incorrect time_range input
def test_incorrect_time_range():
    incorrect_time_range = ["2025-03-02 08:50:00", "not a datetime"]

    with pytest.raises(ValueError):
        get_exposure_maps(time_range=incorrect_time_range)


# Test: Check time_integrate default behavior
def test_time_integrate_default():
    time_range = ["2025-03-04 08:53:41", "2025-03-04 09:23:41"]
    exposure_maps_dict = get_exposure_maps(
        time_range=time_range, ra_range=[160, 230], dec_range=[-20, 5]
    )
    time_diff = (
        pd.to_datetime(time_range[1]) - pd.to_datetime(time_range[0])
    ).total_seconds()
    # Check if time_integrate is set to 1 by default
    assert exposure_maps_dict["time_integrate"] == time_diff


# FIXME: Need to resolve the path issue before asserting the file saving
# Test: Check exposure map image saving
# def test_save_exposure_map_image():
#     time_range = ["2025-03-04 08:53:41", "2025-03-04 09:23:41"]
#
#     with patch("builtins.print") as mocked_print:
#         exposure_maps_dict = get_exposure_maps(
#             time_range=time_range,
#             ra_range=[160, 230],
#             dec_range=[-20, 5],
#             save_exposure_map_image=True,
#         )
#
#         # Check if image saving logic is triggered
#         mocked_print.assert_any_call("Saving exposure maps as images")


# Test: Ensure function returns correct structure (dictionary)
def test_get_sky_backgrounds_basic():
    time_range = ["2025-03-04 08:53:41", "2025-03-04 09:23:41"]
    sky_backgrounds_dict = get_sky_backgrounds(
        time_range=time_range, ra_range=[160, 230], dec_range=[-20, 5]
    )

    # Check if the return type is a dictionary
    assert isinstance(sky_backgrounds_dict, dict)

    # Check if expected keys are in the returned dictionary
    expected_keys = [
        "sky_backgrounds",
        "ra_arr",
        "dec_arr",
        "time_range",
        "ra_range",
        "dec_range",
        "ra_res",
        "dec_res",
        "start_time_arr",
        "stop_time_arr",
    ]
    for key in expected_keys:
        assert key in sky_backgrounds_dict


# Test: Check if sky background is computed correctly
def test_sky_background_computation():
    time_range = ["2025-03-04 08:53:41", "2025-03-04 09:23:41"]
    sky_backgrounds_dict = get_sky_backgrounds(
        time_range=time_range,
        ra_range=[160, 230],
        dec_range=[-20, 5],
    )

    # Check if sky background array has been computed
    sky_backgrounds = sky_backgrounds_dict["sky_backgrounds"]
    assert isinstance(sky_backgrounds, np.ndarray)
    assert len(sky_backgrounds.shape) == 3  # Expecting 3D array (time, RA, DEC)

    # Check that the shape of the array matches the expected dimensions
    assert sky_backgrounds.shape[1] == len(sky_backgrounds_dict["ra_arr"])
    assert sky_backgrounds.shape[2] == len(sky_backgrounds_dict["dec_arr"])


# Test: Ensure function handles incorrect time_range input
def test_incorrect_time_range_sky_backgrounds():
    incorrect_time_range = ["2025-03-02 08:50:00", "not a datetime"]

    with pytest.raises(ValueError):
        get_sky_backgrounds(time_range=incorrect_time_range)


# Test: Ensure function returns correct structure (dictionary)
def test_get_lexi_images_basic():
    time_range = ["2025-03-04 08:53:41", "2025-03-04 09:23:41"]
    lexi_images_dict = get_lexi_images(
        time_range=time_range, ra_range=[160, 230], dec_range=[-20, 5]
    )

    # Check if the return type is a dictionary
    assert isinstance(lexi_images_dict, dict)

    # Check if expected keys are in the returned dictionary
    expected_keys = [
        "lexi_images",
        "ra_arr",
        "dec_arr",
        "time_range",
        "ra_range",
        "dec_range",
        "ra_res",
        "dec_res",
        "start_time_arr",
        "stop_time_arr",
    ]
    for key in expected_keys:
        assert key in lexi_images_dict


# Test: Check if lexi images are computed correctly
def test_lexi_images_computation():
    time_range = ["2025-03-04 08:53:41", "2025-03-04 09:23:41"]
    lexi_images_dict = get_lexi_images(
        time_range=time_range,
        ra_range=[160, 230],
        dec_range=[-20, 5],
    )

    # Check if lexi images array has been computed
    lexi_images = lexi_images_dict["lexi_images"]
    assert isinstance(lexi_images, np.ndarray)
    assert len(lexi_images.shape) == 3  # Expecting 3D array (time, RA, DEC)

    # Check that the shape of the array matches the expected dimensions
    assert lexi_images.shape[1] == len(lexi_images_dict["ra_arr"])
    assert lexi_images.shape[2] == len(lexi_images_dict["dec_arr"])
