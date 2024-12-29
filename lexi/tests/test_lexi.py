"""Test the functions in the lexi package."""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime
from lexi import (
    validate_input,
    download_files_from_github,
    get_lexi_data,
    get_spc_prams,
    get_exposure_maps,
)


class TestLexiFunctions(unittest.TestCase):

    def test_validate_input_time_range(self):
        self.assertTrue(
            validate_input("time_range", ["2025-03-03T11:22:33", "2025-03-03T11:22:33"])
        )
        self.assertRaises(
            ValueError, validate_input, "time_range", "2023-03-03T11:22:33"
        )

    def test_validate_input_time_zone(self):
        self.assertTrue(validate_input("time_zone", "UTC"))
        self.assertFalse(validate_input("time_zone", "INVALID_TIMEZONE"))

    def test_validate_input_ra_range(self):
        self.assertTrue(validate_input("ra_range", [0, 360]))
        self.assertFalse(validate_input("ra_range", [-10, 400]))

    @patch("lexi.requests.get")
    def test_download_files_from_github(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"name": "testfile", "download_url": "http://example.com/testfile"}
        ]
        mock_get.return_value = mock_response

        result = download_files_from_github(["testfile"], "repo", "folder_path")
        self.assertEqual(len(result), 1)

    @patch("lexi.get_lexi_data")
    def test_get_lexi_data(self, mock_get_lexi_data):
        mock_df = pd.DataFrame(
            {"data": [1, 2, 3]}, index=pd.date_range("2025-03-03", periods=3)
        )
        mock_get_lexi_data.return_value = mock_df

        result = get_lexi_data(time_range=["2025-03-03", "2025-03-04"], verbose=False)
        pd.testing.assert_frame_equal(result, mock_df)

    @patch("lexi.get_spc_prams")
    def test_get_spc_prams(self, mock_get_spc_prams):
        mock_df = pd.DataFrame(
            {"param": [10, 20, 30]}, index=pd.date_range("2025-03-03", periods=3)
        )
        mock_get_spc_prams.return_value = mock_df

        result = get_spc_prams(time_range=["2025-03-03", "2025-03-04"], verbose=False)
        pd.testing.assert_frame_equal(result, mock_df)

    @patch("lexi.get_exposure_maps")
    def test_get_exposure_maps(self, mock_get_exposure_maps):
        mock_result = {
            "exposure_maps": np.array([[1, 2], [3, 4]]),
            "ra_arr": np.array([0, 1]),
            "dec_arr": np.array([-1, 0]),
        }
        mock_get_exposure_maps.return_value = mock_result

        result = get_exposure_maps(
            time_range=["2025-03-03", "2025-03-04"], verbose=False
        )
        self.assertEqual(result["exposure_maps"].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
