# lexi/__init__.py

# Import the version from your setup.py file
from pkg_resources import get_distribution, DistributionNotFound

# Add the docstring to the package
__doc__ = """
The lexi is a package developed using the Python programming language. The package is
designed to provide a simple list of functions to work with the LEXI dataset. The package has
following usable modules:
    - get_spc_prams: This module is used to get the spacecraft parameters from the LEXI dataset using
      a specified time range.
    - get_exposure_maps: This module is used to get the exposure maps from the LEXI dataset using a
      specified time range and some other input parameters.
    - get_sky_backgrounds: This module is used to get the sky backgrounds from the LEXI dataset which
      corresponds to the exposure maps. The module uses the exposure maps to get the sky backgrounds.
    - get_lexi_images: This module is used to get the LEXI images from the LEXI dataset using a
      specified time range and some other input parameters. The module uses the exposure maps and sky
      backgrounds to get the LEXI images. One can either get a background corrected image or a raw
      image from the data set.

The package is developed by the LEXI team at the Boston University.
For more information, please visit the LEXI website at https://lexi-bu.github.io/ or read the README file.
"""

try:
    __version__ = get_distribution("lexi").version
except DistributionNotFound:
    # Package is not installed
    __version__ = "0.0.0"
