# lexi/__init__.py

# Import the version from your setup.py file
from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution("lexi").version
except DistributionNotFound:
    # Package is not installed
    __version__ = "0.0.0"
