"""Print the version number for the current installation."""
import logging


def version():
    """
    Print the lexi version number.

    Returns
    -------
    None.

    """
    from importlib.metadata import version

    ver = version("lexi")
    logging.info("lexi version: " + ver)
