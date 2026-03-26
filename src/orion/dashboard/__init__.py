"""Orion Dashboard static files."""
import os

BUILD_DIR = os.path.join(os.path.dirname(__file__), "build")


def get_build_path():
    """Return the path to the dashboard build directory."""
    if not os.path.isdir(BUILD_DIR):
        raise RuntimeError(
            f"Cannot find dashboard static files to run frontend. "
            f"Expected to be located at: {BUILD_DIR}"
        )
    return BUILD_DIR
