#!/usr/bin/env python
"""
Module to install extra dependencies
=====================================

Install git-based dependencies that cannot be declared in pyproject.toml
due to PyPI restrictions on direct references.

"""
import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

SHORT_DESCRIPTION = "Install extra dependencies from git"

GIT_EXTRAS_PATH = Path(__file__).parents[2] / "git_extras.toml"


def _load_toml(path):
    """Load a TOML file, using tomllib (3.11+) or tomli (3.10)."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    with open(path, "rb") as f:
        return tomllib.load(f)


def _filter_dependencies(dependencies):
    """Filter dependencies based on PEP 508 environment markers."""
    from packaging.requirements import Requirement

    filtered = []
    for dep_str in dependencies:
        req = Requirement(dep_str)
        if req.marker is None or req.marker.evaluate():
            filtered.append(dep_str)
    return filtered


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    install_parser = parser.add_parser(
        "install", help=SHORT_DESCRIPTION, description=SHORT_DESCRIPTION
    )

    install_parser.add_argument(
        "extra",
        nargs="?",
        help="Name of the extra to install (e.g., dehb, bohb, hebo, track). "
        "Use --list to see available extras.",
    )

    install_parser.add_argument(
        "--list",
        action="store_true",
        dest="list_extras",
        help="List available extras and their dependencies",
    )

    install_parser.set_defaults(func=main)

    return install_parser


def list_extras():
    """List available git extras and their dependencies."""
    data = _load_toml(GIT_EXTRAS_PATH)

    print("Available extras (install with: orion install <extra>):\n")
    for name, section in sorted(data.items()):
        deps = section.get("dependencies", [])
        filtered = _filter_dependencies(deps)
        if filtered:
            print(f"  {name}")
            for dep in filtered:
                print(f"    - {dep}")
            print()
        else:
            print(
                f"  {name}  (no installable dependencies for Python {sys.version.split()[0]})"
            )
            print()


def main(args):
    """Install extra dependencies from git."""
    if args.get("list_extras"):
        list_extras()
        return 0

    extra = args.get("extra")
    if not extra:
        print("Usage: orion install <extra>")
        print("       orion install --list")
        return 1

    data = _load_toml(GIT_EXTRAS_PATH)

    if extra not in data:
        available = ", ".join(sorted(data.keys()))
        print(f"Unknown extra: '{extra}'")
        print(f"Available extras: {available}")
        return 1

    deps = data[extra].get("dependencies", [])
    to_install = _filter_dependencies(deps)

    if not to_install:
        print(
            f"Extra '{extra}' has no installable dependencies "
            f"for Python {sys.version.split()[0]}"
        )
        return 0

    print(f"Installing extra '{extra}':")
    for dep in to_install:
        print(f"  - {dep}")
    print()

    returncode = subprocess.call([sys.executable, "-m", "pip", "install"] + to_install)

    if returncode == 0:
        print(f"\nExtra '{extra}' installed successfully.")
    else:
        print(f"\nInstallation failed (exit code {returncode}).")

    return returncode
