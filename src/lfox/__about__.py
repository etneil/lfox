# SPDX-FileCopyrightText: 2023-present Ethan Neil <ethan.neil@colorado.edu>
#
# SPDX-License-Identifier: MIT

# The version lives in pyproject.toml and is read back out of the installed
# distribution metadata, so there is exactly one place to bump it.
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lfox")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"
