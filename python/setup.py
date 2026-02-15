"""Build script for the gaitkit C extension.

This file is required alongside pyproject.toml so that
`pip install .` (or `python -m build`) compiles the native
C acceleration module shipped with gaitkit.
"""

import sys

from setuptools import Extension, setup

if sys.platform.startswith("win"):
    _extra_compile_args = ["/O2"]
else:
    _extra_compile_args = ["-O3"]

ext_modules = [
    Extension(
        name="gaitkit.native._gait_native",
        sources=["src/gaitkit/native/_gait_native.c"],
        extra_compile_args=_extra_compile_args,
    )
]

setup(ext_modules=ext_modules)
