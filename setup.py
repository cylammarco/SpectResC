#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import Extension, setup, find_packages
import sys

import numpy

# Platform-specific compile args
if sys.platform == "win32":
    extra_compile_args = ["/O2"]  # Optimize for Windows
else:
    extra_compile_args = ["-O3", "-fPIC"]  # Optimize for Unix/macOS

setup(
    name="spectresc",
    include_dirs=["/usr/local/lib", numpy.get_include()],
    package_dir={"": "src"},  # Tell setuptools packages are under src/
    packages=find_packages(where="src"),  # Find packages in src/
    ext_modules=[
        Extension(
            "spectresc",
            sources=["src/spectresc/spectres.c"],
            extra_compile_args=extra_compile_args,
        )
    ],
)
