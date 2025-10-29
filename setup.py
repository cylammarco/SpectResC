#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
from setuptools import Extension, setup, find_packages

setup(
    name="spectresc",
    include_dirs=["/usr/local/lib", numpy.get_include()],
    package_dir={"": "src"},  # Tell setuptools packages are under src/
    packages=find_packages(where="src"),  # Find packages in src/
    ext_modules=[
        Extension(
            "spectresc",
            sources=["src/spectresc/spectres.c"],
            extra_compile_args=["-O3", "-fPIC", "-shared"],
        )
    ],
)
