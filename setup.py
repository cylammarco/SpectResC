#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
from setuptools import Extension, setup

setup(
    include_dirs=["/usr/local/lib", numpy.get_include()],
    ext_modules=[
        Extension(
            "spectresc",
            sources=["src/spectresc/spectres.c"],
            extra_compile_args=["-O3", "-fPIC", "-shared"],
        )
    ],
)
