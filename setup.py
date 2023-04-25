#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup, Extension
from pathlib import Path

import numpy

setup(
    name="spectresc",
    maintainer="Marco C Lam",
    maintainer_email="lam@mail.tau.ac.il",
    version="1.0.0.rc2",
    description="SpectRes in C",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Marco C Lam",
    download_url="https://github.com/cylammarco/SpectResC",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: C",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ],
    platforms=["Windows", "Linux", "Mac OS-X"],
    zip_safe=False,
    include_dirs=["/usr/local/lib", numpy.get_include()],
    ext_modules=[
        Extension(
            "spectresc",
            sources=["src/spectresc/spectres.c"],
            extra_compile_args=["-O3", "-fPIC", "-shared"],
        )
    ],
)
