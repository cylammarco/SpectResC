#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import numpy
from setuptools import Extension, setup

setup(
    name="spectresc",
    maintainer="Marco C Lam",
    maintainer_email="mlam@roe.ac.uk",
    version="1.1.0-dev0",
    install_requires=[
        "numpy",
    ],
    description="SpectRes in C",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Marco C Lam",
    download_url="https://github.com/cylammarco/SpectResC",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: C",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
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
