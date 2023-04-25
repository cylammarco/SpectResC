# SpectResC
[![Coverage Status](https://coveralls.io/repos/github/cylammarco/SpectResC/badge.svg?branch=main)](https://coveralls.io/github/cylammarco/SpectResC?branch=main)
[![Readthedocs Status](https://readthedocs.org/projects/spectres/badge/?version=latest&style=flat)](https://spectres.readthedocs.io)
[![arXiv](https://img.shields.io/badge/arXiv-1705.05165-00ff00.svg)](https://arxiv.org/abs/1705.05165)
[![PyPI version](https://badge.fury.io/py/spectresc.svg)](https://badge.fury.io/py/spectresc)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7863171.svg)](https://doi.org/10.5281/zenodo.7863171)

This is a Python package written with C extension to provide significant performance gain on [SpectRes](https://github.com/ACCarnall/SpectRes), and some performance over the numba implementation:

![alt text](https://github.com/cylammarco/SpectResC/blob/main/speed_test/speed_test.png?raw=true)

We keep the implementation as close to SpectRes as possible. As of SpectRes v2.2.0, we do not see discrepant results between using SpectRes and SpectReC.

## Installation

SpectResC can be installed using pip

```
pip install spectresc
```

## Documentation

Please refer to the original SpectRes for the [documentation](https://spectres.readthedocs.io).

## Citation

If you have made use of SpectResC, please reference:

1. the original SpectRes [arXiv article](https://arxiv.org/abs/1705.05165)
2. the [zenodo DOI](https://zenodo.org/record/7863171) for SpectResC
