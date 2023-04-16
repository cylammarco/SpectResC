from distutils.core import setup, Extension
import numpy

setup(
    name='spectresc',
    version='0.1',
    description='Python package for the spectres function written in C',
    include_dirs=['/usr/local/lib', numpy.get_include()],
    ext_modules=[
        Extension('spectresc',
        sources=['src/spectresc/spectres.c'],
        extra_compile_args=['-O3', '-fPIC', '-shared'])
    ]
)