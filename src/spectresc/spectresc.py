import os
import ctypes

import numpy as np

# Load the shared library
libname = "spectresc"
if os.name == "nt":
    libname += ".dll"
else:
    libname += ".so"

libpath = os.path.join(os.path.dirname(__file__), "lib", libname)
lib = ctypes.CDLL(libpath)

# Define the argument and return types for the C function
lib.spectres.argtypes = [
    ctypes.py_object,
    ctypes.py_object,
    ctypes.py_object,
    ctypes.py_object,
    ctypes.c_double,
    ctypes.c_bool,
]
lib.spectres.restype = [
    ctypes.py_object,
    ctypes.py_object,
]


def spectres(
    new_wavs, spec_wavs, spec_fluxes, spec_errs=None, fill=-10.0, verbose=True
):
    """
    Function for resampling spectra (and optionally associated
    uncertainties) onto a new wavelength basis.

    Parameters
    ----------

    new_wavs : numpy.ndarray
        Array containing the new wavelength sampling desired for the
        spectrum or spectra.

    spec_wavs : numpy.ndarray
        1D array containing the current wavelength sampling of the
        spectrum or spectra.

    spec_fluxes : numpy.ndarray
        Array containing spectral fluxes at the wavelengths specified in
        spec_wavs, last dimension must correspond to the shape of
        spec_wavs. Extra dimensions before this may be used to include
        multiple spectra.

    spec_errs : numpy.ndarray (optional)
        Array of the same shape as spec_fluxes containing uncertainties
        associated with each spectral flux value.

    fill : float (optional)
        Where new_wavs extends outside the wavelength range in spec_wavs
        this value will be used as a filler in new_fluxes and new_errs.

    verbose : bool (optional)
        Setting verbose to False will suppress the default warning about
        new_wavs extending outside spec_wavs and "fill" being used.

    Returns
    -------

    new_fluxes : numpy.ndarray
        Array of resampled flux values, first dimension is the same
        length as new_wavs, other dimensions are the same as
        spec_fluxes.

    new_errs : numpy.ndarray
        Array of uncertainties associated with fluxes in new_fluxes.
        Only returned if spec_errs was specified.
    """

    # Call the C function with the input arrays
    new_fluxes, new_fluxes_err = lib.spectres(
        new_wavs, spec_wavs, spec_fluxes, spec_errs, fill, verbose
    )

    return new_fluxes, new_fluxes_err
