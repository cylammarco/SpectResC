import numpy as np
from timeit import timeit, Timer

from spectresc import spectres as sc
from spectres import spectres_numba as sn
from spectres import spectres as sp


def call_spectres(size):
    inwaves = np.linspace(0.5, 15, size)
    outwaves = np.linspace(1, 10, 200)
    influxes = np.ones_like(inwaves)
    r = sp(outwaves, inwaves, influxes)
    return r


"""
def call_spectres_err(size):
    inwaves = np.linspace(0.5, 15, size)
    outwaves = np.linspace(1, 10, 200)
    influxes = np.ones_like(inwaves)
    inerrs = 0.1*influxes
    r = sp(outwaves, inwaves, influxes, spec_errs = inerrs)
    return r
"""


def call_spectresn(size):
    inwaves = np.linspace(0.5, 15, size)
    outwaves = np.linspace(1, 10, 200)
    influxes = np.ones_like(inwaves)
    r = sn(outwaves, inwaves, influxes)
    return r


"""
def call_spectresn_err(size):
    inwaves = np.linspace(0.5, 15, size)
    outwaves = np.linspace(1, 10, 200)
    influxes = np.ones_like(inwaves)
    inerrs = 0.1*influxes
    r = sn(outwaves, inwaves, influxes, spec_errs = inerrs)
    return r
"""


def call_spectresc(size):
    inwaves = np.linspace(0.5, 15, size)
    outwaves = np.linspace(1, 10, 200)
    influxes = np.ones_like(inwaves)
    r = sc(outwaves, inwaves, influxes)
    return r


"""
def call_spectresc_err(size):
    inwaves = np.linspace(0.5, 15, size)
    outwaves = np.linspace(1, 10, 200)
    influxes = np.ones_like(inwaves)
    inerrs = 0.1*influxes
    r = sc(outwaves, inwaves, influxes, spec_errs = inerrs)
    return r
"""

if __name__ == "__main__":
    insizes = [1000, 10000, 100000]
    repeats = 1000

    """
    a = call_spectres_err()
    b = call_spectresn_err()
    c = call_spectresc_err()
    """
    for size in insizes:
        print(
            f"Running spectres with {size} input wavelengths for "
            f"{repeats} times"
        )
        timer = Timer(
            stmt="call_spectres()",
            setup=(
                "from spectres import spectres as sp; from __main__"
                " import call_spectres"
            ),
        )
        time = timer.timeit(repeats)
        print(
            "Total runtime = {time} s, time per call = {time / repeats} s",
        )

        # Call the compiled version once with the same argument types so that
        # it's ready for the speed test - this could really be done outside
        # the loop
        call_spectresn()
        print(
            f"Running compiled version of spectres with {size} input "
            f"wavelengths for {repeats} times"
        )
        timer = Timer(
            stmt="call_spectresn()",
            setup=(
                "from spectres import spectres as sp; from __main__"
                " import call_spectresn"
            ),
        )
        time_c = timer.timeit(repeats)

        print(
            f"Total runtime = {time_c} s, time per call = "
            f"{time_c / repeats} s",
        )
        print(f"Speedup thanks to Numba = {time / time_c}")

        # Using C extension
        print(
            f"Running SpectResC with {size} input "
            f"wavelengths for {repeats} times"
        )
        timer = Timer(
            stmt="call_spectresc()",
            setup=(
                "from spectresc import spectres as sc; from __main__"
                " import call_spectresc"
            ),
        )
        time_d = timer.timeit(repeats)

        print(
            f"Total runtime = {time_d} s, time per call = "
            f"{time_d / repeats} s",
        )
        print(f"Speedup thanks to C = {time / time_d}")
