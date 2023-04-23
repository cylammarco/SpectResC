import numpy as np
from timeit import timeit, Timer

from spectresc import spectres as sc
from spectres import spectres_numba as sn
from spectres import spectres as sp


def call_spectres(size_i, size_o):
    inwaves = np.linspace(0.5, 15, size_i)
    outwaves = np.linspace(1, 10, size_o)
    influxes = np.ones_like(inwaves)
    r = sp(outwaves, inwaves, influxes)
    return r


def call_spectres_err(size_i, size_o):
    inwaves = np.linspace(0.5, 15, size_i)
    outwaves = np.linspace(1, 10, size_o)
    influxes = np.ones_like(inwaves)
    inerrs = 0.1 * influxes
    r = sp(outwaves, inwaves, influxes, spec_errs=inerrs)
    return r


def call_spectresn(size_i, size_o):
    inwaves = np.linspace(0.5, 15, size_i)
    outwaves = np.linspace(1, 10, size_o)
    influxes = np.ones_like(inwaves)
    r = sn(outwaves, inwaves, influxes)
    return r


def call_spectresn_err(size_i, size_o):
    inwaves = np.linspace(0.5, 15, size_i)
    outwaves = np.linspace(1, 10, size_o)
    influxes = np.ones_like(inwaves)
    inerrs = 0.1 * influxes
    r = sn(outwaves, inwaves, influxes, spec_errs=inerrs)
    return r


def call_spectresc(size_i, size_o):
    inwaves = np.linspace(0.5, 15, size_i)
    outwaves = np.linspace(1, 10, size_o)
    influxes = np.ones_like(inwaves)
    r = sc(outwaves, inwaves, influxes)
    return r


def call_spectresc_err(size_i, size_o):
    inwaves = np.linspace(0.5, 15, size_i)
    outwaves = np.linspace(1, 10, size_o)
    influxes = np.ones_like(inwaves)
    inerrs = 0.1 * influxes
    r = sc(outwaves, inwaves, influxes, spec_errs=inerrs)
    return r


if __name__ == "__main__":
    insizes = [100, 1000, 10000]
    outsizes = [100, 1000, 10000]
    repeats = 1000

    _ = call_spectres(10, 10)
    _ = call_spectresn(10, 10)
    _ = call_spectresc(10, 10)

    _ = call_spectres_err(10, 10)
    _ = call_spectresn_err(10, 10)
    _ = call_spectresc_err(10, 10)

    for size_o in outsizes:
        for size_i in insizes:
            print("+" * 80 + "\n")
            print(
                f"Running SpectRes with {size_i} input wavelengths and"
                f" {size_o} output wavelengths for {repeats} times"
            )
            timer = Timer(
                stmt=f"call_spectres({size_i}, {size_o})",
                setup=(
                    "from spectres import spectres as sp; from __main__"
                    " import call_spectres"
                ),
            )
            time = timer.timeit(repeats)
            print(
                (
                    f"Total runtime = {time} s, time per call ="
                    f" {time / repeats} s"
                ),
            )

            # Call the compiled version once with the same argument types so that
            # it's ready for the speed test - this could really be done outside
            # the loop
            call_spectresn(10, 10)
            print(
                "Numba: Running SpectRes with numba with"
                f" {size_i} input wavelengths and {size_o} output wavelengths"
                f" for {repeats} times"
            )
            timer = Timer(
                stmt=f"call_spectresn({size_i}, {size_o})",
                setup=(
                    "from spectres import spectres_numba as sn; from __main__"
                    " import call_spectresn"
                ),
            )
            time_n = timer.timeit(repeats)

            print(
                (
                    f"Numba: Total runtime = {time_n} s, time per call = "
                    f"{time_n / repeats} s"
                ),
            )
            print(f"Numba: Speedup = {time / time_n} times")

            # Using C extension
            print(
                f"C extension: Running SpectResC with {size_i} input"
                f" wavelengths and {size_o} output wavelengths for"
                f" {repeats} times"
            )
            timer = Timer(
                stmt=f"call_spectresc({size_i}, {size_o})",
                setup=(
                    "from spectresc import spectres as sc; from __main__"
                    " import call_spectresc"
                ),
            )
            time_c = timer.timeit(repeats)

            print(
                (
                    f"C extension: Total runtime = {time_c} s, time per call ="
                    f" {time_c / repeats} s"
                ),
            )
            print(f"C extension: Speedup = {time / time_c} times")
            print("\n" + "+" * 80)


    for size_o in outsizes:
        for size_i in insizes:
            print("+" * 80 + "\n")
            print(
                f"Running SpectRes with {size_i} input wavelengths & errors and"
                f" {size_o} output wavelengths for {repeats} times"
            )
            timer = Timer(
                stmt=f"call_spectres_err({size_i}, {size_o})",
                setup=(
                    "from spectres import spectres as sp; from __main__"
                    " import call_spectres_err"
                ),
            )
            time = timer.timeit(repeats)
            print(
                (
                    f"Total runtime = {time} s, time per call ="
                    f" {time / repeats} s"
                ),
            )

            # Call the compiled version once with the same argument types so that
            # it's ready for the speed test - this could really be done outside
            # the loop
            call_spectresn(10, 10)
            print(
                "Numba: Running SpectRes with numba with"
                f" {size_i} input wavelengths & errors and {size_o} output wavelengths"
                f" for {repeats} times"
            )
            timer = Timer(
                stmt=f"call_spectresn_err({size_i}, {size_o})",
                setup=(
                    "from spectres import spectres_numba as sn; from __main__"
                    " import call_spectresn_err"
                ),
            )
            time_n = timer.timeit(repeats)

            print(
                (
                    f"Numba: Total runtime = {time_n} s, time per call = "
                    f"{time_n / repeats} s"
                ),
            )
            print(f"Numba: Speedup = {time / time_n} times")

            # Using C extension
            print(
                f"C extension: Running SpectResC with {size_i} input"
                f" wavelengths & errors and {size_o} output wavelengths for"
                f" {repeats} times"
            )
            timer = Timer(
                stmt=f"call_spectresc_err({size_i}, {size_o})",
                setup=(
                    "from spectresc import spectres as sc; from __main__"
                    " import call_spectresc_err"
                ),
            )
            time_c = timer.timeit(repeats)

            print(
                (
                    f"C extension: Total runtime = {time_c} s, time per call ="
                    f" {time_c / repeats} s"
                ),
            )
            print(f"C extension: Speedup = {time / time_c} times")
            print("\n" + "+" * 80)
