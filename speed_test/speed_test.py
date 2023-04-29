from timeit import Timer

import numpy as np
from matplotlib import pyplot as plt
from spectres import spectres as sp
from spectres import spectres_numba as sn
from spectresc import spectres as sc


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
    # 256, 512, 1024, 2048, 4096
    insizes = 2 ** np.arange(8, 13)
    outsizes = 2 ** np.arange(8, 13)
    repeats = 1000

    sp_time_array = np.zeros((len(insizes), len(outsizes)))
    sp_time_array_with_err = np.zeros_like(sp_time_array)
    sn_time_array = np.zeros_like(sp_time_array)
    sn_time_array_with_err = np.zeros_like(sp_time_array)
    sc_time_array = np.zeros_like(sp_time_array)
    sc_time_array_with_err = np.zeros_like(sp_time_array)

    _ = call_spectres(10, 10)
    _ = call_spectresn(10, 10)
    _ = call_spectresc(10, 10)

    _ = call_spectres_err(10, 10)
    _ = call_spectresn_err(10, 10)
    _ = call_spectresc_err(10, 10)

    for o, size_o in enumerate(outsizes):
        for i, size_i in enumerate(insizes):
            print("+" * 80 + "\n")
            print(
                f"Running SpectRes with {size_i} input wavelengths and"
                f" {size_o} output wavelengths for 100 x {repeats} times"
            )
            timer = Timer(
                stmt=f"call_spectres({size_i}, {size_o})",
                setup=(
                    "from spectres import spectres as sp; from __main__"
                    " import call_spectres"
                ),
            )
            time = np.median(timer.repeat(100, repeats))
            print(
                (
                    f"Total runtime = {time * 100} s, time per call ="
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
                f" for 100 x {repeats} times"
            )
            timer = Timer(
                stmt=f"call_spectresn({size_i}, {size_o})",
                setup=(
                    "from spectres import spectres_numba as sn; from __main__"
                    " import call_spectresn"
                ),
            )
            time_n = np.median(timer.repeat(100, repeats))

            print(
                (
                    f"Numba: Total runtime = {time_n * 100} s, time per call = "
                    f"{time_n / repeats} s"
                ),
            )
            print(f"Numba: Speedup = {time / time_n} times")

            # Using C extension
            print(
                f"C extension: Running SpectResC with {size_i} input"
                f" wavelengths and {size_o} output wavelengths for"
                f" 100 x {repeats} times"
            )
            timer = Timer(
                stmt=f"call_spectresc({size_i}, {size_o})",
                setup=(
                    "from spectresc import spectres as sc; from __main__"
                    " import call_spectresc"
                ),
            )
            time_c = np.median(timer.repeat(100, repeats))

            print(
                (
                    f"C extension: Total runtime = {time_c * 100} s, time per call ="
                    f" {time_c / repeats} s"
                ),
            )
            print(f"C extension: Speedup = {time / time_c} times")
            print("\n" + "+" * 80)

            sp_time_array[o][i] = time
            sn_time_array[o][i] = time_n
            sc_time_array[o][i] = time_c

    np.save("sp_time_array.npy", sp_time_array)
    np.save("sn_time_array.npy", sn_time_array)
    np.save("sc_time_array.npy", sc_time_array)

    for o, size_o in enumerate(outsizes):
        for i, size_i in enumerate(insizes):
            print("+" * 80 + "\n")
            print(
                f"Running SpectRes with {size_i} input wavelengths & errors"
                f" and {size_o} output wavelengths for 100 x {repeats} times"
            )
            timer = Timer(
                stmt=f"call_spectres_err({size_i}, {size_o})",
                setup=(
                    "from spectres import spectres as sp; from __main__"
                    " import call_spectres_err"
                ),
            )
            time = np.median(timer.repeat(100, repeats))
            print(
                (
                    f"Total runtime = {time * 100} s, time per call ="
                    f" {time / repeats} s"
                ),
            )

            # Call the compiled version once with the same argument types so that
            # it's ready for the speed test - this could really be done outside
            # the loop
            call_spectresn(10, 10)
            print(
                f"Numba: Running SpectRes with numba with {size_i} input"
                f" wavelengths & errors and {size_o} output wavelengths for"
                f" 100 x {repeats} times"
            )
            timer = Timer(
                stmt=f"call_spectresn_err({size_i}, {size_o})",
                setup=(
                    "from spectres import spectres_numba as sn; from __main__"
                    " import call_spectresn_err"
                ),
            )
            time_n = np.median(timer.repeat(100, repeats))

            print(
                (
                    f"Numba: Total runtime = {time_n * 100} s, time per call = "
                    f"{time_n / repeats} s"
                ),
            )
            print(f"Numba: Speedup = {time / time_n} times")

            # Using C extension
            print(
                f"C extension: Running SpectResC with {size_i} input"
                f" wavelengths & errors and {size_o} output wavelengths for"
                f" 100 x {repeats} times"
            )
            timer = Timer(
                stmt=f"call_spectresc_err({size_i}, {size_o})",
                setup=(
                    "from spectresc import spectres as sc; from __main__"
                    " import call_spectresc_err"
                ),
            )
            time_c = np.median(timer.repeat(100, repeats))

            print(
                (
                    f"C extension: Total runtime = {time_c * 100} s, time per call ="
                    f" {time_c / repeats} s"
                ),
            )
            print(f"C extension: Speedup = {time / time_c} times")
            print("\n" + "+" * 80)

            sp_time_array_with_err[o][i] = time
            sn_time_array_with_err[o][i] = time_n
            sc_time_array_with_err[o][i] = time_c

    np.save("sp_time_array_with_err.npy", sp_time_array_with_err)
    np.save("sn_time_array_with_err.npy", sn_time_array_with_err)
    np.save("sc_time_array_with_err.npy", sc_time_array_with_err)
