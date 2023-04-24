import numpy as np
from spectresc import spectres as sc
from spectres import spectres as sp


def test_with_random_arrays():
    inwaves = np.linspace(4000.0, 10000.0, 1000)
    outwaves = np.linspace(4500.0, 8500.0, 731)
    outwaves_2 = np.linspace(4500.0, 8500.0, 679)
    influxes = np.random.random(len(inwaves)) * 10000.0
    influxes_err = np.random.random(len(inwaves)) * 100.0

    outfluxes_1 = sc(outwaves, inwaves, influxes)
    outfluxes_2, _ = sc(outwaves, inwaves, influxes, influxes_err)
    outfluxes_3, _ = sc(outwaves_2, inwaves, influxes, influxes_err)

    assert np.isclose(np.sum(outfluxes_1), np.sum(outfluxes_2))
    assert np.isclose(
        np.sum(outfluxes_2 * (outwaves[1] - outwaves[0])),
        np.sum(outfluxes_3 * (outwaves_2[1] - outwaves_2[0])),
        rtol=0.01,
    )


def test_with_random_arrays_against_spectres():
    inwaves = np.linspace(4000.0, 10000.0, 1000)
    outwaves = np.linspace(4500.0, 8500.0, 731)
    outwaves_2 = np.linspace(4500.0, 8500.0, 679)
    influxes = np.random.random(len(inwaves)) * 10000.0
    influxes_err = np.random.random(len(inwaves)) * 100.0

    outfluxes_1_sp = sp(outwaves, inwaves, influxes)
    outfluxes_2_sp, _ = sp(outwaves, inwaves, influxes, influxes_err)
    outfluxes_3_sp, _ = sp(outwaves_2, inwaves, influxes, influxes_err)

    outfluxes_1_sc = sc(outwaves, inwaves, influxes)
    outfluxes_2_sc, _ = sc(outwaves, inwaves, influxes, influxes_err)
    outfluxes_3_sc, _ = sc(outwaves_2, inwaves, influxes, influxes_err)

    assert np.isclose(outfluxes_1_sp, outfluxes_1_sc).all()
    assert np.isclose(outfluxes_2_sp, outfluxes_2_sc).all()
    assert np.isclose(outfluxes_3_sp, outfluxes_3_sc).all()


def test_with_random_arrays_against_spectres_output_range_mismatched_left():
    inwaves = np.linspace(4000.0, 10000.0, 1000)
    outwaves = np.linspace(3500.0, 8500.0, 731)
    outwaves_2 = np.linspace(3500.0, 8500.0, 679)
    influxes = np.random.random(len(inwaves)) * 10000.0
    influxes_err = np.random.random(len(inwaves)) * 100.0

    outfluxes_1_sp = sp(outwaves, inwaves, influxes)
    outfluxes_2_sp, _ = sp(outwaves, inwaves, influxes, influxes_err)
    outfluxes_3_sp, _ = sp(outwaves_2, inwaves, influxes, influxes_err)

    outfluxes_1_sc = sc(outwaves, inwaves, influxes)
    outfluxes_2_sc, _ = sc(outwaves, inwaves, influxes, influxes_err)
    outfluxes_3_sc, _ = sc(outwaves_2, inwaves, influxes, influxes_err)

    assert np.isclose(outfluxes_1_sp, outfluxes_1_sc, equal_nan=True).all()
    assert np.isclose(outfluxes_2_sp, outfluxes_2_sc, equal_nan=True).all()
    assert np.isclose(outfluxes_3_sp, outfluxes_3_sc, equal_nan=True).all()


def test_with_random_arrays_against_spectres_output_range_mismatched_right():
    inwaves = np.linspace(4000.0, 10000.0, 1000)
    outwaves = np.linspace(4500.0, 11500.0, 731)
    outwaves_2 = np.linspace(4500.0, 11500.0, 679)
    influxes = np.random.random(len(inwaves)) * 10000.0
    influxes_err = np.random.random(len(inwaves)) * 100.0

    outfluxes_1_sp = sp(outwaves, inwaves, influxes)
    outfluxes_2_sp, _ = sp(outwaves, inwaves, influxes, influxes_err)
    outfluxes_3_sp, _ = sp(outwaves_2, inwaves, influxes, influxes_err)

    outfluxes_1_sc = sc(outwaves, inwaves, influxes)
    outfluxes_2_sc, _ = sc(outwaves, inwaves, influxes, influxes_err)
    outfluxes_3_sc, _ = sc(outwaves_2, inwaves, influxes, influxes_err)

    assert np.isclose(outfluxes_1_sp, outfluxes_1_sc, equal_nan=True).all()
    assert np.isclose(outfluxes_2_sp, outfluxes_2_sc, equal_nan=True).all()
    assert np.isclose(outfluxes_3_sp, outfluxes_3_sc, equal_nan=True).all()
