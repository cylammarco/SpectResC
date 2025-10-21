import numpy as np
from spectres import spectres as sp
from spectresc import spectres as sc


# ===========================================================
#  One-dimensional flux arrays
# ===========================================================


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


# ===========================================================
#  Multi-dimensional flux arrays
# ===========================================================


def test_multidim_fluxes_2d_against_spectres():
    """Test 2D flux array: shape (Nspec, Nwave)"""
    inwaves = np.linspace(4000.0, 10000.0, 1000)
    outwaves = np.linspace(4500.0, 8500.0, 731)
    n_spec = 5
    influxes = np.random.random((n_spec, len(inwaves))) * 10000.0
    influxes_err = np.random.random((n_spec, len(inwaves))) * 100.0

    outfluxes_sp, _ = sp(outwaves, inwaves, influxes, influxes_err)
    outfluxes_sc, _ = sc(outwaves, inwaves, influxes, influxes_err)

    assert np.isclose(outfluxes_sp, outfluxes_sc).all()


def test_multidim_fluxes_3d_against_spectres():
    """Test 3D flux array: shape (Nbatch, Nspec, Nwave)"""
    inwaves = np.linspace(4000.0, 10000.0, 1000)
    outwaves = np.linspace(4500.0, 8500.0, 731)
    n_batch, n_spec = 3, 4
    influxes = np.random.random((n_batch, n_spec, len(inwaves))) * 10000.0
    influxes_err = np.random.random((n_batch, n_spec, len(inwaves))) * 100.0

    outfluxes_sp, _ = sp(outwaves, inwaves, influxes, influxes_err)
    outfluxes_sc, _ = sc(outwaves, inwaves, influxes, influxes_err)

    assert np.isclose(outfluxes_sp, outfluxes_sc).all()


def test_multidim_fluxes_sum_conservation():
    """Check flux conservation across multi-spectra 2D array"""
    inwaves = np.linspace(4000.0, 10000.0, 1000)
    outwaves = np.linspace(4500.0, 8500.0, 731)
    influxes = np.random.random((10, len(inwaves))) * 1000.0

    outfluxes = sc(outwaves, inwaves, influxes)
    waves_mask_for_inwaves = (inwaves >= outwaves[0]) & (inwaves <= outwaves[-1])

    assert np.allclose(
        np.sum(influxes[:, waves_mask_for_inwaves], axis=-1) * np.diff(inwaves).mean(),
        np.sum(outfluxes, axis=-1) * np.diff(outwaves).mean(),
        rtol=0.01,
    )
    # the simple masking above will always give a larger sum on the outfluxes
    assert (
        np.sum(influxes[:, waves_mask_for_inwaves], axis=-1) * np.diff(inwaves).mean()
        < np.sum(outfluxes, axis=-1) * np.diff(outwaves).mean()
    ).all()
