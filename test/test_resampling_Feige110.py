import os

# from matplotlib import pyplot as plt
import numpy as np
from spectresc import spectres as sc

try:
    PATH = os.path.dirname(os.path.realpath(__file__))
except:
    PATH = os.path.dirname(os.path.realpath(__name__))

data = np.loadtxt(os.path.join(PATH, "fFeige110.dat"))

inwaves = data[:, 0]
outwaves = data[:, 0][::10]
influxes = data[:, 1]


def test_resample_standard_star():
    r1 = sc(outwaves, inwaves, influxes)
    r2, _ = sc(outwaves, inwaves, influxes, np.sqrt(influxes))

    assert np.isclose(np.nansum(r1), np.nansum(r2))

    # plt.figure(figsize=(8, 8))
    # plt.plot(inwaves, influxes, label="Input")
    # plt.plot(outwaves, r1, label="Resampled")
    # plt.plot(outwaves, r2, label="Resampled with uncertainties")
    # plt.ylim(0, 6e-13)
    # plt.legend()
    # plt.savefig(os.path.join(PATH, "fFeige110.png"))
