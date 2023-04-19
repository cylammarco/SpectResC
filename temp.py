import numpy as np
from spectresc import spectres as sc

inwaves = np.linspace(0.5, 15, 100)
outwaves = np.linspace(1, 10, 200)
influxes = np.ones_like(inwaves)
r1 = sc(outwaves, inwaves, influxes)
r2 = sc(outwaves, inwaves, influxes, influxes)
