import numpy as np
from spectresc import spectres as sc

inwaves = np.linspace(1, 15, 50)
outwaves = np.linspace(2, 10, 20)
influxes = np.copy(inwaves) * 0.1
r1 = sc(outwaves, inwaves, influxes)
r2 = sc(outwaves, inwaves, influxes, influxes)


print(r1)
print(r2)
