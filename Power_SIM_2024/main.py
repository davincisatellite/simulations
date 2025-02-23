import numpy as np
import matplotlib.pyplot as plt

import orbit 
import attitude as att

theta_vals = np.linspace(0, 720, 200)

semi_major = 6700 #Semi major axis [km]
ecc = 0.05 #Eccentricity
inc = 98 #Inclination [º]
arg = 0 #Argument of Pe [º]
decl = 5 #Solar declination (Positive south of equator) [º]
earth_rad = 6378 #Earth radius [km]
righ_ascention = 20 #Right ascention node from sun-pointing [º]

radii = orbit.posVectorPlanar(theta_vals, semi_major, ecc)

eclipse = np.empty(np.shape(radii)[1])



for i in range(np.shape(radii)[1]):
    eclipse[i] = orbit.eclypseCheck(radii[:,i], earth_rad)




plt.plot(theta_vals, eclipse)

plt.show()