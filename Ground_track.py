from Ground_track_utilities import *

orbit_1 = [
        6880000,  # semi_major axis
        0.01,  # eccentricity
        np.pi/2 + 9/180 * np.pi,  # inclination
        0,  # argument_of_periapsis
        0,  # longitude_of_ascending_node
        0,  # true_anomaly
    ]

orbit_2 = [6890000, 0.015, 98 * np.pi/180, 52*np.pi/180, 4*np.pi/180, 0]

orbit_3 = [6900000, 0.01, np.pi/2 + 9/180 * np.pi, 0, 0, 0]

orbit_4 = [6920000, 0.015, 98 * np.pi/180, 52*np.pi/180, 4*np.pi/180, 0]

run_analysis(orbit_1)
run_analysis(orbit_2)
run_analysis(orbit_3)
run_analysis(orbit_4)

