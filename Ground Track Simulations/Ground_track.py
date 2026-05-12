from Ground_track_utilities import *

keplerian_state = [6890000,  # semi_major axis
                   0.010,  # eccentricity
                   52 * np.pi / 180,  # inclination
                   0.0,  # argument_of_periapsis
                   0 * np.pi / 180,  # longitude_of_ascending_node
                   0.0]

sim_id = 1

reference_area = 0.02725090
drag_coefficient = 2.2
radiation_pressure_coefficient = 1.2

satellite_mass = 2.2

start_date = (2028, 1, 1, 0, 0, 0.0)
end_date = (2028, 1, 2, 1, 0, 20.0)

run_analysis(keplerian_state = keplerian_state,
             reference_area = reference_area,
             drag_coefficient = drag_coefficient,
             radiation_pressure_coefficient = radiation_pressure_coefficient,
             satellite_mass = satellite_mass,
             start_date = start_date,
             end_date = end_date,
             sim_id = sim_id
             )
