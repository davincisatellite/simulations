"""This code was written using tudat-space code version 3.10.14. 
Tudat itself is held under Copyright (c) 2010-2020, Delft University of 
Technology. All rigths reserved.

Most questions about how tudat works can be found in the library documentation
found here: https://docs.tudat.space/en/latest/index.html. Tip: check the 
examples.

For any remaining questions you can reach the author/s below (This list can be 
expanded):
- Tomás Reis (2025) - tomas.qreis@gmail.com
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tudat_setup import * 
from tudatpy.astro import element_conversion
from tudatpy.util import result2array
from tudatpy.data import save2txt

# Load spice kernels.
spice.load_standard_kernels()

if __name__ == "__main__":

    ### Spacecraft properties definition. 
    # Defines spacecraft mass in [kg]. 
    sc_mass = 2.8

    # Creates solar array distribution as: [+X, -X, +Y, -Y, +Z, -Z]
    # TODO: Better estimate for maximum power production for each cell.
    solar_cell = 1.08 # [Watt]
    solar_array = [4*solar_cell, 4*solar_cell, 4*solar_cell, 
                   4*solar_cell, 2*solar_cell, 2*solar_cell]

    ### Propagation-related defitions. 
    # Defines simulation start date and time. (YYYY-MM-DD:HH:MM:SS)
    python_date = datetime.fromisoformat("2025-04-21T00:00:00.000000")
    # Converts into tudat time format. 
    tudat_date = time_conversion.datetime_to_tudat(python_date)
    # Defines initial time as seconds since J2000. 
    starting_time = tudat_date.epoch()

    # Defines total propagation time in hours. 
    prop_time = 12.0
    # Defines constant time step in seconds. 
    time_step = 60.0

    # Defines initial keplerian orbital elements. 
    keplerian_elems = np.array([
        8e6,                    # Semi-major axis [m]
        0.0,                    # eccentricity 
        np.deg2rad(90),         # Inclination [rads]. Degrees in ().
        0.0,                    # arg of periapsis [rads]
        0.0,                    # longitude of ascending node [rads]
        0.0,                    # true anomaly [rads]
    ])

    # Defines initial spacecraft attitude versus J2000 coordinate frame.
    # TODO: Make this relative to some more practical coordinate frame. 
    # See tudat_setup.py file. 
    # TODO: Check whether this is necessary at all. 
    
    initial_att = np.array(
        [[1,0,0], 
         [0,1,0],
         [0,0,1]]
    )

    # Creates environment bodies. 
    bodies = create_bodies(
        sc_mass= sc_mass,
        initial_att= initial_att,
        rotation= True,
        starting_time= starting_time,
        time_step= time_step
    )

    bodies = create_rotational_settings(
        bodies= bodies,
        time_step= time_step
        )

    # Defines initial state from keplerian orbital parameters. 
    initial_state = element_conversion.keplerian_to_cartesian(
        keplerian_elements= keplerian_elems, 
        gravitational_parameter= bodies.get("Earth").gravitational_parameter
    )

    # This section propagates the actual orbit to the desired stop time. 
    if propagate := True:
        
        # Creates propagation termination settings. 
        time_termination_settings = propagation_setup.propagator.time_termination(
            termination_time= starting_time + prop_time*60**2
        )

        # Defines dependent variables. 
        dependent_variables = [
            # Stores received irradiance in [W/m^2]. Considers eclypse as 
            # on/off, no penumbra. 
            propagation_setup.dependent_variable.received_irradiance(
                target_body= "davinci",
                source_body= "Sun"
            ),
            # Stores position vector of spacecraft relative to the sun in the
            # Earth-centered coordinate frame. 
            propagation_setup.dependent_variable.relative_position(
                body= "davinci", 
                relative_body= "Sun"
            ),
            # Stores the rotation matrix for converting from body_fixed to 
            # inertial reference frame (J2000).
            propagation_setup.dependent_variable.inertial_to_body_fixed_rotation_frame(
                body= "davinci"
            ),

            ### REMOVABLE DEPENDENTS TODO: REMOVE ONCE UNNECESSARY
            # Stores kepler elements. NOTE: Verification purposes only. 
            propagation_setup.dependent_variable.keplerian_state(
                body= "davinci",
                central_body= "Earth"
            )

        ]

        # Retrieves propagation settings.
        propagation_settings = create_prop_settings(
            bodies= bodies,
            initial_state= initial_state,
            initial_time= starting_time,
            termination_condition= time_termination_settings,
            fixed_step_size= time_step,
            dependent_variables= dependent_variables,
            atmo_sim= False
        )

        # Propagate dynamics. 
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            bodies, 
            propagation_settings
        )
        
        # Extract state history and dependent variables. 
        state_history = dynamics_simulator.propagation_results.state_history
        dependent_history = dynamics_simulator.propagation_results.dependent_variable_history
        state_array = result2array(state_history)
        dependent_array = result2array(dependent_history)

        # Saves values to data files. 
        # TODO: Rework this into a csv format. 
        save2txt(
            solution= state_history, 
            filename= "state_data.dat",
            directory= data_dir
        )
        save2txt(
            solution= dependent_history,
            filename= "dependent_data.dat",
            directory= data_dir
        )


    if plot_data := False:

        #######  Plotting Shenanigans  #######
        # TODO: Put this in its own little python file.  
        plt.rcParams.update({'font.size': 18})

        ax = plt.figure().add_subplot(projection='3d')
        ax1 = plt.figure().add_subplot()

        times = state_array[:,0] / 60**2

        ax.plot(state_array[:,1], state_array[:,2], state_array[:,3])

        ax1.plot(times, dependent_array[:,1])

        ax.set_zlim(-5e6, 5e6)
        ax.set_xlim(-5e6, 5e6)
        ax.set_ylim(-5e6, 5e6)

        plt.show()

        #######  ---  #######







    
