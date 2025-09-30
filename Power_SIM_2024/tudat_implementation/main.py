"""This code was written using tudat-space code version 3.10.14. 
Tudat itself is held under Copyright (c) 2010-2020, Delft University of 
Technology. All rigths reserved.

Most questions about how tudat works can be found in the library documentation
found here: https://docs.tudat.space/en/latest/index.html. Tip: check the 
examples.

For any remaining questions you can reach the author/s below (This list 
can/should be expanded):
- Tomás Reis (2025) - tomas.qreis@gmail.com
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tudat_setup import * 
from assist_functions import *
from power_sim import *
from modes import *
from tumbling_code.main import random_quaternion_tumbling
from tudatpy.astro import element_conversion
from tudatpy.util import result2array
from tudatpy.data import save2txt

# Load tudat spice kernels.
spice.load_standard_kernels()



if __name__ == "__main__":

    # Check value for using tumbling for power calculations instead of specific
    # attitude power. 
    tumblingCheck = True

    ### Spacecraft properties definition. 
    # Defines spacecraft mass in [kg]. 
    sc_mass = 2.8

    # Creates solar array distribution as: [+X, -X, +Y, -Y, +Z, -Z]
    # TODO: Better estimate for maximum power production for each cell.
    solar_cell = 1.08 # [Watt]
    solarArray = [4*solar_cell, 4*solar_cell, 2*solar_cell, 
                   4*solar_cell, 2*solar_cell, 0*solar_cell]
    
    # Total tumbling power based on the sphere of quaternions method in
    # tumbling_code. 

    # NOTE: This tumbling code produces estimates which are an average of 
    # random satellite attitudes; meaning there is some variation with each
    # run of the program. numVals defines the number of values produced, might 
    # be helpful to simulate with more values to see what the variation is. 
    # But that's some statistics stuff that I don't want to do. 
    if tumblingCheck: 
        tumblingPowers = tumbling_powers(solarArray= solarArray, numVals= 1)
        # TODO: Decide whether to do this or a set of tests for different 
        # tumbling powers. 
    
    # Battery capacity. [W*h]
    # Taken from iEPS Type A,B,C datasheet. 
    cellVolt = 3.6      # V
    cellCap = 3.2       # Ah
    cellInSeries = 4
     
    batt_max = cellVolt * cellInSeries * cellCap

    # Battery start charge. [W*h]
    batt_start = batt_max/2

    ### Propagation-related defitions. 
    # Defines simulation start date and time. (YYYY-MM-DD:HH:MM:SS)
    python_date = datetime.fromisoformat("2025-04-21T00:00:00.000000")
    # Converts into tudat time format. 
    tudat_date = time_conversion.datetime_to_tudat(python_date)
    # Defines initial time as seconds since J2000. 
    starting_time = tudat_date.epoch()

    # Defines total propagation time in hours. 
    prop_time = 2.0
    # Defines constant time step in seconds. 
    time_step = 20.0

    # Defines initial keplerian orbital elements. 
    keplerian_elems = np.array([
        8e6,                    # Semi-major axis [m]
        0.05,                   # eccentricity 
        np.deg2rad(86),         # Inclination [rads]. Degrees in ().
        np.deg2rad(0),          # arg of periapsis [rads]
        np.deg2rad(0),          # longitude of ascending node [rads]
        np.deg2rad(0),          # true anomaly [rads]
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

    ### Propagation ###
    # This section propagates the actual orbit to the desired stop time. 
    # Make any changes to environment, orbit or vehicle properties and this
    # needs to be run again. 
    if propagate := False:
        
        # Creates propagation termination settings. 
        time_termination_settings = propagation_setup.propagator.time_termination(
            termination_time= starting_time + prop_time*60**2
        )

        # Defines dependent variables. 
        dependent_variables = [
            # Stores received irradiance in [W/m^2]. Considers eclypse as 
            # on/off, no penumbra. Dependent column 1 (Zero is time)
            propagation_setup.dependent_variable.received_irradiance_shadow_function(
                target_body= "davinci",
                source_body= "Sun"
            ),
            # Stores position vector of spacecraft relative to the sun in the
            # Earth-centered coordinate frame. Dependent columns (2 to 4)
            propagation_setup.dependent_variable.relative_position(
                body= "davinci", 
                relative_body= "Sun"
            ),
            # Stores the rotation matrix for converting from body_fixed to 
            # inertial reference frame (J2000). Dependent columns (5 to 13)
            propagation_setup.dependent_variable.inertial_to_body_fixed_rotation_frame(
                body= "davinci"
            ),

            ### REMOVABLE DEPENDENTS TODO: REMOVE ONCE UNNECESSARY
            # Stores kepler elements. Columns (-1 to -6) 
            # NOTE: Verification purposes only.  
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
            filename= "state_data.csv",
            directory= data_dir
        )
        save2txt(
            solution= dependent_history,
            filename= "dependent_data.csv",
            directory= data_dir
        )

    ### Post Processing ###
    # All the stuff here doesn't require propagation, but reads off saved vals. 
    # Turn off propagate if you just wanna mess around with this. 
    if power_behavior := True:
        # Reads values from saved csv. 
        state_array, dependent_array = read_files()

        if tumblingCheck:
            # TODO: Implement loop that tests power production through all 
            # average powers in the tumbling array. 
            powerSolar = tumblingPowers
        else:
            inertial_to_body_rot_frame = dependent_array[:,5:14]
            solar_pos_relative_to_sat = dependent_array[:,2:5]
            # Defines times array (In hours). 
            times = (dependent_array[:,0] - dependent_array[0,0]) / 60**2 

            # Transforms relative solar position into body-fixed-frame. 
            solar_pos_body_frame = np.zeros(np.shape(
                solar_pos_relative_to_sat))
            for i in range(np.shape(inertial_to_body_rot_frame)[0]):
                solar_pos_body_frame[i,:] = np.matmul(
                    np.reshape(inertial_to_body_rot_frame[i,:], [3,3]),
                    solar_pos_relative_to_sat[i,:]
                )

            # Outputs power production throughout the orbit, not considering 
            # eclipse. 
            solar_pos_body_frame_unit =\
                solar_pos_body_frame / np.linalg.norm(
                    solar_pos_body_frame, axis= 1)[:,None]
            powerSolar = power_output(
                solar_pos_body_frame= solar_pos_body_frame_unit,
                solar_arr= solarArray)
        

        # Extracts shadow function (1: Illuminated to 0: Fully eclipsed)
        shadowArray = dependent_array[:,1]
        # Uses shadow array to get final power production through orbit. 
        powerSolar = powerSolar * shadowArray

        ### TODO: This whole section can be done better. 
        # Imports true anomaly values. 
        trueAnomaly = dependent_array[:,-1] * 180/np.pi

        # Finds closest index to 360º (One orbit) 
        indx = (np.abs(trueAnomaly - 360)).argmin()
        print(indx)

        # Calculates average of power production through orbit. 
        orbitAvg = np.average(powerSolar[:indx])

        print(f"Tumbling power production average: {tumblingPowers} W")
        print(f"Orbit average power: {orbitAvg} W")

        ### Battery charge operations ###

        if batteryChargeCheck := False:
            # Initializes empty battery charge array. 
            batt_current = np.zeros(np.size(times))
            batt_current[0] = batt_start

            # Placeholder. Defines starting mode as the "active" mode. 
            # TODO: replace with a more accurate starting mode. 
            modeCurrent = modes['active']

            # NOTE: Find a way to avoid another for loop here. Something with 
            # integration might work? But needs to detect when maximum battery
            # charge is reached.
            for i in range(np.size(times) -1):
                batt_old = batt_current[i]

                # State machine
                # Currently only flips between "active" and safe modes. 
                modeOld = modeCurrent

                checkRun, switchMode = modeOld.check_run(
                    batteryCharge= batt_current[i], sunlight= shadowArray[i])

                if checkRun:
                    modeCurrent = modeOld
                else:
                    modeCurrent = modes[switchMode]

                # TODO: Implement transient power consumptions. Probably in some 
                # every nth orbit kind of way? Would be best to figure out actual
                # consumption. 
                # - dice: Every 6th orbit. (1.33W * 19s)
                # - short overpass: Every 100th orbit. (4.23W * 130s)
                # - medium overpass: Every 100th orbit. (4.23W * 600s)
                # - long overpass: Every 17th orbit. (4.23W * 690s)
                # - point: Every 6th orbit. (2.69W * 72s)

                powerNet = powerSolar[i] - modeCurrent.powerDrain

                # Computes produced energy in step, adds to current batt charge. 
                charge = (powerNet * time_step) / 60**2      # [W*h]
                
                batt_new = batt_old + charge

                if batt_new >= batt_max:
                    batt_current[i+1] = batt_max
                elif batt_new <= 0.0:
                    print(f"Warning!: Full Discharge.")
                    batt_current[i+1] = 0.0
                else:
                    batt_current[i+1] = batt_new

            # TODO: Plot these nicely. 
            plt.plot(times, powerSolar, "-r", label="Power Produced")
            plt.plot(times, batt_current, "--b", label="Stored Energy")
            plt.grid()
            plt.legend()
            plt.show()

    ######      VERIFICATION
    ### Verified for simple Earth-pointing. 
    if plot_nadir_verification := False:
        # Extracts rotation matrices. 
        inertial_to_body_rot_frame = dependent_array[:,5:14]

        # Extracts position vectors. 
        positions = state_array[:,1:4]
        positions_body_fixed = np.zeros(np.shape(positions))

        # Loops through rows and does frame transformation. 
        for i in range(np.shape(inertial_to_body_rot_frame)[0]):
            positions_body_fixed[i,:] = np.matmul(
                np.reshape(inertial_to_body_rot_frame[i,:], [3,3]),
                positions[i,:]
            )

        # Calculates orbital period (assuming circular).
        period = 2*np.pi*((dependent_array[0,-6]**3)/bodies.get_body("Earth").gravitational_parameter)**(1/2)

        plt.rcParams.update({'font.size': 18})

        fig = plt.figure(figsize=(15,10), dpi=80)

        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)
        
        ax1.plot(state_array[:,0], positions_body_fixed[:,0])
        ax2.plot(state_array[:,0], positions_body_fixed[:,1])
        ax3.plot(state_array[:,0], positions_body_fixed[:,2])

        ax1.grid()
        ax2.grid()
        ax3.grid()

        ax1.set_xticks(np.arange(state_array[0,0], state_array[-1,0], 
                                      period))
        ax2.set_xticks(np.arange(state_array[0,0], state_array[-1,0], 
                                      period))
        ax3.set_xticks(np.arange(state_array[0,0], state_array[-1,0], 
                                      period))
        
        fig.tight_layout()

        plt.show()

        fig = plt.figure(figsize=(15,10), dpi=80)

        ax = fig.add_subplot(1, 1, 1)

        ax.plot(state_array[:,0], positions_body_fixed[:,0])
        ax.plot(state_array[:,0], -np.linalg.norm(state_array[:,1:4], axis= 1),
                "--")
        
        ax.grid()

        ax.set_xticks(np.arange(state_array[0,0], state_array[-1,0], 
                                      period))
        
        fig.tight_layout()

        plt.show()
    
    ######      PLOTTING
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







    
