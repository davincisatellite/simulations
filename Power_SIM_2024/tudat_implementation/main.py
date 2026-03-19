"""This code was written using tudat-space code version 3.10.14. 
Tudat itself is held under Copyright (c) 2010-2020, Delft University of 
Technology. All rights reserved.

Most questions about how tudat works can be found in the library documentation
found here: https://docs.tudat.space/en/latest/index.html. Tip: check the 
examples.

For any remaining questions you can reach the author/s below (This list 
can/should be expanded):
- Tomás Reis (2025) - tomas.qreis@gmail.com
"""

import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tudat_setup import * 
from assist_functions import *
from power_sim import *
from modes_conditions import *

from datetime import datetime

# Forces re-propagation of values.
forceProp = False

def single_orbit(args):
    """Worker function for a single orbit propagation."""
    (eccentricity, inclination, semiMajorAxis,
     propDurationTime, timeStep, propStartTime,
     tumblingPowers, propagationDir, scMass) = args

    # Checks if an .npz file with propagation data for a given orbit has already been saved
    # assumes only eccentricity, inclination, and semi major axis are varying
    propagation_file = propagationDir + f"ecc_{eccentricity}_inc_{inclination}_a_{semiMajorAxis}.npz"

    if os.path.exists(propagation_file) and not(forceProp):
        # If previously ran, just take from saved data
        print("using saved data c:")
        propagation_data    = np.load(propagation_file)
        stateArr            = propagation_data["stateArr"]
        dependentArr        = propagation_data["dependentArr"]
    else:
        print("propagating...")
        # Each worker needs its own bodies instance — tudat objects aren't picklable
        bodies_local = create_bodies(
            sc_mass=scMass,
            initial_att=np.eye(3),
            rotation=True,
            starting_time=propStartTime,
            time_step=timeStep
        )

        bodies_local = create_rotational_settings(bodies=bodies_local, time_step=timeStep)

        # Propagates the orbit if this combination was not analysed before
        _, _, stateArr, dependentArr = propagate_orbit(
            propDurationTime=propDurationTime,
            timeStep=timeStep,
            stateStartKep=np.array([
                semiMajorAxis, eccentricity, np.deg2rad(inclination),
                np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)
            ]),
            propStartTime=propStartTime,
            bodies=bodies_local
        )
        np.savez(propagation_file, stateArr=stateArr, dependentArr=dependentArr)

    orbitAvg = orbit_average(
        stateArr=stateArr,
        dependentArr=dependentArr,
        tumblingPowers=tumblingPowers,
        tumblingCheck=True,
    )

    return (eccentricity, inclination, semiMajorAxis, orbitAvg)

if __name__ == "__main__":
    # Reproducibility of the power tumbling averages - quarternions are randomly generated
    np.random.seed(42)

    # power data
    dataDir = f"data/"

    # propagation data
    propagationDir = f"data_propagation/"
    os.makedirs(propagationDir, exist_ok=True)

    # Reads run count.
    with open("runcount.txt", "r") as f:
        runCount = int(f.read())

    # Check value for using tumbling for power calculations instead of specific
    # attitude power. 
    tumblingCheck = True

    ### Spacecraft properties definition. 
    # Defines spacecraft mass in [kg]. 
    scMass = 2.8

    # Creates solar array distribution as: [+X, -X, +Y, -Y, +Z, -Z]
    # baseSolarArray = [4 * solar_cell, 4 * solar_cell, 4 * solar_cell,
    #               4 * solar_cell, 2 * solar_cell, 2 * solar_cell]
    # TODO: Better estimate for maximum power production for each cell.
    solar_cell = 1.08 # [Watt]
    solarArray = [4*solar_cell, 4*solar_cell, 0*solar_cell,
                   4*solar_cell, 2*solar_cell, 0*solar_cell]
    
    # Total tumbling power based on the sphere of quaternions method in
    # tumbling_code. 

    # NOTE: This tumbling code produces estimates which are an average of 
    # random satellite attitudes; meaning there is some variation with each
    # run of the program. numVals defines the number of values produced, might 
    # be helpful to simulate with more values to see what the variation is. 
    # But that's some statistics stuff that I don't want to do atm.  -Tom.
    if tumblingCheck: 
        tumblingPowers = tumbling_powers(solarArray= solarArray, numVals= 1)
        # TODO: Decide whether to do this or a set of tests for different 
        # tumbling powers. 
    
    # Battery capacity. [W*h]
    # Taken from iEPS Type A,B,C datasheet. 
    cellVolt = 3.6      # V
    cellCap = 3.2       # Ah
    cellInSeries = 4
    battMax = cellVolt * cellInSeries * cellCap
    # Battery start charge. [W*h]
    battStart = battMax/2

    # Initial date and time for propagation. 
    propStartISO = "2025-06-21T07:00:00.000000"
    # Defines total propagation time in hours. 
    propDurationTime = 5.0
    # Defines constant time step in seconds. 
    timeStep = 20.0
    # Creates times array.
    times = np.arange(start= 0.0, stop= propDurationTime*60**2, step= timeStep)

    # Simulation range for orbital power averages. 
    semiMajorVals = np.linspace(start= 6720e3, stop= 6920e3, num= 21)
    eccVals = np.linspace(start= 0.0, stop= 0.015, num= 2)
    incVals = np.linspace(start= 50.0, stop= 140.0, num= 81)

    totalProps = len(semiMajorVals) * len(eccVals) * len(incVals)
    currentProps = 0

    # Initializes array of average orbital values. 
    orbitAverages = np.empty((len(semiMajorVals), len(incVals), len(eccVals)))

    # Initializes current run directories.
    runDir = dataDir + f"run_num_{runCount}/"
    valuesDir = dataDir + f"run_num_{runCount}/orbit_averages/"
    plotsDir = dataDir + f"run_num_{runCount}/plots/"

    ### Propagation-related definitions.
    if propOrbits := True:
        # Increases runcount.
        runCount += 1
        # Initializes current run directories.
        runDir = dataDir + f"run_num_{runCount}/"
        valuesDir = dataDir + f"run_num_{runCount}/orbit_averages/"
        plotsDir = dataDir + f"run_num_{runCount}/plots/"
        # Updates runcount file.
        with open("runcount.txt", "w") as f:
            f.write('%d' % runCount)

        # Creates run directories.
        os.mkdir(runDir)
        os.mkdir(valuesDir)
        os.mkdir(plotsDir)

        # Saves solar panels and tumbling average powers file.
        tumblingArr = np.zeros(np.size(solarArray))
        tumblingArr[0] = tumblingPowers[0]
        firstLine = np.array(["Tumbling Power", "Solar Panels"])
        saveArr = np.vstack((firstLine, np.column_stack((tumblingArr, solarArray))))

        np.savetxt(valuesDir + 'sim_params.txt', saveArr, delimiter=',', fmt="%s")

        # Defines simulation start date and time (UTC). (YYYY-MM-DDTHH:MM:SS)
        pythonDate = datetime.fromisoformat(propStartISO)
        # Converts into tudat time format. 
        tudatDate = time_conversion.datetime_to_tudat(pythonDate)
        # Defines initial time as seconds since J2000. 
        propStartTime = tudatDate.epoch()

        # Build all argument combinations.
        all_args = [
            (ecc, inc, sma, propDurationTime, timeStep, propStartTime,
             tumblingPowers, propagationDir, scMass)
            for ecc in eccVals
            for inc in incVals
            for sma in semiMajorVals
        ]

        # Run propagations in parallel.
        max_workers = os.cpu_count() - 1  # uses all but 1 cpu cores
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(single_orbit, args): args for args in all_args}
            for future in as_completed(futures):
                ecc, inc, sma, orbitAvg = future.result()
                # Map back to indices.
                i = list(eccVals).index(ecc)
                j = list(incVals).index(inc)
                k = list(semiMajorVals).index(sma)

                orbitAverages[k, j, i] = orbitAvg
                currentProps += 1
                print(f"Completed propagation {currentProps} out of {totalProps}.")

        # Dump orbital averages to csv files.
        for i, eccentricity in enumerate(eccVals):
            filename = valuesDir + f"orbit_avg_eccentricity{eccentricity}.csv"
            np.savetxt(filename, orbitAverages[:, :, i], delimiter=",")

    ### Plots power average data.
    if plotAvgs := True:
        plot_average_heatmap(
            eccVals= eccVals,
            incVals= incVals,
            semiMajorVals= semiMajorVals,
            dataDir= dataDir,
            runCount= runCount,
            showUncompliant= False,
            powerReq= 2.52
        )

    ##### Battery charge operations #####
    dataDir = "data_battery/"
    # Runs battery charge orbit propagation.
    # TODO: Turn this into an isolated function
    if batteryChargeProp := False:

        ### Propagation parameters.
        # Initial keplerian elements. Formatted as array for easy saving later.
        # Semi-major, eccentricity, inclination, arg of Pe, Long of ascending node, true anomaly.
        stateStartKepArr = np.array([6860e3, 0.0, np.deg2rad(90), np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)])
        stateStartKep = np.array([
            stateStartKepArr[0],               # Semi-major axis [m]
            stateStartKepArr[1],               # eccentricity
            stateStartKepArr[2],               # Inclination [rads].
            stateStartKepArr[3],               # arg of periapsis [rads]
            stateStartKepArr[4],               # longitude of ascending node [rads]
            stateStartKepArr[5]                # true anomaly [rads]
        ])
        # Initial date and time for propagation.
        propStartISO = "2025-06-21T07:00:00.000000"
        # Defines total propagation time in hours.
        propDurationTime = 24.0
        # Defines constant time step in seconds.
        timeStep = 20.0
        # Converts into tudat time format.
        tudatDate = time_conversion.datetime_to_tudat(datetime.fromisoformat(propStartISO))
        # Defines initial time as seconds since J2000.
        propStartTime = tudatDate.epoch()

        # Reads run count.
        with open("runcount_battery.txt", "r") as f:
            runCount = int(f.read())

        # Increases runcount.
        runCount += 1

        # Updates runcount file.
        with open("runcount_battery.txt", "w") as f:
            f.write('%d' % runCount)

        # Run directory addresses.
        runDir              = dataDir + f"run_num_{runCount}/"
        propsDir            = dataDir + f"run_num_{runCount}/propagation/"
        outputsDir          = dataDir + f"run_num_{runCount}/outputs/"
        plotsDir            = dataDir + f"run_num_{runCount}/plots/"

        # Creates run directories.
        os.mkdir(runDir)
        os.mkdir(propsDir)
        os.mkdir(outputsDir)
        os.mkdir(plotsDir)

        # Creates environment bodies.
        bodies = create_bodies(
            sc_mass                     =scMass,
            initial_att                 =np.eye(3),
            rotation                    =False,
            starting_time               =propStartTime,
            time_step                   =timeStep
        )

        # TODO: Should not be used since we dont do attitude sims, but propagation breaks if its not here.
        bodies = create_rotational_settings(
            bodies                      =bodies,
            time_step                   =timeStep
        )

        # Propagates orbit.
        stateHistory, dependentHistory, stateArr, dependentArr = \
            propagate_orbit(
                propDurationTime        =propDurationTime,
                timeStep                =timeStep,
                stateStartKep           =stateStartKep,
                propStartTime           =propStartTime,
                bodies                  =bodies
            )

        # Saves dependent array as csv.
        np.savetxt(propsDir + 'dependent_vals.txt', dependentArr, delimiter=',', fmt="%s")
        # Saves propagation parameters.
        np.savetxt(propsDir + 'start_kepler_elems.txt', stateStartKepArr, delimiter=',', fmt="%s")

    # Runs battery charge simulation.
    if batteryChargeSim  := False:
        battery_sim(
            dataDir         = dataDir,
            battStart       = battStart,
            battMax         = battMax,
            solarArray      = solarArray,
            tumblingPowers  = tumblingPowers,
            runCount        = 1
        )

    ### Plots battery charge vs time.
    if batteryChargePlots := False:
        plot_battery_charge(
            dataDir         = dataDir,
            battMax         = battMax,
            runCount        = [1,2,3],
            plotCombined    = True
        )


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
