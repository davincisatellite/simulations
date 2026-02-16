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

import numpy as np

from tudat_setup import * 
from assist_functions import *
from power_sim import *
from modes_conditions import *

from datetime import datetime

if __name__ == "__main__":

    dataDir = f"data/"

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
    eccVals = np.linspace(start= 0.00, stop= 0.015, num= 2)
    incVals = np.linspace(start= 60.0, stop= 120.0, num= 21)

    totalProps = len(semiMajorVals) * len(eccVals) * len(incVals)
    currentProps = 0

    # Initializes array of average orbital values. 
    orbitAverages = np.empty((len(semiMajorVals), len(incVals), len(eccVals)))


    # Initializes current run directories.
    runDir = dataDir + f"run_num_{runCount}/"
    valuesDir = dataDir + f"run_num_{runCount}/orbit_averages/"
    plotsDir = dataDir + f"run_num_{runCount}/plots/"

    ### Propagation-related definitions.
    if propOrbits := False:
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

        # Creates environment bodies. 
        bodies = create_bodies(
            sc_mass= scMass,
            initial_att= np.eye(3),
            rotation= True,
            starting_time= propStartTime,
            time_step= timeStep
        )

        # TODO: Should be skipped for now, this defines attitude behavior if needed.
        bodies = create_rotational_settings(
            bodies= bodies,
            time_step= timeStep
        )


        for i, eccentricity in enumerate(eccVals):
            for j,inclination in enumerate(incVals):
                for k,semiMajorAxis in enumerate(semiMajorVals):
                    # Progress indicator. 
                    currentProps += 1
                    print(f"Running propagation {currentProps} out of {totalProps}.")

                    # Defines initial keplerian orbital elements. 
                    stateStartKep = np.array([
                        semiMajorAxis,          # Semi-major axis [m]
                        eccentricity,           # eccentricity 
                        inclination,            # Inclination [rads]. Degrees in ().
                        np.deg2rad(0),          # arg of periapsis [rads]
                        np.deg2rad(0),          # longitude of ascending node [rads]
                        np.deg2rad(0),          # true anomaly [rads]
                    ])

                    # Propagates orbit. 
                    stateHistory, dependentHistory, stateArr, dependentArr = \
                        propagate_orbit(
                            propDurationTime= propDurationTime,
                            timeStep= timeStep,
                            stateStartKep= stateStartKep,
                            propStartTime= propStartTime,
                            bodies= bodies
                        )
                    
                    # Returns orbit average. 
                    orbitAvg = orbit_average(
                        stateArr= stateArr,
                        dependentArr= dependentArr, 
                        tumblingPowers= tumblingPowers,
                        tumblingCheck= True,
                    )

                    # Saves orbit average. 
                    orbitAverages[k,j,i] = orbitAvg

            # Dumps orbital averages to csv file.
            filename = valuesDir + f"orbit_avg_eccentricity{eccentricity}.csv"
            np.savetxt(filename, orbitAverages[:,:,i], delimiter= ",")

    ### Plots power average data. 
    # TODO: Put this in its own separate function. 
    if plotAvgs := False:
        plot_average_heatmap(
            eccVals= eccVals,
            incVals= incVals,
            semiMajorVals= semiMajorVals,
            dataDir= dataDir,
            runCount= 5,
            showUncompliant= False
        )

    ##### Battery charge operations #####
    dataDir = "data_battery/"
    # Runs battery charge orbit propagation.
    if batteryChargeProp := False:

        ### Propagation parameters.
        # Initial keplerian elements. Formatted as array for easy saving later.
        # Semi-major, eccentricity, inclination, arg of Pe, Long of ascending node, true anomaly.
        stateStartKepArr = np.array([6720e3, 0.0, np.deg2rad(90), np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)])
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
    if batteryChargeSim  := True:
        # Which propagation run are you using.
        runCount            = 1
        # Run directory addresses.
        runDir          = dataDir + f"run_num_{runCount}/"
        propsDir        = runDir + "propagation/"
        outputsDir      = runDir + "outputs/"
        plotsDir        = runDir + "plots/"

        # Extracts dependent values array.
        dependentArr    = np.loadtxt(propsDir + 'dependent_vals.txt', delimiter=',')
        times           = dependentArr[:, 0]
        sunlight        = dependentArr[:, 1]
        timeStep        = times[1] - times[0]

        # Empty array of battery charge.
        battArr         = np.empty(np.size(times))

        # Initializes mode as idle.
        activeMode          = modeIdle
        # Initializes battery charge.
        battArr[0]          = battStart

        # Saves solar panels and tumbling average powers to file.
        tumblingArr = np.zeros(np.size(solarArray))
        tumblingArr[0] = tumblingPowers[0]
        firstLine = np.array(["Tumbling Power", "Solar Panels"])
        saveArr = np.vstack((firstLine, np.column_stack((tumblingArr, solarArray))))

        np.savetxt(outputsDir + 'solar_panel_vals.txt', saveArr, delimiter=',', fmt="%s")

        for idx, time in enumerate(times[:-1]):

            # Sunlit from propagation shadow function.
            sunlitCurrent               = sunlight[idx]

            # Mode check.
            if activeMode.name == "idle":
                # Checks for comms condition.
                if modeComms.check_active(
                    batteryCharge      = battArr[idx],
                    sunlit                  = sunlitCurrent,
                    currentTime             = time
                ):  activeMode = modeComms

                # Checks for payload condition.
                elif modePayload.check_active(
                    batteryCharge      = battArr[idx],
                    sunlit                  = sunlitCurrent,
                    currentTime             = time
                ):  activeMode = modePayload

                else: activeMode = modeIdle
            else:
                if activeMode.check_active(
                    batteryCharge      = battArr[idx],
                    sunlit                  = sunlitCurrent,
                    currentTime             = time
                ):
                    activeMode = activeMode
                    """print(f"Time: {(time - times[0])/60}")
                    print(f"Active mode activation time: {(activeMode.timeActivated - times[0])/60}")"""
                else:
                    activeMode = modeIdle

            # Power production.
            powerNet            = tumblingPowers[0] - activeMode.powerActive
            
            # Update battery charge.
            battArr[idx+1]      = battArr[idx] + powerNet*(timeStep/60**2)          # W*h

            # Checks if zero battery or max battery is reached. 
            if battArr[idx+1] < 0.0:
                print(f"Battery at zero charge!!")
                battArr[idx+1] = 0.0
            elif battArr[idx+1] > battMax:
                battArr[idx+1] = battMax 

        # Stacks times and battery charge arrays.
        outputArr = np.column_stack((times, battArr.T))

        # Saves battery charge vs time array.
        np.savetxt(outputsDir + 'battery_charge.txt', outputArr, delimiter=',', fmt="%s")


    ### Plots battery charge vs time.
    if batteryChargePlots := True:

        # Which propagation run are you using.
        runCount = 1

        # Run directory addresses.
        runDir = dataDir + f"run_num_{runCount}/"
        propsDir = runDir + "propagation/"
        outputsDir = runDir + "outputs/"
        plotsDir = runDir + "plots/"

        outputArr           = np.loadtxt(outputsDir + 'battery_charge.txt', delimiter=',')
        dependentArr        = np.loadtxt(propsDir + 'dependent_vals.txt', delimiter=',')

        times = (outputArr[:,0] - outputArr[0,0]) / 60          # Min

        plt.plot(times, dependentArr[:, 1] * battMax, "--")
        plt.plot(times, outputArr[:,1])


        plt.grid()

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

        #ax = plt.figure().add_subplot(projection='3d')
        ax1 = plt.figure().add_subplot()

        times = state_array[:,0] / 60**2

        #ax.plot(state_array[:,1], state_array[:,2], state_array[:,3])

        ax1.plot(times, dependent_array[:,1])

        #ax.set_zlim(-5e6, 5e6)
        #ax.set_xlim(-5e6, 5e6)
        #ax.set_ylim(-5e6, 5e6)

        plt.show()

        #######  ---  #######







    
