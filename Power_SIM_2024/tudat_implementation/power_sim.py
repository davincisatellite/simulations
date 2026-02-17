import numpy as np
import numpy.typing as npt

from modes_conditions import *
from tumbling_code.attitude import generate_random_attitudes


def power_output_vector(solar_pos_body_frame, solar_arr):
    # Extract coordinates from solar position in body frame. 
    x = solar_pos_body_frame[:,0]
    y = solar_pos_body_frame[:,1]
    z = solar_pos_body_frame[:,2]
    zeros = np.zeros(np.shape(x))

    # Calculates power production in each of satellite's axes.
    Px = solar_arr[0]*np.max([x, zeros], axis=0) + \
        solar_arr[1]*np.abs(np.min([x, zeros], axis=0))
    Py = solar_arr[2]*np.max([y, zeros], axis=0) + \
        solar_arr[3]*np.abs(np.min([y, zeros], axis=0))
    Pz = solar_arr[4]*np.max([z, zeros], axis=0) + \
        solar_arr[5]*np.abs(np.min([z, zeros], axis=0))

    return Px+Py+Pz


def power_output(orient, solar_arr):
    """Retrieves power production depending on satellite orientation. Considers
    an external reference frame where X-axis is sun-pointing. Assumes direct
    correlation between cosine of incidence and power production.
    Inputs:
    - orient: 3x3 array of satellite orientation within the out-of-vehicle
    frame.
    - solar_arr: 6-element array of the solar panel power production.
    """
    # Extracts X-component (Sun-facing) of each satellite attitude axis.
    Dx_x = orient[0,0]
    Dy_x = orient[1,0]
    Dz_x = orient[2,0]

    # Calculates power production in each of satellite's axes.
    Px = solar_arr[0]*max([Dx_x, 0]) + solar_arr[1]*np.abs(min([Dx_x, 0]))
    Py = solar_arr[2]*max([Dy_x, 0]) + solar_arr[3]*np.abs(min([Dy_x, 0]))
    Pz = solar_arr[4]*max([Dz_x, 0]) + solar_arr[5]*np.abs(min([Dz_x, 0]))

    return Px+Py+Pz


def tumbling_powers(solarArray, numVals = 1):
    """Creates an array of tumbling power values via the 
    random_quaternion_tumbling method. 
    """
    if numVals == 0:
        print("ERROR: No tumbling values requested.")
        tumblingPowers = 0
    else:
        tumblingPowers = np.empty(numVals)

        for i in range(numVals):
            tumblingPowers[i] = np.average(
                random_quaternion_tumbling(solarArray= solarArray))

    return tumblingPowers


# New method for tumbling. Creates a bunch of quaternions, transforms them
# into rotation matrices and takes these as a set of attitudes.
def random_quaternion_tumbling(solarArray, numAttitudes = 2000,
                                randomAvg = True, randomSet = False):
    # Generate random set of attitudes.
    random_atts = generate_random_attitudes(numAttitudes)
    # Initializes empty power vector.
    powersRand = np.empty(0)

    for att_index in range(numAttitudes):
        # Calculate power produced in each attitude.
        if randomAvg:
            powersRand = np.append(powersRand, power_output(
                random_atts[att_index], solarArray))

    powersRand = np.average(powersRand)

    return powersRand


def orbit_average(
        stateArr: np.array,
        dependentArr: np.array,
        tumblingPowers: np.array, 
        tumblingCheck: bool= True,
        solarArray: np.array= np.zeros(6)
):
    if tumblingCheck:
        # TODO: Implement loop that tests power production through all 
        # average powers in the tumbling array. 
        powerSolar = tumblingPowers
    else:
        # TODO: This whole implementation is now wrong. needs to be fixed.
        '''inertial_to_body_rot_frame = dependentArr[:,5:14]
        solar_pos_relative_to_sat = dependentArr[:,2:5]
        # Defines times array (In hours). 
        times = (dependentArr[:,0] - dependentArr[0,0]) / 60**2 

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
            solar_arr= solarArray)'''
        
    # NOTE: Sections that use the dependent array are very sensitive to 
    # adding or removing dependent variables to the propagation.
    # Name-based dictionaries here would probably help a lot.  

    # Uses shadow array to get final power production through orbit. 
    powerSolar = powerSolar * dependentArr[:,1]

    # Imports true anomaly values. 
    trueAnomaly = dependentArr[:,-1] * 180/np.pi

    # Finds indices close to 360º within a margin (1.5º)
    indx = np.argwhere(np.abs(trueAnomaly - 360) < 5)

    # Calculates average of power production through orbit considering the 
    # last index in the array. Should ensure same number of orbits considered
    # for all averages. 
    orbitAvg = np.average(powerSolar[:indx[-1,0]])

    return orbitAvg

def battery_sim(
        dataDir: str,
        battStart: float,
        battMax: float,
        solarArray: list,
        tumblingPowers: npt.NDArray,
        runCount: int
):
    """
    Args:
        dataDir: String. Directory of battery sim output data.
        battStart: Float. Battery starting charge. [W*h]
        battMax: Float. Battery maximum charge. [W*h]
        solarArray: Numpy array. Solar array power generation. [W]
        tumblingPowers: Numpy array. Averaged out tumbling power generation. [W]
        runCount: Int. Run number. Should relate to individual orbit propagations.

    Returns:
        None, saves results directly to files.
    """
    # Averages out tumbling powers if non-singular array is input.
    if tumblingPowers.size > 1:
        tumblingPower = np.average(tumblingPowers)
    else:
        tumblingPower = tumblingPowers[0]

    # Run directory addresses.
    runDir = dataDir + f"run_num_{runCount}/"
    propsDir = runDir + "propagation/"
    outputsDir = runDir + "outputs/"

    # Extracts dependent values array.
    dependentArr = np.loadtxt(propsDir + 'dependent_vals.txt', delimiter=',')
    times = dependentArr[:, 0]
    sunlight = dependentArr[:, 1]
    timeStep = times[1] - times[0]

    # Empty array of battery charge.
    battArr = np.empty(np.size(times))
    # Initializes array of active mode IDs.
    iDs     = np.empty(np.size(times))

    # Initializes mode as idle.
    activeMode = modeIdle
    # Initializes battery charge.
    battArr[0] = battStart


    # Saves solar panels and tumbling average powers to file.
    tumblingArr = np.zeros(np.size(solarArray))
    tumblingArr[0] = tumblingPower
    firstLine = np.array(["Tumbling Power", "Solar Panels"])
    saveArr = np.vstack((firstLine, np.column_stack((tumblingArr, solarArray))))

    np.savetxt(outputsDir + 'solar_panel_vals.txt', saveArr, delimiter=',', fmt="%s")

    for idx, time in enumerate(times[:-1]):

        # Sunlit from propagation shadow function.
        sunlitCurrent = sunlight[idx]

        # Mode check.
        if activeMode.name == "idle":
            # Checks for comms condition.
            if modeComms.check_active(
                    batteryCharge=battArr[idx],
                    sunlit=sunlitCurrent,
                    currentTime=time
            ):
                activeMode = modeComms

            # Checks for payload condition.
            elif modePayload.check_active(
                    batteryCharge=battArr[idx],
                    sunlit=sunlitCurrent,
                    currentTime=time
            ):
                activeMode = modePayload

            else:
                activeMode = modeIdle
        else:
            if activeMode.check_active(
                    batteryCharge=battArr[idx],
                    sunlit=sunlitCurrent,
                    currentTime=time
            ):
                activeMode = activeMode
                """print(f"Time: {(time - times[0])/60}")
                print(f"Active mode activation time: {(activeMode.timeActivated - times[0])/60}")"""
            else:
                activeMode = modeIdle

        # Saves active mode ID.
        iDs[idx] = activeMode.iD

        # Power production.
        powerNet = tumblingPower*sunlitCurrent - activeMode.powerActive

        # Update battery charge.
        battArr[idx + 1] = battArr[idx] + powerNet * (timeStep / 60 ** 2)  # W*h

        # Checks if zero battery or max battery is reached.
        if battArr[idx + 1] < 0.0:
            print(f"Battery at zero charge!!")
            battArr[idx + 1] = 0.0
        elif battArr[idx + 1] > battMax:
            battArr[idx + 1] = battMax

            # Stacks times and battery charge arrays.
    outputArr = np.column_stack((times, battArr.T, iDs.T))

    # Saves battery charge vs time array.
    np.savetxt(outputsDir + 'battery_charge.txt', outputArr, delimiter=',', fmt="%s")

    return None


