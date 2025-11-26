import numpy as np


from tumbling_code.main import random_quaternion_tumbling


def power_output(solar_pos_body_frame, solar_arr):
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
        inertial_to_body_rot_frame = dependentArr[:,5:14]
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
            solar_arr= solarArray)
        
    # NOTE: Sections that use the dependent array are very sensitive to 
    # adding or removing dependent variables to the propagation.
    # Name-based dictionaries here would probably help a lot.  

    # Uses shadow array to get final power production through orbit. 
    powerSolar = powerSolar * dependentArr[:,1]

    # Imports true anomaly values. 
    trueAnomaly = dependentArr[:,-1] * 180/np.pi

    # Finds closest index to 360º (One orbit) 
    indx = (np.abs(trueAnomaly - 360)).argmin()

    # Calculates average of power production through orbit. 
    orbitAvg = np.average(powerSolar[:indx])

    return orbitAvg
