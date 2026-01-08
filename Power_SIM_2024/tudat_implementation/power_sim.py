import numpy as np


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

