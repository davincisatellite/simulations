import numpy as np
import matplotlib.pyplot as plt
import time

import tumbling_code.orbit as orbit
import tumbling_code.attitude as att


plt_dir = "./Plots/"

# Earth std. grav. parameter. [km^3/s^2]
earth_mu = 0.39860e6 

"""
Assumptions and units stuff.
- Distance in km.
- All other units in SI. NOTE: This means radians.
"""

# New method for tumbling. Creates a bunch of quaternions, transforms them
# into rotation matrices and takes these as a set of attitudes. 
def random_quaternion_tumbling(solarArray, numAttitudes = 2000,
                                randomAvg = True, randomSet = False):
    # Generate random set of atttiudes. 
    random_atts = att.generate_random_attitudes(numAttitudes)

    # Initializes empty power vector.
    powersRand = np.empty(0)

    for att_index in range(numAttitudes):
        # Calculate power produced in each attitude. 
        if randomAvg:
            powersRand = np.append(powersRand, att.power_output(
                random_atts[att_index], solarArray))

    powersRand = np.average(powersRand)

    return powersRand

if __name__ == "__main__":
    ###########################################################################
    # ORBITAL STUFF ###########################################################
    ###########################################################################

    # Kepler elements for orbit. 
    semi_major = 6700 #Semi major axis [km]
    ecc = 0.05 #Eccentricity
    inc = 90 * np.pi/180 #Inclination [º]
    pe_arg = 0 * np.pi/180  #Argument of Pe [º]
    sol_dec = 5 * np.pi/180  #Solar declination (Positive south of equator) [º]
    earth_rad = 6378 #Earth radius [km]
    righ_ascention = 90 * np.pi/180  #Right ascention node from sun-pointing [º]

    # Initializes constant time-step and upper time limit. 
    time_step = 10
    time_limit = 1.5 * 60**2  
    time_current = 0

    print(f"Period: {(2*np.pi)*np.sqrt((semi_major**3)/earth_mu) / 60**2}")

    # Initializes planar position state vector. 
    # NOTE: Assumes initial true anomaly = 0º. 
    start_position= orbit.posVector3D(
            pos_vector_planar= orbit.posVectorPlanar(0.0, semi_major, ecc), 
            pe_arg= pe_arg, 
            inclination= inc,
            right_ascention= righ_ascention,
            sol_dec= sol_dec
        )
    state_arr = np.hstack([time_current, start_position])
    eclipse_arr = orbit.eclipseCheck(pos_3d= start_position,
                                     r_planet= earth_rad)
    mean_anom = 0

    # Calculates constant mean motion. 
    # NOTE: Assumes constant mean motion. 
    mean_motion = orbit.meanMotion(semi_major= semi_major)

    # Simulates kepler orbit throughout time. 
    while time_current < time_limit:
        
        # Updates time. 
        time_current += time_step

        # Calculates all relevant orbital parameters. 
        mean_anom = orbit.meanAnomaly(m_0= mean_anom, 
                                      mean_mot= mean_motion, 
                                      delta_t= time_step)
        
        ecc_anom = orbit.eccAnomaly(m_t= mean_anom, ecc= ecc)

        true_anom = orbit.trueAnomaly(ecc= ecc, e_t= ecc_anom)

        position_planar = orbit.posVectorPlanar(theta= true_anom,
                                         semi_maj= semi_major, 
                                         ecc= ecc)
        
        # Updates 3D position in sun-pointing coord system. 
        position = orbit.posVector3D(
            pos_vector_planar= position_planar, 
            pe_arg= pe_arg, 
            inclination= inc,
            right_ascention= righ_ascention,
            sol_dec= sol_dec
        )

        # Checks if position is in eclipse and adds to array. 
        eclipse_arr = np.append(
            eclipse_arr, orbit.eclipseCheck(pos_3d= position, 
                                            r_planet= earth_rad)
        )
        
        # Updates states array. 
        state_arr = np.vstack([state_arr, np.hstack([time_current, position])])

    ###########################################################################
    # POWER GENERATION STUFF ##################################################
    ###########################################################################
    # Creates solar array distribution: [+X, -X, +Y, -Y, +Z, -Z] W
    # TODO: Better estimate for maximum power production for each cell.
    solar_cell = 1.08 # [Watt]
    solar_array = [4*solar_cell, 4*solar_cell, 4*solar_cell, 4*solar_cell,
                2*solar_cell, 2*solar_cell]

    # Special cases of solar panel distribution. 
    # Missing one string on the +Y side.
    if missing_halfY := False:
        solar_array = solar_array * np.array([1, 1, 0.5, 1, 1, 1])
    # Missing the full set on the -Z side. 
    if missing_fullZ := False:   
        solar_array = solar_array * np.array([1, 1, 1, 1, 1, 0])

    # Old method for tumbling. Creates a bunch of rotation quaternions and
    # applies them in succesion. Generates a spread of attitudes. 
    if old_method := False:
        # Generates a set of evenly spaced out vectors on a unit sphere.
        rot_vector_array = att.fibonacci_sphere(samples= 400)

        # Initializes the attitude of the satellite with its vehicle X-axis pointing 
        # towards the sun.
        initial_att_1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
        initial_att_2 = np.array([[0,1,0],[1,0,0],[0,0,1]])
        initial_att_3 = np.array([[0,0,1],[0,1,0],[1,0,0]])

        initial_atts = [initial_att_1, initial_att_2, initial_att_3]

        # Initializes empty vector to store power values.
        power_library = []

        # Generates random array of rotation angles. 
        angle_array = np.random.uniform(low=0, high=360, size=100) * (np.pi/180)
        att_indx = 0

        for initial_att in initial_atts:

            new_attitude = initial_att
            rot_index = 1

            # Loops through available rotation vectors.
            for rot_vector in rot_vector_array:
                i = 0
                # Initializes power array for each rotation vector.
                power_array = np.empty(np.size(angle_array))
                # Rotates initial attitude throughout 360º on the given rotation vector.
                for angle in angle_array:

                    # Gets quaternion for rotation angle.
                    q = att.get_quaternion(rot_vector, angle)

                    # Rotates initial attitude.
                    p_x2 = att.quaternion_rotation(new_attitude[0,:], q)
                    p_y2 = att.quaternion_rotation(new_attitude[1,:], q)
                    p_z2 = att.quaternion_rotation(new_attitude[2,:], q)

                    # Updates attitude. 
                    new_attitude = np.array([p_x2, p_y2, p_z2])
                    
                    # Calculates power at the given attitude.
                    # Proportional to cosine of incidence angle. Only parameter.
                    power = att.power_output(new_attitude, solar_array)

                    # Stores power in array.
                    power_array[i] = power
                    i += 1

                rot_index += 1

                # Adds power to overall library.
                if np.size(power_library) == 0:
                    power_library = power_array
                else:
                    power_library = np.vstack([power_library, power_array])


            # Calculates weighted average
            tumbling_avg = np.average(power_library, axis= 1)

            fig = plt.figure(figsize=(15,8), dpi=80)

            ax = fig.add_subplot(1, 1, 1)

            ax.hist(tumbling_avg, bins= 20)

            plt.savefig(f"power_hist_{att_indx}.png")

            print(f"Average Tumbling Power Generation: {np.average(tumbling_avg)} W")

            att_indx += 1

        plt.show()    

    # Just some plotting for fun. 
    # Feel free to disable if you don't like spaghetti. :)
    if tumbling_spaghetti := False:
        fig = plt.figure(figsize=(15,8), dpi=80)

        ax = fig.add_subplot(1, 1, 1)

        for i in range(np.shape(rot_vector_array)[0]):
            rot_vector = rot_vector_array[i]
            i_label = f"Rotation Vector {rot_vector}"
            ax.scatter(angle_array*180/np.pi, power_library[i,:], label= i_label)

        ax.grid()

        #plt.legend()
        plt.show()






