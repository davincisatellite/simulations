import numpy as np
import matplotlib.pyplot as plt
import time

import orbit 
import attitude as att


plt_dir = "./Plots/"


if __name__ == "__main__":
    ###########################################################################
    # ORBITAL STUFF ###########################################################
    ###########################################################################

    theta_vals = np.linspace(0, 720, 200)

    semi_major = 6700 #Semi major axis [km]
    ecc = 0.05 #Eccentricity
    inc = 98 #Inclination [º]
    arg = 0 #Argument of Pe [º]
    decl = 5 #Solar declination (Positive south of equator) [º]
    earth_rad = 6378 #Earth radius [km]
    righ_ascention = 20 #Right ascention node from sun-pointing [º]

    radii = orbit.posVectorPlanar(theta_vals, semi_major, ecc)

    eclipse = np.empty(np.shape(radii)[1])

    for i in range(np.shape(radii)[1]):
        eclipse[i] = orbit.eclypseCheck(radii[:,i], earth_rad)


    ###########################################################################
    # POWER GENERATION STUFF ##################################################
    ###########################################################################
    # Creates solar array distribution: [+X, -X, +Y, -Y, +Z, -Z] W
    # TODO: Better estimate for maximum power production for each cell.
    solar_cell = 1.08 # [Watt]
    solar_array = [4*solar_cell, 4*solar_cell, 4*solar_cell, 4*solar_cell,
                2*solar_cell, 2*solar_cell]


    # Special cases. 
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


    # New method for tumbling. Creates a bunch of quaternions, transforms them
    # into rotation matrices and takes these as a set of attitudes. 
    if new_method := True:
        # Generate random set of atttiudes. 
        num_attitudes = 2000
        random_atts = att.generate_random_attitudes(num_attitudes)
        set_atts = att.generate_set_attitudes(num_attitudes)

        # Initializes empty power vector.
        powers_rand = np.empty(0)
        powers_set = np.empty(0)

        for att_index in range(num_attitudes):
            # Calculate power produced in each attitude. 
            powers_rand = np.append(powers_rand, att.power_output(
                random_atts[att_index], solar_array))
            powers_set = np.append(powers_set, att.power_output(
                set_atts[att_index], solar_array))


        # Returns average powers. 
        print(f"Average power (random): {np.average(powers_rand)} W.")
        print(f"Average power (set): {np.average(powers_set)} W.")

        # Plots histogram of power values.  
        fig = plt.figure(figsize=(15,8), dpi=80)

        ax = fig.add_subplot(2, 1, 1)
        ax.hist(powers_rand, bins= 40)

        ax = fig.add_subplot(2, 1, 2)
        ax.hist(powers_set, bins= 40)

        plt.show()


    # TODO: Add some statistics in here to get a deviation and all that other
    # juicy stuff.

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






