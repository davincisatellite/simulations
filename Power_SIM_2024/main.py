import numpy as np
import matplotlib.pyplot as plt
import time

import orbit 
import attitude as att

theta_vals = np.linspace(0, 720, 200)

###########################################################################
# ORBITAL STUFF ###########################################################
###########################################################################

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

# Generates a set of evenly spaced out vectors on a unit sphere.
rot_vector_array = att.fibonacci_sphere(samples= 400)

# Verification array. 
# rot_vector_array = [[1,0,0],[0,1,0],[0,0,1]]

# Creates solar array distribution: [+X, -X, +Y, -Y, +Z, -Z] W
solar_cell = 1.08 # [Watt]
solar_array = [4*solar_cell, 4*solar_cell, 4*solar_cell, 4*solar_cell,
               2*solar_cell, 2*solar_cell]

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

if old_method := False:
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


if new_method := True:
    # Generate random set of atttiudes. 
    num_attitudes = 2000
    random_atts = att.generate_random_attitudes(num_attitudes)

    powers = np.empty(0)

    for attittude in random_atts:
        # Calculate power produced in each attitude. 
        powers = np.append(powers, att.power_output(attittude, solar_array))

    # Returns average power. 
    print(f"Average power: {np.average(powers)} W.")


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






