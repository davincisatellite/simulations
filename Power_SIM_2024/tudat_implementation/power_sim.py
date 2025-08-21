import numpy as np

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
