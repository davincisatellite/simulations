import numpy as np

def power_output(orient, solar_arr):
    """Retrieves power production depending on satellite orientation. Considers
    an external reference frame where X-axis is sun-pointing. Assumes direct
    correlation between cosine of incidence and power production.

    ------
    Parameters:
    - orient: 3x3 array of satellite orientation within the out-of-vehicle
    frame.
    - solar_arr: 6-element array of the solar panel power production.
    
    ------
    Return:
    - Total power production in satellite. 

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