import numpy as np




def point_mass_earth_accel(r):
    mu = 3.986004418e14 #m3s-2
    a = -mu / np.linalg.norm(r)**3 * r
    return a

def find_long_lat_form_state(t, r):
    
    r_mag = np.linalg.norm(r)
    
    r_xy = np.linalg.norm(r[0:2])
    
    lam = np.arctan2(r[1]/r_xy, r[0]/r_xy)
    
    lat = np.arcsin(r[2]/r_mag)
    
    long = lam - 7.292115e-5 * t
    
    return long, lat
    