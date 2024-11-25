import numpy as np

from algebra import rotationX, rotationY, rotationZ

def orbitRad(theta, semi_maj, ecc):
    """
    Returns the magnitude of the orbital radius from center of planet at a specific true anomaly.
    - theta: True anomaly [º]
    - semi_maj: Semi-major axis [km]
    - ecc: Eccentricity [-]
    """
    return (semi_maj * (1 - ecc**2))/(1 + ecc*np.cos(theta * np.pi/180))

def posVectorPlanar(theta, semi_maj, ecc):
    """
    Returns the planar position vector in a 3-element array.
    - theta: True anomaly [º]
    - semi_maj: Semi-major axis [km]
    - ecc: Eccentricity [-]
    """
    pos_mag = orbitRad(theta=theta, semi_maj=semi_maj, ecc=ecc)

    pos_vector = np.array(
        [pos_mag * np.cos(theta *  np.pi/180), pos_mag * np.sin(theta * np.pi/180), 0]
    )

    return pos_vector

def posVector3D(pos_vector_planar, pe_arg, inclination, right_ascention, sol_dec):
    """
    Returns the position vector in a sun-pointing, earth-fixed, coordinate frame.
    - pos_vector_planar: 3-elem array for planar orbital position. [km]
    - pe_arg: argument of periapsis. [º]
    - inclination: orbital inclination (versus equator) [º]
    - right_ascention: Right ascention of ascending node, relative to sun pointing vector. [º]
    - sol_dec: solar declination angle versus equator. Positive in Northern latitudes. [º]
    """

    pos_vector_planar = pos_vector_planar[..., None]

    R_z_arg = rotationZ(-pe_arg)
    R_x_inc = rotationX(-inclination)
    R_z_ras = rotationZ(-right_ascention)
    R_y_sdec = rotationY(sol_dec)

    # If you can find a better way to do matrix multiplication be my guest.
    R_t = np.matmul(np.matmul(R_y_sdec, R_z_ras), np.matmul(R_x_inc, R_z_arg))

    return np.matmul(R_t, pos_vector_planar).T[0]

def eclypseCheck(pos_3d, r_planet):
    """
    Returns a true-false depending on whether the position is eclypsed by the given planet radius.
    - pos_3d: 3-element array of position on sun-pointing planet-fixed coordinate frame. [km]
    - r_planet: Planetary radius [km]
    """
    y_z_mag = np.sqrt(pos_3d[1]**2 + pos_3d[2]**2)

    if pos_3d[0] < 0 and y_z_mag < r_planet:
        return True
    else:
        return False


""" 
Verification code. Remove at some point.

ver_a = 50000
ver_ecc = 0.05
ver_theta = 90
ver_inc = 0
ver_decl = 0
ver_ras = 90
ver_arg = 0
ver_radius = 20000

pos_vector_2d = posVectorPlanar(theta= ver_theta, semi_maj= ver_a, ecc= ver_ecc)
print(pos_vector_2d)

pos_vector_3d = posVector3D(pos_vector_planar= pos_vector_2d, pe_arg= ver_arg, inclination= ver_inc, right_ascention= ver_ras, sol_dec= ver_decl)
print(pos_vector_3d)

print(f"The spacecraft is eclypsed:{eclypseCheck(pos_vector_3d, ver_radius)}") """