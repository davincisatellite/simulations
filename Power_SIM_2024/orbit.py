import numpy as np

from algebra import rotation_x, rotation_y, rotation_z

def orbit_rad(theta, semi_maj, ecc):
    """
    Returns the magnitude of the orbital radius from center of planet at a specific true anomaly.
    - theta: True anomaly [º]
    - semi_maj: Semi-major axis [km]
    - ecc: Eccentricity [-]
    """
    return (semi_maj * (1 - ecc**2))/(1 + ecc*np.cos(theta * np.pi/180))

def pos_vector_planar(theta, semi_maj, ecc):
    """
    Returns the planar position vector in a 3-element array.
    - theta: True anomaly [º]
    - semi_maj: Semi-major axis [km]
    - ecc: Eccentricity [-]
    """
    pos_mag = orbit_rad(theta=theta, semi_maj=semi_maj, ecc=ecc)

    pos_vector = np.array(
        [pos_mag * np.cos(theta *  np.pi/180), pos_mag * np.sin(theta * np.pi/180), 0]
    )

    return pos_vector

def pos_vector_3D(pos_vector_planar, pe_arg, inclination, right_ascention, sol_dec):
    """
    Returns the position vector in a sun-pointing, earth-fixed, coordinate frame.
    - pos_vector_planar: 3-elem array for planar orbital position. [km]
    - pe_arg: argument of periapsis. [º]
    - inclination: orbital inclination (versus equator) [º]
    - right_ascention: Right ascention of ascending node, relative to sun pointing vector. [º]
    - sol_dec: solar declination angle versus equator. Positive in Northern latitudes. [º]
    """

    pos_vector_planar = pos_vector_planar[..., None]

    R_z_arg = rotation_z(pe_arg)
    R_x_inc = rotation_x(inclination)
    R_z_ras = rotation_z(right_ascention)
    R_y_sdec = rotation_y(sol_dec)

    # If you can find a better way to do matrix multiplication be my guest.
    R_t = np.matmul(np.matmul(R_z_arg, R_x_inc), np.matmul(R_z_ras, R_y_sdec))

    return np.matmul(R_t, pos_vector_planar).T


ver_a = 50000
ver_ecc = 0.05
ver_theta = 90
ver_inc = 45
ver_decl = 0
ver_ras = 90
ver_arg = 0

pos_vector_2d = pos_vector_planar(theta= ver_theta, semi_maj= ver_a, ecc= ver_ecc)
print(pos_vector_2d)

pos_vector_3d = pos_vector_3D(pos_vector_planar= pos_vector_2d, pe_arg= ver_arg, inclination= ver_inc, right_ascention= ver_ras, sol_dec= ver_decl)
print(pos_vector_3d)