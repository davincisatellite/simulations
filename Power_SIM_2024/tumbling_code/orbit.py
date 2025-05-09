import numpy as np

from algebra import rotationX, rotationY, rotationZ

def orbitRad(theta, semi_maj, ecc):
    """
    Returns the magnitude of the orbital radius from center of planet at a specific true anomaly.
    - theta: True anomaly [rad]
    - semi_maj: Semi-major axis [km]
    - ecc: Eccentricity [-]
    """
    return (semi_maj * (1 - ecc**2))/(1 + ecc*np.cos(theta))


def posVectorPlanar(theta, semi_maj, ecc):
    """
    Returns the planar position vector in a 3-element array.
    - theta: np.array, true anomaly [rad]
    - semi_maj: Semi-major axis [km]
    - ecc: Eccentricity [-]
    """
    pos_mag = orbitRad(theta=theta, semi_maj=semi_maj, ecc=ecc)

    pos_vector = np.array(
        [pos_mag * np.cos(theta), 
         pos_mag * np.sin(theta), 
         0.0]
    )

    return pos_vector


def posVector3D(pos_vector_planar, pe_arg, inclination, right_ascention, sol_dec):
    """
    Returns the position vector in a sun-pointing, earth-fixed, coordinate frame.
    - pos_vector_planar: 3-elem array for planar orbital position. [km]
    - pe_arg: argument of periapsis. [rad]
    - inclination: orbital inclination (versus equator) [rad]
    - right_ascention: Right ascention of ascending node, relative to sun pointing vector. [rad]
    - sol_dec: solar declination angle versus equator. Positive in Northern latitudes. [rad]
    """

    pos_vector_planar = pos_vector_planar[..., None]

    R_z_arg = rotationZ(-pe_arg)
    R_x_inc = rotationX(-inclination)
    R_z_ras = rotationZ(-right_ascention)
    R_y_sdec = rotationY(sol_dec)

    # If you can find a better way to do matrix multiplication be my guest.
    R_t = np.matmul(np.matmul(R_y_sdec, R_z_ras), np.matmul(R_x_inc, R_z_arg))

    return np.matmul(R_t, pos_vector_planar).T[0]


def eclipseCheck(pos_3d, r_planet):
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


def meanMotion(semi_major, mu= 0.39860e6):
    """Calculates mean-motion.
    Inputs:
    - semi-major: Semi major axis. [km]
    - mu: standard gravitational parameter in [km^3/s^2]. Defaults to Earth.
    """
    return np.sqrt(mu/semi_major**3)


def meanAnomaly(m_0, mean_mot, delta_t):
    """Calculates mean anomaly at time t.
    Inputs:
    - m_0: Previous value of mean anomaly. [rad]
    - mean_mot: Mean motion value. [rad/s]
    - delta_t: Time step between m_0 and current mean anomaly. [s]
    Returns:
    - mean anomaly after time step. [rad]"""
    if delta_t == 0:
        return m_0
    else:
        return m_0 + mean_mot*delta_t
    

def eccAnomaly(m_t, ecc, converg_limit= 10**(-12)):
    """Calculates eccentric anomaly via initial guess and convergence limit.
    Inputs:
    - m_t: Mean anomaly at given time. [rad]
    - ecc: Eccentricity. 
    - coverg_limit: Convergence limit between successive iterations. Defaults \
    to 10^-8.
    """
    # Initializes guess and difference.
    e_prev, diff = np.pi, np.pi

    # Iterates eccentric anomaly while difference between guesses is above 
    # limit. 
    while diff > converg_limit:
        e_next = m_t + ecc* np.sin(e_prev)

        diff = abs(e_next - e_prev)

        e_prev = e_next
    
    return e_next


def trueAnomaly(ecc, e_t):
    """Calculates true anomaly with given eccentricity and eccentric anomaly.
    Inputs:
    - ecc: Eccentricity. 
    - e_t: Eccentric anomaly. [rad]"""
    # NOTE: Not using arctan2 here might cause issues in the future. 
    return 2*np.arctan(np.sqrt((1+ecc) / (1-ecc))*np.tan(e_t / 2))


