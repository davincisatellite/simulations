import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from nrlmsise00 import msise_flat
import datetime
import initializer as init
import sys
import multiprocessing as mp

# constants

k = 1.380649e-23  # J/K
R = 8.31446261815324  # J/(K⋅mol)
N_A = 6.02214076e+23  # 1/mol

species_names = ["He", "O", "N2", "O2", "Ar", "H", "N", "O+"]
species_M = np.array([4.00260, 15.999, 2*14.007, 2*15.999,
                     39.9, 1.008, 14.007, 15.999])  # g/mol


def make_log(results, len_data):
    def logger(evaluation):
        results.append(0)
        print(
            f"{len(results)} / {len_data} completed, {len_data - len(results)} remaining")
    return logger


def find_nrlmsise00_magnetic_and_solar_properties(
    date: datetime.date,
    nrlmsis00_dates=init.nrlmsis00_dates,
    nrlmsis00_Aps=init.nrlmsis00_Aps,
    nrlmsis00_f107_obs=init.nrlmsis00_f107_obs,
    nrlmsis00_f107_obs_ctr81d=init.nrlmsis00_f107_obs_81d,
    remove_past_data=True
):
    """will find the Ap, F107 and F107 81 day avearge values for a specific date. The inputs to this function
    are created by the initializer file by default and dont really need to be touched. Only requried input is the date
    at which the magnetic and solar properties are to be found. Function can reduce the size of the nrlmsis00 data
    arrays loaded by the initializer file to decrease computation time. This will only allow to evaluate dates in 
    ascentding order. 

    Args:
        date (datetime.date): date at wich the magnetic and solar properties are to be found
        nrlmsis00_dates (numpy.ndarray, optional): array of datetime.date. Defaults to init.nrlmsis00_dates.
        nrlmsis00_Aps (numpy.ndarray, optional): array of array of Ap values (magnetic activity index). Defaults to init.nrlmsis00_Aps.
        nrlmsis00_f107_obs (numpy.ndarray, optional): array of f107 observed values. Defaults to init.nrlmsis00_f107_adj.
        nrlmsis00_f107_obs_ctr81d (numpy.ndarray, optional): array of 107 observed centered average of 81 days. Defaults to init.nrlmsis00_f107_adj_81d.
        remove_past_data (bool, optional): whether to reduce the size of the nrlmsis data arrays. Defaults to True.

    Returns:
        Ap, f107_adj, f107_adj_81d: 3 variables: array of Ap values, previous day f107_obs value, f107_obs_81d value respectively
    """

    date_index = np.argmin(np.abs(nrlmsis00_dates - date))

    if date_index > 2 and remove_past_data:
        # if selected date is later than the second entry in the nrlmsis00_dates array, remove past records to save
        # computation time in the future.

        init.nrlmsis00_dates = nrlmsis00_dates[date_index-2:]
        init.nrlmsis00_Aps = nrlmsis00_Aps[date_index-2:]
        init.nrlmsis00_f107_obs = nrlmsis00_f107_obs[date_index-2:]
        init.nrlmsis00_f107_obs_81d = nrlmsis00_f107_obs_ctr81d[date_index-2:]

    return nrlmsis00_Aps[date_index], nrlmsis00_f107_obs[date_index-1], nrlmsis00_f107_obs_ctr81d[date_index]


def find_nrlmsise00_species_and_temp(
        date: datetime.datetime,
        altitude: float,
        latitude: float,
        longitude: float):
    """calculates properties of species and temperature of the astmosphere
        doc: https://pynrlmsise00.readthedocs.io/en/latest/reference/nrlmsise00.html#nrlmsise00.msise_flat

    Args:
        date (datetime.datetime): date and time at which the properties are required
        altitude (float): Altitude at which porperties are required in meters
        latitude (float): latitude in radians north
        longitude (float): longitude in radians east

    Returns:
        total_density, species_densities, temperatures: 
        density of atmosphere in kg/m3,
        array of species densities in kg/m3
        array of 2 temperatures in Kelvin: Exospheric temperature and temperature at altitude
    """

    Aps, f107_obs_m1d, f107_obs_ctr81d = find_nrlmsise00_magnetic_and_solar_properties(
        date.date())

    data = msise_flat(
        date,  # Date and time as a datetime.dateime.
        altitude/1000,  # Altitude in km
        np.rad2deg(latitude),  # Latitude in degrees north.
        np.rad2deg(longitude),  # Longitude in degrees east.
        # The observed f107a (81-day running mean of f107) centred at date
        f107_obs_ctr81d,
        f107_obs_m1d,  # The observed f107 value on the previous day.
        4  # The ap value at date.
    )

    species_number_densities = np.append(data[0:5], data[6:9])
    species_densities = species_M * species_number_densities / N_A * 1000  # kg/m3
    total_density = data[5] * 1000  # kg/m3
    temperatures = data[9:11]  # K

    return total_density, species_densities, temperatures


# Doornbos equations

def get_S_j(V_inc, T_inc, m_j):
    return V_inc / (np.sqrt(2 * k * T_inc/m_j))


def get_G_j(S_j):
    return 1 / (2*S_j**2)


def get_P_ij(S_j, gamma_i):
    return 1/S_j * np.exp(-gamma_i**2 * S_j**2)


def get_Z_ij(gamma_i, S_j):
    return 1 + math.erf(gamma_i * S_j)


def get_V_re_V_inc(alpha, T_sat, V_inc):
    return np.sqrt(0.5*(1 + alpha*((4 * R * T_sat) / V_inc**2 - 1)))


def get_alpha():

    alpha = 0.85  # placeholder!!

    return alpha


def are_vectors_parallel(v1, v2):
    return np.allclose(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)) or \
        np.allclose(v1 / np.linalg.norm(v1), -v2 / np.linalg.norm(v2))


def get_uli(u_d, n_i):
    if are_vectors_parallel(u_d, n_i):
        # panel is perpedicular to flow find random vector perpendicular to flow (lift should be zero in this case)
        # Generate a random vector
        # Generates a random vector with 3 components
        random_vector = np.random.rand(3)

        # Project the random vector onto the plane defined by u_d
        projection = random_vector - np.dot(random_vector, u_d) * u_d
        normalized_projection = projection / np.linalg.norm(projection)
        return normalized_projection
    else:
        return - (np.cross(np.cross(u_d, n_i), u_d)) / np.linalg.norm((np.cross(np.cross(u_d, n_i), u_d)))


def get_Cd_ij(V_inc, T_inc, m_j, T_sat, gamma_i, A_i, A_ref):
    """_summary_

    Args:
        V_inc (_type_): _description_
        T_inc (_type_): _description_
        m_j (_type_): _description_
        T_sat (_type_): _description_
        gamma_i (_type_): _description_
        A_i (_type_): _description_
        A_ref (_type_): _description_

    Returns:
        _type_: _description_
    """

    S_j = get_S_j(V_inc, T_inc, m_j)
    P_ij = get_P_ij(S_j, gamma_i)
    G_j = get_G_j(S_j)
    Q_j = 1 + G_j
    Z_ij = get_Z_ij(gamma_i, S_j)
    alpha = get_alpha()
    V_re_V_inc = get_V_re_V_inc(alpha, T_sat, V_inc)

    Cd_ij = P_ij / np.sqrt(np.pi)
    Cd_ij += gamma_i * Q_j * Z_ij
    Cd_ij += gamma_i / 2 * V_re_V_inc * \
        (gamma_i * np.sqrt(np.pi) * Z_ij + P_ij)
    Cd_ij *= (A_i / A_ref)
    # print(Cd_ij)
    return Cd_ij


def get_Cl_ij(V_inc, T_inc, m_j, T_sat, l_i, gamma_i, A_i, A_ref):
    """_summary_

    Args:
        V_re (_type_): _description_
        T_inc (_type_): _description_
        m_j (_type_): _description_
        T_sat (_type_): _description_
        l_i (_type_): _description_
        gamma_i (_type_): _description_
        A_i (_type_): _description_
        A_ref (_type_): _description_

    Returns:
        _type_: _description_
    """
    S_j = get_S_j(V_inc, T_inc, m_j)
    P_ij = get_P_ij(S_j, gamma_i)
    G_j = get_G_j(S_j)
    Z_ij = get_Z_ij(gamma_i, S_j)
    alpha = get_alpha()
    V_re_V_inc = get_V_re_V_inc(alpha, T_sat, V_inc)

    Cl_ij = l_i * G_j * Z_ij
    Cl_ij += l_i / 2 * V_re_V_inc * (gamma_i * np.sqrt(np.pi) * Z_ij + P_ij)
    Cl_ij *= (A_i / A_ref)

    return Cl_ij


def find_CR_i(V_vect: np.ndarray, n_i, T_inc, T_sat,  A_i, A_ref, rho_tot, rho_j_array, m_j_array=species_M):
    """_summary_

    Args:
        V_vect (numpy.ndarray): velocity vector of satellite
        n_i (_type_): normal unit vector of panel
        T_inc (_type_): _description_
        T_sat (_type_): _description_
        A_i (_type_): _description_
        A_ref (_type_): _description_
        rho_tot (_type_): _description_
        rho_j_array (_type_): _description_
        m_j_array (_type_, optional): _description_. Defaults to species_M.

    Returns:
        _type_: _description_
    """

    CR_i = 0

    V_inc = np.linalg.norm(V_vect)

    u_d = V_vect / V_inc
    u_li = get_uli(u_d, n_i)

    gamma_i = -u_d.dot(n_i)
    l_i = -u_li.dot(n_i)

    for rho_j, m_j in zip(rho_j_array, m_j_array):

        Cd_ij = get_Cd_ij(V_inc, T_inc, m_j, T_sat, gamma_i, A_i, A_ref)
        Cl_ij = get_Cl_ij(V_inc, T_inc, m_j, T_sat, l_i, gamma_i, A_i, A_ref)

        CR_i += (rho_j/rho_tot) * (Cd_ij * u_d + Cl_ij * u_li)

    return CR_i


def find_CM_i(r_cg_i, l_ref, V_vect, n_i, T_inc, T_sat,  A_i, A_ref, rho_tot, rho_j_array, m_j_array=species_M):

    CR_i = find_CR_i(V_vect, n_i, T_inc, T_sat,  A_i,
                     A_ref, rho_tot, rho_j_array, m_j_array)

    CM_i = (1 / l_ref) * np.cross(r_cg_i, CR_i)

    return CM_i


def find_CR(sat_properties_object):
    """_summary_

    Args:
        sat_properties_object (_type_): _description_
        sat_tudat_object (_type_): _description_

    Returns:
        _type_: _description_
    """

    T_body_inertial = sat_properties_object.body_fixed_to_inertial_frame
    T_inertial_body = sat_properties_object.inertial_to_body_fixed_frame
    cartesian_state = sat_properties_object.state
    V_vector = -cartesian_state[3:]
    lat = sat_properties_object.lat
    long = sat_properties_object.long
    altitude = sat_properties_object.altitude
    epoch = sat_properties_object.epoch
    epoch_datetime = datetime.datetime.fromtimestamp(
        init.J2000_to_unix_timestamp_delta_t + epoch)

    panel_pos_vectors_body_fixed = sat_properties_object.panel_pos_vectors
    panel_pos_vectors_intertial = np.array(
        [T_body_inertial @ xi for xi in panel_pos_vectors_body_fixed])

    panel_areas = sat_properties_object.panel_areas
    # r_cg_body = sat_properties_object.r_cg_body
    Aref = sat_properties_object.Aref
    # lref = sat_properties_object.lref

    C_R = 0
    i_tot = 0  # total number of panels pointing into flow

    total_density, species_densities, temperatures = find_nrlmsise00_species_and_temp(
        epoch_datetime,
        altitude,
        lat,
        long)

    for r_i, A_i in zip(panel_pos_vectors_intertial, panel_areas):

        r_i_hat = r_i / np.linalg.norm(r_i)

        if V_vector.dot(r_i_hat) < 0:
            # panel i is pointing into the flow
            i_tot += 1

            C_R += find_CR_i(
                V_vector,
                r_i_hat,
                temperatures[-1],
                283,
                A_i,
                Aref,
                total_density,
                species_densities)

    return T_inertial_body @ C_R


def find_R(sat_properties_object):
    """_summary_

    Args:
        sat_properties_object (_type_): _description_
        sat_tudat_object (_type_): _description_

    Returns:
        _type_: _description_
    """

    T_body_inertial = sat_properties_object.body_fixed_to_inertial_frame
    cartesian_state = sat_properties_object.state
    V_vector = -cartesian_state[3:]
    lat = sat_properties_object.lat
    long = sat_properties_object.long
    altitude = sat_properties_object.altitude
    epoch = sat_properties_object.epoch
    epoch_datetime = datetime.datetime.fromtimestamp(
        init.J2000_to_unix_timestamp_delta_t + epoch)

    panel_pos_vectors_body_fixed = sat_properties_object.panel_pos_vectors
    panel_pos_vectors_intertial = np.array(
        [T_body_inertial @ xi for xi in panel_pos_vectors_body_fixed])

    panel_areas = sat_properties_object.panel_areas
    # r_cg_body = sat_properties_object.r_cg_body
    Aref = sat_properties_object.Aref
    # lref = sat_properties_object.lref

    C_R = 0
    i_tot = 0  # total number of panels pointing into flow

    total_density, species_densities, temperatures = find_nrlmsise00_species_and_temp(
        epoch_datetime,
        altitude,
        lat,
        long)

    for r_i, A_i in zip(panel_pos_vectors_intertial, panel_areas):

        r_i_hat = r_i / np.linalg.norm(r_i)

        if V_vector.dot(r_i_hat) < 0:
            # panel i is pointing into the flow
            i_tot += 1

            C_R += find_CR_i(
                V_vector,
                r_i_hat,
                temperatures[-1],
                283,
                A_i,
                Aref,
                total_density,
                species_densities)

    R = 0.5 * C_R * total_density * np.linalg.norm(V_vector)**2 * Aref

    return T_body_inertial @ R


def find_CM(sat_properties_object):
    """_summary_

    Args:
        sat_properties_object (_type_): _description_
        sat_tudat_object (_type_): _description_

    Returns:
        _type_: _description_
    """

    T_body_inertial = sat_properties_object.body_fixed_to_inertial_frame
    T_inertial_body = sat_properties_object.inertial_to_body_fixed_frame
    cartesian_state = sat_properties_object.state
    V_vector = -cartesian_state[3:]
    lat = sat_properties_object.lat
    long = sat_properties_object.long
    altitude = sat_properties_object.altitude
    epoch = sat_properties_object.epoch
    epoch_datetime = datetime.datetime.fromtimestamp(
        init.J2000_to_unix_timestamp_delta_t + epoch)

    panel_pos_vectors_body_fixed = sat_properties_object.panel_pos_vectors
    panel_pos_vectors_intertial = np.array(
        [T_body_inertial @ xi for xi in panel_pos_vectors_body_fixed])

    panel_areas = sat_properties_object.panel_areas
    r_cg_body_fixed = sat_properties_object.r_cg_body
    r_cg_inertial = T_body_inertial @ r_cg_body_fixed

    Aref = sat_properties_object.Aref
    lref = sat_properties_object.lref

    C_M = 0
    i_tot = 0  # total number of panels pointing into flow

    total_density, species_densities, temperatures = find_nrlmsise00_species_and_temp(
        epoch_datetime,
        altitude,
        lat,
        long)

    for r_i, A_i in zip(panel_pos_vectors_intertial, panel_areas):

        r_i_hat = r_i / np.linalg.norm(r_i)

        if V_vector.dot(r_i_hat) < 0:
            # panel i is pointing into the flow
            i_tot += 1

            r_cg_i = r_i - r_cg_inertial

            C_M += find_CM_i(
                r_cg_i,
                lref,
                V_vector,
                r_i_hat,
                temperatures[-1],
                283,
                A_i,
                Aref,
                total_density,
                species_densities)

    return T_inertial_body @ C_M


def find_M(sat_properties_object):
    """_summary_

    Args:
        sat_properties_object (_type_): _description_
        sat_tudat_object (_type_): _description_

    Returns:
        _type_: _description_
    """

    T_body_inertial = sat_properties_object.body_fixed_to_inertial_frame
    cartesian_state = sat_properties_object.state
    V_vector = -cartesian_state[3:]
    lat = sat_properties_object.lat
    long = sat_properties_object.long
    altitude = sat_properties_object.altitude
    epoch = sat_properties_object.epoch
    epoch_datetime = datetime.datetime.fromtimestamp(
        init.J2000_to_unix_timestamp_delta_t + epoch)

    panel_pos_vectors_body_fixed = sat_properties_object.panel_pos_vectors
    panel_pos_vectors_intertial = np.array(
        [T_body_inertial @ xi for xi in panel_pos_vectors_body_fixed])

    panel_areas = sat_properties_object.panel_areas
    r_cg_body_fixed = sat_properties_object.r_cg_body
    r_cg_inertial = T_body_inertial @ r_cg_body_fixed

    Aref = sat_properties_object.Aref
    lref = sat_properties_object.lref

    C_M = 0
    i_tot = 0  # total number of panels pointing into flow

    total_density, species_densities, temperatures = find_nrlmsise00_species_and_temp(
        epoch_datetime,
        altitude,
        lat,
        long)

    for r_i, A_i in zip(panel_pos_vectors_intertial, panel_areas):

        r_i_hat = r_i / np.linalg.norm(r_i)

        if V_vector.dot(r_i_hat) < 0:
            # panel i is pointing into the flow
            i_tot += 1

            r_i_cg = r_i - r_cg_inertial

            C_M += find_CM_i(
                r_i_cg,
                lref,
                V_vector,
                r_i_hat,
                temperatures[-1],
                283,
                A_i,
                Aref,
                total_density,
                species_densities)

    M = 0.5 * C_M * total_density * np.linalg.norm(V_vector)**2 * Aref * lref

    return T_body_inertial @ M


def get_nrlmsise00_data_for_mp(i, j, k, l, epoch, altitude, lat, long):
    i_out, j_out, k_out, l_out = i, j, k, l
    total_density, species_densities, temperatures = find_nrlmsise00_species_and_temp(
        epoch,
        altitude,
        lat,
        long)

    return [i_out, j_out, k_out, l_out, total_density, species_densities, temperatures]


def prepare_nrlmsise00_data(
    epoch_array,
    altitude_array,
    lat_array,
    long_array,
    mp_nodes=None
):

    density_matrix = np.zeros((len(epoch_array), len(
        altitude_array), len(lat_array), len(long_array)))
    species_densities_matrix = np.zeros((len(epoch_array), len(
        altitude_array), len(lat_array), len(long_array), 8))
    temp_matrix = np.zeros((len(epoch_array), len(
        altitude_array), len(lat_array), len(long_array), 2))

    input_list = []

    for i, epoch in enumerate(epoch_array):
        for j, altitude in enumerate(altitude_array):
            for k, lat in enumerate(lat_array):
                for l, long in enumerate(long_array):
                    input_list.append([i, j, k, l, epoch, altitude, lat, long])

    results = []
    tracking_lst = []
    if mp_nodes:
        pool = mp.Pool(mp_nodes)
        for i, input in enumerate(input_list):
            result = pool.apply_async(get_nrlmsise00_data_for_mp, args=input, callback=make_log(
                tracking_lst, len(input_list)))
            results.append(result)
        pool.close()
        pool.join()

    else:
        i = 0
        for input in input_list:
            result = get_nrlmsise00_data_for_mp(*input)
            results.append(result)
            i += 1
            print(f"{i} / {len(input_list)} completed")

    for r in results:
        i, j, k, l = r[0], r[1], r[2], r[3]
        density_matrix[i][j][k][l] = r[4]
        species_densities_matrix[i][j][k][l] = r[5]
        temp_matrix[i][j][k][l] = r[6]

    np.save("density_matrix.npy", density_matrix)
    np.save("species_densities_matrix.npy", species_densities_matrix)
    np.save("temp_matrix.npy", temp_matrix)


def plot_nrlmsise00_atmos(
    density_matrix,
    species_densities_matrix,
    temp_matrix,
    epoch_array,
    altitude_array,
    lat_array,
    long_array
):

    plt.figure()

    plt.plot(epoch_array, species_densities_matrix[:, 0, 0, 0, 0])
    plt.plot(epoch_array, species_densities_matrix[:, 0, 0, 0, 1])
    plt.plot(epoch_array, species_densities_matrix[:, 0, 0, 0, 2])
    plt.plot(epoch_array, species_densities_matrix[:, 0, 0, 0, 3])
    plt.plot(epoch_array, species_densities_matrix[:, 0, 0, 0, 4])
    plt.plot(epoch_array, species_densities_matrix[:, 0, 0, 0, 5])

    plt.yscale("log")

    plt.figure()

    plt.plot(epoch_array, density_matrix[:, 0, 0, 0])

    plt.figure()

    plt.plot(altitude_array, density_matrix[0, :, 0, 0])

    plt.yscale("log")

    plt.show()

    pass


def testing_func_1():
    epoch_datetime = datetime.datetime.today()

    altitude = 500000

    lat = 0
    long = 0

    total_density, species_densities, temperatures = find_nrlmsise00_species_and_temp(
        epoch_datetime,
        altitude,
        lat,
        long)

    V_vect = np.array([-7000, 0, 0])
    n_i = np.array([1, 0, 0]) / np.linalg.norm(np.array([1, 0, 0]))

    theta_array = np.linspace(0, -90, 100)

    CR_array = np.zeros((len(theta_array), 3))

    for i, theta in enumerate(theta_array):
        n_i = np.array([np.cos(np.deg2rad(theta)),
                       np.sin(np.deg2rad(theta)), 0])
        print(V_vect.dot(n_i))
        CR_array[i] = find_CR_i(V_vect, n_i, temperatures[0], 283,
                                5, 5, total_density, species_densities, m_j_array=species_M)

    Cd = CR_array.T[0]
    Cl = CR_array.T[1]

    plt.plot(theta_array, Cd)
    plt.plot(theta_array, Cl)

    plt.show()

    print(CR_array)


if __name__ == "__main__":
    testing_func_1()
    pass
