import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup
from tudatpy.kernel.astro import element_conversion, time_conversion
from tudatpy.kernel import constants
from tudatpy.io import save2txt


def create_bodies():
    spice.load_standard_kernels()

    bodies_to_create = ["Earth", "Sun", "Moon"]
    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"

    # Get the default properties of the planetary bodies from the SPICE library
    body_settings = environment_setup.get_default_body_settings(bodies_to_create,
                                                                global_frame_origin,
                                                                global_frame_orientation)

    # Create environment
    bodies = environment_setup.create_system_of_bodies(body_settings)

    return bodies


def dvs(bodies):
    # Create vehicle object
    bodies.create_empty_body('DVS')

    # set mass
    bodies.get_body('DVS').mass = 2.4

    area = (0.022422 * 4 + 0.010201 * 2) / 4
    cd = 1.2
    cr = 1.2

    # set aerodynamics
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(area, [cd, 0, 0])
    environment_setup.add_aerodynamic_coefficient_interface(bodies, 'DVS', aero_coefficient_settings)

    # set solar radiation
    radiation_coefficient_settings = environment_setup.radiation_pressure.cannonball(
        'Sun', area, cr, ['Earth'])
    environment_setup.add_radiation_pressure_interface(bodies, 'DVS', radiation_coefficient_settings)

    return bodies


def acceleration(bodies):
    bodies_to_propagate = ['DVS']
    central_bodies = ['Earth']

    acceleration_acting_on_vehicle = dict(
        Earth=
        [
            propagation_setup.acceleration.spherical_harmonic_gravity(4, 4),
            propagation_setup.acceleration.aerodynamic()
        ],
        Moon=
        [
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Sun=
        [
            propagation_setup.acceleration.point_mass_gravity(),
            propagation_setup.acceleration.cannonball_radiation_pressure()
        ]
    )
    # Create global acceleration dictionary.
    acceleration_settings = {'DVS': acceleration_acting_on_vehicle}

    # Create acceleration models.
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies)

    return acceleration_models


def kepl_to_cart(kepl_elm, mu):
    state = element_conversion.keplerian_to_cartesian(kepl_elm, mu)
    return state


def dep_variables():
    dependent_variables_to_save = [
        propagation_setup.dependent_variable.keplerian_state(
            "DVS", "Earth"
        ),
        propagation_setup.dependent_variable.latitude(
            "DVS", "Earth"
        ),
        propagation_setup.dependent_variable.longitude(
            "DVS", "Earth"
        ),
        propagation_setup.dependent_variable.altitude(
            "DVS", "Earth"
        )
    ]
    return dependent_variables_to_save


def integrator(start_epoch, time_propagation, acceleration_models, initial_state, dependent_variables):
    fixed_step_size = 10.0
    integrator_settings = propagation_setup.integrator.runge_kutta_4(
        fixed_step_size
    )
    central_bodies = ['Earth']
    bodies_to_propagate = ['DVS']

    end_epoch = start_epoch + time_propagation
    # Create propagation settings.
    termination_settings = propagation_setup.propagator.time_termination(end_epoch)
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        start_epoch,
        integrator_settings,
        termination_settings,
        output_variables=dependent_variables
    )

    propagator_settings.print_settings.print_initial_and_final_conditions = True

    return propagator_settings


def simulator(bodies, propagator_settings):
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, propagator_settings)

    propagation_results = dynamics_simulator.propagation_results

    state_history = propagation_results.state_history
    dependent_variables = propagation_results.dependent_variable_history

    return (state_history, dependent_variables)


def save_file(state_history, dependent_variables):
    save2txt(solution=state_history,
             filename='State_history.dat',
             directory='./data/ground_track'
             )

    save2txt(solution=dependent_variables,
             filename='dep_variables.dat',
             directory='./data/ground_track'
             )
    filename = './data/ground_track/State_history.dat'
    filename_dep = './data/ground_track/dep_variables.dat'

    return (filename, filename_dep)


def ground_station(filename_state, filename_dep_variables):
    path = './data/ground_track'
    location_groundstation_lon = 4.3754
    location_groundstation_lat = 51.9899
    R = 6371360

    df = pd.read_csv(filename_dep_variables, sep='\s+', header=None)
    epoch = df.iloc[:, 0].to_numpy()
    kepler_elements = df.iloc[:, 1:7].to_numpy()
    lat = df.iloc[:, 7].to_numpy()
    lon = df.iloc[:, 8].to_numpy()
    alt = df.iloc[:, 9].to_numpy()
    sma = int(kepler_elements[0, 0] / 1000 // 1)
    date = epoch_to_date(epoch[0])

    time_epoch = [t / constants.JULIAN_DAY * 24 - epoch[0] / constants.JULIAN_DAY * 24 for t in epoch]

    Lambda = np.arccos(R / (R + np.mean(alt)))
    Phi_E = np.linspace(0, 2 * np.pi, num=1000)

    # Create hemisphere function
    mask_E = []
    for i in range(len(Phi_E)):
        val = (-Phi_E[i]) % 2 * np.pi
        if val >= 0 and val < np.pi:
            mask_E.append(1.0)
        else:
            mask_E.append(-1.0)

    # Calculate horizon coordinates on the map.
    colat_horizon = np.arccos(
        np.cos(Lambda) * np.cos((90 - location_groundstation_lat) / 180 * np.pi) + np.sin(Lambda) *
        np.sin((90 - location_groundstation_lat) / 180 * np.pi) * np.cos(Phi_E % 2 * np.pi))

    DL = (
        (mask_E * np.arccos(
            (np.cos(Lambda) - np.cos(colat_horizon) * np.cos((90 - location_groundstation_lat) / 180 * np.pi)) /
            (np.sin((90 - location_groundstation_lat) / 180 * np.pi) * np.sin(colat_horizon)))))

    LAT_horizon = (90 - (colat_horizon / np.pi * 180))
    LON_horizon_abs = ((location_groundstation_lon / 180 * np.pi - DL) / np.pi * 180)
    LON_horizon = np.where(LON_horizon_abs <= 180, LON_horizon_abs, LON_horizon_abs - 360)

    # plot groundtrack with visibility area of ground station for the last timesteps
    plt.figure(figsize=(12, 8), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.scatter(lon / np.pi * 180, lat / np.pi * 180, label='DVS', marker='.', s=2)
    ax.scatter(location_groundstation_lon, location_groundstation_lat, color='red', marker='*', s=200)
    ax.plot(LON_horizon, LAT_horizon, color='red')
    ax.gridlines(draw_labels=True)
    rel_path_ground_track = f'Ground_Track_{sma}.png'
    path_0 = f'{path}/{rel_path_ground_track}'
    plt.savefig(path_0)

    # plot elevation and azimuth of the passes (max and horizon)
    sin_rho = R / (R + alt)
    DL_e = np.deg2rad(np.rad2deg(lon) - location_groundstation_lon)
    Lambda_e = np.arccos(
        np.cos(lat) * np.cos(np.deg2rad(location_groundstation_lat)) + np.sin(lat) *
        np.sin(np.deg2rad(location_groundstation_lat)) * np.cos(DL_e))

    eta = np.arctan2(sin_rho * np.sin(Lambda_e), 1 - sin_rho * np.cos(Lambda_e))
    elevation_abs = np.rad2deg(np.arccos(np.sin(eta) / sin_rho))
    elevation_lambda_check = np.where(Lambda_e <= np.arccos(R / (R + alt)), elevation_abs, 0)
    elevation = np.where(np.abs(DL_e) <= 0.5 * np.pi, elevation_lambda_check, 0)

    # Plot elevation
    fig = plt.figure(figsize=(10, 10), dpi=125)
    ax = fig.add_subplot(211)
    ax.set_title(f'Azimuth and Elevation')
    ax.plot(time_epoch, elevation, color='red')
    ax.set_xlabel('Time [hours since start of day]')
    ax.set_ylabel('Elevation [deg]')
    plt.grid()

    # Azimuth calculation

    # Create hemisphere function
    mask_DL = []
    for i in range(len(DL_e)):
        val = (DL_e[i]) % 2 * np.pi
        if val >= 0 and val < np.pi:
            mask_DL.append(1.0)
        else:
            mask_DL.append(-1.0)

    # azimuth
    azimuth_abs = np.rad2deg(mask_DL * np.arccos((np.cos(np.deg2rad(location_groundstation_lat)) - np.cos(lat) *
                                                  np.cos(Lambda_e)) / (np.sin(lat) * np.sin(Lambda_e))))
    azimuth_lambda_check = np.where(Lambda_e <= np.arccos(R / (R + alt)), azimuth_abs, 0)
    azimuth = np.where(np.abs(DL_e) <= 0.5 * np.pi, azimuth_lambda_check, 0)

    ax = fig.add_subplot(212)
    ax.plot(time_epoch, azimuth % 360, color='red')
    ax.set_xlabel('Time [hours since start of day]')
    ax.set_ylabel('azimuth [deg]')
    plt.grid()
    rel_path_angles = f'Azimuth_Elevation_Visibility_{sma}.png'
    path_1 = f'{path}/{rel_path_angles}'
    plt.savefig(path_1)

    time_visibility = np.array([])
    time_between_obs = np.array([])
    var = 0
    i = 0
    temp2 = False

    while i < len(time_epoch):

        if azimuth[i] != 0:
            temp = time_epoch[i]
            if temp2:
                time_between_obs = np.append(time_between_obs, temp - temp2)
                temp2 = False

            j = i
            while j < len(time_epoch):
                if azimuth[j] == 0:
                    time_visibility = np.append(time_visibility, time_epoch[j] - time_epoch[i])
                    i = j - 1
                    temp2 = time_epoch[j]
                    break
                j += 1
        i += 1

    return time_visibility, time_between_obs, rel_path_ground_track, rel_path_angles, date


def epoch_to_date(epoch):
    j2000 = 2451545.0
    jd = epoch / constants.JULIAN_DAY + j2000

    return time_conversion.julian_day_to_calendar_date(jd)


def jd_to_date(epoch):
    import math

    j2000 = 2451545.0
    jd = j2000 + epoch // 1
    fr = epoch - epoch // 1
    if fr >= 1:
        jd += int(fr)
        fr -= int(fr)

    jd_0 = jd + 0.5
    L1 = math.trunc(jd_0 + 68569)
    L2 = math.trunc(4 * L1 / 146097)
    L3 = L1 - math.trunc((146097 * L2 + 3) / 4)
    L4 = math.trunc(4000 * (L3 + 1) / 1461001)
    L5 = L3 - math.trunc(1461 * L4 / 4) + 31
    L6 = math.trunc(80 * L5 / 2447)
    L7 = math.trunc(L6 / 11)
    D = L5 - math.trunc(2447 * L6 / 80)
    M = L6 + 2 - 12 * L7
    Y = 100 * (L2 - 49) + L4 + L7

    hr = math.trunc(fr * 24)
    min = math.trunc((fr * 24 - hr) * 60)
    sec = math.trunc((fr - hr / 24 - min / (24 * 60)) * 24 * 60 * 60)

    time = [Y, M, D, hr, min, sec]

    return time


def write_txt_file(keplerian_elements, time_visibility, time_between_obs, date, ground_track_path, angles_path,
                   txt_file):
    time_visibility *= 60     # min
    time_between_obs *= 3600  # sec
    time_visibility = np.round(time_visibility, 2)
    time_between_obs = np.round(time_between_obs, 2)
    keplerian_elements[0] = keplerian_elements[0] / 1000
    keplerian_elements[2] = keplerian_elements[2] * 180 / np.pi
    keplerian_elements[3] = keplerian_elements[3] * 180 / np.pi
    keplerian_elements[4] = keplerian_elements[4] * 180 / np.pi
    keplerian_elements[5] = keplerian_elements[5] * 180 / np.pi

    keplerian_elements = np.round(keplerian_elements, 3)

    with open(txt_file, 'a') as file:
        file.write(f'Epoch: {date}\n')
        file.write('Keplerian elements\n')
        file.write(f'  a = {keplerian_elements[0]}\t[km]\n')
        file.write(f'  e = {keplerian_elements[1]}\t\t[-]\n')
        file.write(f'  i = {keplerian_elements[2]}\t\t[deg]\n')
        file.write(f' om = {keplerian_elements[3]}\t\t[deg]\n')
        file.write(f' OM = {keplerian_elements[4]}\t\t[deg]\n\n')

        max_col_width = len('Visibility time [mm:ss]') + 5
        file.write('Visibility time [mm:ss]     Time before next Observation [hh:mm:ss]\n')
        len1 = len('Visibility time [mm:ss]')
        len2 = len('Time before next Observation [hh:mm:ss]')
        for i in (['-'] * len1):
            file.write(i)
        file.write('     ')
        for i in (['-'] * len2):
            file.write(i)
        file.write('\n')

        for t1, t2 in zip(time_visibility, time_between_obs):
            min = int(t1//1)
            sec = int((t1 - min)*60//1)
            tim = f'{min}:{sec}'
            if min > 9:
                file.write(f'{tim:<{max_col_width}}{timedelta(seconds=t2)}\n')
            else:
                file.write(f' {tim:<{max_col_width-1}}{timedelta(seconds=t2)}\n')

        file.write('\n')
        file.write(f'Ground Track path: {ground_track_path}\n')
        file.write(f'Observability path: {angles_path}\n')

        length = len('Visibility time [mm:ss]     Time before next Observation [hh:mm:ss]')

        section = ['#'] * length
        for i in section:
            file.write(i)

        file.write('\n\n')

    return txt_file


def run_analysis(keplerian_state):
    bodies = create_bodies()
    bodies = dvs(bodies)
    acceleration_models = acceleration(bodies)
    mu = bodies.get_body('Earth').gravitational_parameter
    initial_state = kepl_to_cart(keplerian_state, mu)
    dependent_variables_to_save = dep_variables()
    start_date = datetime.datetime(2025, 2, 1)
    start_epoch = (time_conversion.calendar_date_to_julian_day(
        start_date) - constants.JULIAN_DAY_ON_J2000) * constants.JULIAN_DAY
    time_propagation = 48 * 60 * 60
    propagator_settings = integrator(start_epoch, time_propagation, acceleration_models, initial_state,
                                     dependent_variables_to_save)
    state_history, dependent_variables = simulator(bodies, propagator_settings)
    state_path, dep_variables_path = save_file(state_history, dependent_variables)
    time_visibility, time_between_obs, path_ground_track, path_angles, date = ground_station(state_path,
                                                                                             dep_variables_path)
    txt_file = './data/ground_track/ground_station_info.txt'
    txt_file = write_txt_file(keplerian_state, time_visibility, time_between_obs, date, path_ground_track, path_angles,
                              txt_file)

    return txt_file
