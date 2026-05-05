import numpy as np
import pandas as pd
import os
from tudatpy.dynamics import environment_setup, simulator, propagation_setup
from tudatpy.interface import spice
from tudatpy.astro import time_representation, element_conversion
from tudatpy.dynamics.propagation_setup import dependent_variable
from tudatpy.dynamics.propagation import create_dependent_variable_dictionary
from tudatpy.util import result2array


def create_bodies(reference_area, drag_coefficient, radiation_pressure_coefficient, satellite_mass):
    spice.load_standard_kernels()

    bodies_to_create = ["Earth", "Sun", "Moon"]

    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"
    body_settings = environment_setup.get_default_body_settings(bodies_to_create, global_frame_origin, global_frame_orientation)

    body_settings.add_empty_settings("DVS")
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(reference_area, [drag_coefficient, 0.0, 0.0])
    body_settings.get("DVS").aerodynamic_coefficient_settings = aero_coefficient_settings

    occulting_bodies_dict = dict()
    occulting_bodies_dict["Sun"] = ["Earth"]
    vehicle_target_settings = environment_setup.radiation_pressure.cannonball_radiation_target(reference_area, radiation_pressure_coefficient, occulting_bodies_dict)
    body_settings.get("DVS").radiation_pressure_target_settings = vehicle_target_settings

    bodies = environment_setup.create_system_of_bodies(body_settings)
    bodies.get("DVS").mass = satellite_mass  # kg

    return bodies


def prop_setup(bodies):
    bodies_to_propagate = ["DVS"]
    central_bodies = ["Earth"]

    accelerations_settings_dvs = dict(
        Sun=[
            propagation_setup.acceleration.radiation_pressure(),
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Earth=[
            propagation_setup.acceleration.spherical_harmonic_gravity(5, 5),
            propagation_setup.acceleration.aerodynamic()
        ],
        Moon=[
            propagation_setup.acceleration.point_mass_gravity()
        ]
    )

    acceleration_settings = {"DVS": accelerations_settings_dvs}

    acceleration_models = propagation_setup.create_acceleration_models(bodies, acceleration_settings, bodies_to_propagate, central_bodies)

    return acceleration_models


def epochs(start_date, end_date):
    simulation_start_epoch = time_representation.date_time_components_to_epoch(*start_date)
    simulation_end_epoch = time_representation.date_time_components_to_epoch(*end_date)

    return simulation_start_epoch, simulation_end_epoch


def kepl_to_cart(keplerian_parameters, gravitational_parameter):
    cartesian_state = element_conversion.keplerian_to_cartesian(keplerian_parameters, gravitational_parameter)
    return cartesian_state


def dep_variables():
    dependent_variables_to_save = [
        dependent_variable.keplerian_state("DVS", "Earth"),
        dependent_variable.latitude("DVS", "Earth"),
        dependent_variable.longitude("DVS", "Earth"),
        dependent_variable.altitude("DVS", "Earth")
    ]
    return dependent_variables_to_save


def integrator(simulation_end_epoch, accelerations, initial_state, simulation_start_epoch, dependent_variables_to_save):

    termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)
    fixed_step_size = 10.0
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(fixed_step_size, coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4)

    central_bodies = ['Earth']
    bodies_to_propagate = ['DVS']

    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        accelerations,
        bodies_to_propagate,
        initial_state,
        simulation_start_epoch,
        integrator_settings,
        termination_condition,
        output_variables = dependent_variables_to_save
    )
    propagator_settings.print_settings.print_initial_and_final_conditions = True

    return propagator_settings


def sim(bodies, propagator_settings):

    dynamics_simulator = simulator.create_dynamics_simulator(bodies, propagator_settings)

    states_history = dynamics_simulator.propagation_results.state_history
    dep_vars_history = dynamics_simulator.propagation_results.dependent_variable_history

    return dynamics_simulator, states_history, dep_vars_history


def save_simulation_to_csv_pandas(filename, states_array, keplerian_state_array, latitude, longitude, altitude, sim_id, initial_keplerian_state):

    base_dir = os.path.dirname(__file__)
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)

    initial_sma = initial_keplerian_state[0]
    initial_e   = initial_keplerian_state[1]
    initial_i   = initial_keplerian_state[2]

    file_exists = os.path.isfile(filepath)
    file_not_empty = file_exists and os.path.getsize(filepath) > 0

    if file_not_empty:
        try:
            existing = pd.read_csv(filepath)

            match = existing[
                (existing["initial_sma"] == initial_sma) &
                (existing["initial_e"] == initial_e) &
                (existing["initial_i"] == initial_i)
                ]

            if not match.empty:
                print(f"[SKIP] Simulation already exists for SMA={initial_sma}, e={initial_e}, i={initial_i}")
                return

        except pd.errors.EmptyDataError:
            print("[INFO] File exists but is empty, treating as new file")
            file_not_empty = False

    df = pd.DataFrame(states_array, columns=["time","x","y","z","vx","vy","vz"])

    df_kepler = pd.DataFrame(
        keplerian_state_array,
        columns=["sma","e","i","arg_periapsis","raan","true_anomaly"]
    )

    df = pd.concat([df, df_kepler], axis=1)

    df["latitude"] = latitude
    df["longitude"] = longitude
    df["altitude"] = altitude

    df["sim_id"] = sim_id
    df["initial_sma"] = initial_sma
    df["initial_e"] = initial_e
    df["initial_i"] = initial_i

    df.to_csv(
        filepath,
        mode='a' if file_not_empty else 'w',
        header=not file_not_empty,
        index=False
    )

    print(f"[✓] Saved sim {sim_id} (SMA={initial_sma}, e={initial_e}, i={initial_i})")


def run_analysis(keplerian_state, reference_area, drag_coefficient, radiation_pressure_coefficient, satellite_mass, start_date, end_date, sim_id):

    bodies = create_bodies(reference_area, drag_coefficient, radiation_pressure_coefficient, satellite_mass)

    accelerations = prop_setup(bodies)

    simulation_start_epoch , simulation_end_epoch = epochs(start_date, end_date)

    mu = bodies.get('Earth').gravitational_parameter
    initial_state = list(kepl_to_cart(keplerian_state, mu))

    dependent_variables_to_save = dep_variables()

    propagator_settings = integrator(simulation_end_epoch, accelerations, initial_state, simulation_start_epoch, dependent_variables_to_save)

    dynamics_simulator, states_history, dep_vars_history = sim(bodies, propagator_settings)

    states_array = result2array(states_history) # shape: epochs x 7, [epoch time, x, y, z, u, v, w]
    dep_var_dict = create_dependent_variable_dictionary(dynamics_simulator)

    keplerian_state_array = dep_var_dict.asarray(dependent_variable.keplerian_state("DVS", "Earth"))
    latitude = dep_var_dict.asarray(dependent_variable.latitude("DVS", "Earth"))
    longitude = dep_var_dict.asarray(dependent_variable.longitude("DVS", "Earth"))
    altitude = dep_var_dict.asarray(dependent_variable.altitude("DVS", "Earth"))

    save_simulation_to_csv_pandas("ground_simulations_data.csv", states_array,  keplerian_state_array, latitude, longitude, altitude, sim_id, keplerian_state)

