""" The code takes all the inputs in SI units"""
import numpy as np

import datetime
import math
# Load the tudatpy modules
import tudatpy.kernel.astro.time_conversion
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation as ns
from tudatpy.kernel.numerical_simulation import environment_setup as es
from tudatpy.kernel.numerical_simulation import environment
from tudatpy.kernel.numerical_simulation import propagation
from tudatpy.kernel.numerical_simulation import propagation_setup as ps
from tudatpy.kernel.astro import element_conversion as ec
from tudatpy.kernel import constants
from tudatpy.util import result2array
import tudatpy.plotting as plotter

# Load spice kernels
spice.load_standard_kernels()

# ------- MISSION INPUTS --------
datetime_utc_obj = datetime.datetime(year=2022, month=3, day=20, hour=12, minute=0, second=0)   # To align the ECEF frame with vernal equinox
# datetime_utc_obj = datetime.datetime.now()
n_days = 0.066  # number of days for the simulation to run
h = 550 * 1000  # height of orbit in m
R = round(spice.get_average_radius("Earth"), 3)  # Radius of Earth in m
a = R + h  # Semi-major axis
e = 0.0  # eccentricity of orbit
icl = np.deg2rad(97.5)  # inclination of orbit
omega = np.deg2rad(0)  # Argument of periapsis
OMEGA = np.deg2rad(157.5)  # Longitude of right ascending node
theta = np.deg2rad(0)  # True anomaly

# Convert the input date to epoch J2000 format
datetime_utc = datetime_utc_obj.strftime("%Y-%m-%dT%H:%M:%S")
datetime_et = spice.convert_date_string_to_ephemeris_time(datetime_utc)
#  Set simulation start and end epochs
simulation_start_epoch = datetime_et
simulation_end_epoch = datetime_et + constants.JULIAN_DAY * n_days

if __name__ == "__main__":

    # ------------------------------------ ENVIRONMENT SETUP ------------------------------------
    # Custom class for rotation model

    # ----- CREATE CELESTIAL BODIES

    # Define string names for bodies to be created from default.
    bodies_to_create = ["Sun", "Earth", "Moon"]

    # Use "Earth"/"J2000" as global frame origin and orientation.
    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"

    # Create default body settings, usually from `spice`.
    body_settings = es.get_default_body_settings(
        bodies_to_create,
        global_frame_origin,
        global_frame_orientation)

    # Create system of selected celestial bodies
    bodies = es.create_system_of_bodies(body_settings)

    # ----- CREATE SPACECRAFT

    bodies.create_empty_body("DVS")

    bodies.get("DVS").mass = 2.5  # [kg]
    # body dimension: 22.7 x 10 x 10 cm^3

    class SimpleCustomGuidanceModel:

        def __int__(self, bodies: environment.SystemOfBodies):
            # Extract the Satellite and Earth bodies
            self.vehicle = bodies.get("DVS")
            self.earth = bodies.get("Earth")
            self.state = bodies.get("DVS").state
            self.position = self.state[:, 1:4]
            self.velocity = self.state[:, 4:8]

            # self.current_time = float("NaN")

        def getRotationMatrix(self):
            dis = np.linalg.norm(self.position)
            speed = np.linalg.norm(self.velocity)
            pos_norm = self.position / dis
            vel_norm = self.velocity / speed

            y = np.array([0, 0, 0]) - pos_norm
            x = np.cross(pos_norm, vel_norm)
            z = np.cross(x, y)

            return np.array([y, x, z]).T


    # Create guidance object
    guidance_model = SimpleCustomGuidanceModel

    # Extract the guidance function
    rotation_model_function = guidance_model.getRotationMatrix

    # Create rotation settings from custom model
    rotation_model_settings = es.rotation_model.custom_rotation_model(base_frame=global_frame_origin,
                                                                         target_frame="DVS",
                                                                         custom_rotation_matrix_function
                                                                         =rotation_model_function,
                                                                         finite_difference_time_step=0.1)

    # es.add_rotation_model(bodies, "DVS", rotation_model_settings)

    #
    #
    # # Create rotation model and add to vehicle
    # es.add_rotation_model(bodies, "DVS", rotation_matrix_settings)
    # # TODO: ADD THE AERODYNAMIC AND RADIATION PRESSURE PROPERTIES TO THE SPACECRAFT
    #
    #
    #
    #
    # # Create the rotational model settings and assign to body settings of vehicle




    # ------------------------------------ PROPAGATION SETUP ------------------------------------
    # Define bodies that are propagated
    bodies_to_propagate = ["DVS"]

    # Define central bodies of propagation
    central_bodies = ["Earth"]

    # ----- CREATE ACCELERATION MODEL
    # Define accelerations acting on DVS by Sun and Earth.
    # TODO: ADD THE RADIATION PRESSURE AND AERODYNAMIC DRAG FORCES
    accelerations_settings_dvs = dict(

        Earth=[ps.acceleration.point_mass_gravity()]

    )

    # Create global accelerations settings dictionary.
    acceleration_settings = {"DVS": accelerations_settings_dvs}

    # Create acceleration models.
    acceleration_models = ps.create_acceleration_models(bodies, acceleration_settings, bodies_to_propagate, central_bodies)

    # ----- DEFINE INITIAL STATE
    # Set initial conditions for the satellite that will be
    # propagated in this simulation. The initial conditions are given in
    # Keplerian elements and later on converted to Cartesian elements
    earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter
    initial_state = ec.keplerian_to_cartesian_elementwise(
        gravitational_parameter=earth_gravitational_parameter,
        semi_major_axis=a,
        eccentricity=e,
        inclination=icl,
        argument_of_periapsis=omega,
        longitude_of_ascending_node=OMEGA,
        true_anomaly=theta,
    )

    # ----- DEFINE DEPENDENT VARIABLES TO SAVE
    # Define list of dependent variables to save
    # Keep the same order
    # TODO: check what is the difference between latitude and geodetic latitude
    # TODO: Add the radiation pressure and aerodynamic drag acceleration to the dependent variables later on
    dependent_variables_to_save = [ps.dependent_variable.total_acceleration("DVS"),
                                   ps.dependent_variable.keplerian_state("DVS", "Earth"),
                                   ps.dependent_variable.latitude("DVS", "Earth"),
                                   ps.dependent_variable.longitude("DVS", "Earth"),
                                   ps.dependent_variable.tnw_to_inertial_rotation_matrix("DVS", "Earth")]
                                   # ps.dependent_variable.inertial_to_body_fixed_rotation_frame("DVS")]

    # ----- CREATE THE PROPAGATOR SETTINGS
    # Create termination settings
    termination_condition = ps.propagator.time_termination(simulation_end_epoch)

    # Create propagation settings
    propagator_settings = ps.propagator.translational(central_bodies, acceleration_models, bodies_to_propagate,
                                                      initial_state, termination_condition,
                                                      output_variables=dependent_variables_to_save)

    # ----- CREATE THE INTEGRATOR SETTINGS
    fixed_step_size = 10.0
    integrator_settings = ps.integrator.runge_kutta_4(simulation_start_epoch, fixed_step_size)

    # ------------------------------------ PROPAGATE THE ORBIT ------------------------------------
    # Create simulation object and propagate the dynamics
    dynamics_simulator = ns.SingleArcSimulator(bodies, integrator_settings, propagator_settings)

    # Extract the resulting state and dependent variable history and convert it to an ndarray
    states = dynamics_simulator.state_history
    states_array = result2array(states)
    dep_vars = dynamics_simulator.dependent_variable_history
    dep_vars_array = result2array(dep_vars)

    np.save("data", dep_vars_array)
    np.save("states", states_array)



