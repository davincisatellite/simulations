import pandas as pd
import numpy as np
import datetime
import sys
import initializer as init
from pprint import pprint
import DVS_aerodynamics as aero_funcs
import time
import matplotlib.pyplot as plt
import os

# Impprting TUdat library coded to work with version 2.12.0 of tudat

from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment, environment_setup, propagation_setup
from tudatpy.kernel.astro import element_conversion, time_conversion
from tudatpy.kernel import constants
from tudatpy.util import result2array
from tudatpy.io import save2txt


def save_dependent_variable_info(filename, dictionary):
    # Open the text file in write mode
    with open(filename, "w") as file:
        # Iterate over the dictionary items
        for key, value in dictionary.items():
            # Write each item to a new row in the text file
            file.write(f"{key}: {value}\n")


class EPS:

    def __init__(self,
                 bodies: environment.SystemOfBodies,
                 satellite,
                 ):
        """_summary_

        Args:
            bodies (environment.SystemOfBodies): _description_
        """

        self.bodies = bodies
        self.satellite = satellite

    def compute_power_consumption(self):

        power_draw = 0  # W

        # TODO: write function that calulates power consumption
        pass

        return power_draw

    def compute_power_generation(self):

        R_E = 6371 * 1000
        R_S = 6.957e+8

        body_fixed_to_inertial_frame = self.bodies.get(
            self.satellite.name).body_fixed_to_inertial_frame
        inertial_to_body_fixed_frame = self.bodies.get(
            self.satellite.name).inertial_to_body_fixed_frame

        # following calculation is described by Doornbos 2011

        r_e_s = self.bodies.get("Sun").state[0:3]
        r_e_sat = self.bodies.get(self.satellite.name).state[0:3]

        r_s_sat = r_e_sat - r_e_s

        r_s_sat_hat = r_s_sat / np.linalg.norm(r_s_sat)

        # r_p_sat = np.dot(r_s_sat_hat, r_e_sat) * r_s_sat_hat
        # r_e_p = r_e_sat - r_p_sat

        # h_g = np.linalg.norm(r_e_p) - R_E

        # R_P = np.linalg.norm(r_p_sat) / np.linalg.norm(r_s_sat) * R_S

        # h_c = h_g

        # h_b = h_g - R_P

        # eta = h_c / (h_c - h_b)

        # print(eta)

        # f_g = 1 - 1/ np.pi * np.arccos(eta) + eta / np.pi * np.sqrt(1 - eta**2)

        # a = np.arcsin( R_S / np.linalg.norm(r_e_s - r_e_sat) )

        # s = r_e_sat

        # b = np.arcsin(R_E / np.linalg.norm(s))

        # c = np.arccos((-s.T @ (r_e_s - r_e_sat)) / (np.linalg.norm(s) * np.linalg.norm(r_e_s - r_e_sat)))

        # x = (c**2 + a**2 - b**2) / (2*c)

        # y = np.sqrt(a**2 - x**2)

        # print(x/a)
        # print((c-x)/b)

        # A = a**2 * np.arccos(x/a) + b**2 * np.arccos((c-x)/b) - c * y

        # v = 1 - A / (np.pi * a**2)

        # print(v)

        v = 1

        S_in = v * 1366

        # print(f_g)

        panel_pos_vectors_body_fixed = self.satellite.panel_pos_vectors
        panel_pos_vectors_intertial = np.array(
            [body_fixed_to_inertial_frame @ xi for xi in panel_pos_vectors_body_fixed])

        solar_panel_areas = self.satellite.panel_areas*0.8

        solar_panel_areas[5] = 0

        power_gen = 0

        for r_i, A_i in zip(panel_pos_vectors_intertial, solar_panel_areas):
            r_i_hat = r_i / np.linalg.norm(r_i)

            if r_s_sat_hat.dot(r_i_hat) < 0:
                # panel i is pointing into the sun

                cos_theta = - np.dot(r_s_sat_hat, r_i_hat)

                # theta = np.rad2deg(np.arccos(cos_theta))
                # print(theta)
                power_gen += S_in * A_i * 0.3 * 0.5 * 0.90392 * cos_theta

        return [power_gen]


class satellite:
    def __init__(
        self,
        satellite_properties_file: str
    ):
        """init of cubesat class.

        Args:
            satellite_properties_file: (str): path csv file containing properties of satellite
        """

        sat_properties = init.extract_values_from_csv(
            satellite_properties_file)

        for key, value in sat_properties.items():
            setattr(self, key, value)

        self.moi_array = np.array([[self.Ixx, 0,        0],
                                   [0,        self.Iyy, 0],
                                   [0,        0,        self.Izz]])

        self.panel_pos_vectors = np.zeros((6, 3))
        self.panel_pos_vectors[0][0] = 0.101/2
        self.panel_pos_vectors[1][1] = 0.101/2
        self.panel_pos_vectors[2][2] = 0.222/2
        self.panel_pos_vectors[3][0] = -0.101/2
        self.panel_pos_vectors[4][1] = -0.101/2
        self.panel_pos_vectors[5][2] = -0.222/2

        self.panel_areas = np.array(
            [self.A_0, self.A_1, self.A_2, self.A_3, self.A_4, self.A_5])

        self.r_cg_body = np.array(
            [self.x_cg_body, self.y_cg_body, self.z_cg_body])


class aerodynamics:
    def __init__(
        self,
        sat_properties_object,
        bodies,
        sim_properties_dict,
        debug_mode=False,
        debug_array_lengths=500000,
        save_dir=None,
        print_sim_progress=False,
    ):

        self.sat_properties_object = sat_properties_object
        self.bodies = bodies
        self.save_dir = save_dir

        # retrieving the satellite object from tudat bodies
        self.sat_tudat_object = self.bodies.get_body(
            self.sat_properties_object.name)

        self.debug_mode = debug_mode

        self.print_sim_progress = print_sim_progress

        self.sim_properties_dict = sim_properties_dict

        self.simulation_start_epoch = (time_conversion.calendar_date_to_julian_day(
            sim_properties_dict["sim_start_datetime"]) - constants.JULIAN_DAY_ON_J2000) * constants.JULIAN_DAY
        self.simulation_end_epoch = self.simulation_start_epoch + \
            sim_properties_dict["sim_duration"]

        if self.print_sim_progress:
            self.last_t_print_sim_progress = time.time()
            print("Starting to print progress reports")

        if self.debug_mode:
            self.get_aero_force_eval_id = 0
            self.get_aero_accel_data = {
                "eval_epochs": np.ones(debug_array_lengths) * np.nan,
                "x_accel": np.ones(debug_array_lengths) * np.nan,
                "y_accel": np.ones(debug_array_lengths) * np.nan,
                "z_accel": np.ones(debug_array_lengths) * np.nan,
            }
            self.get_aero_torque_eval_id = 0
            self.get_aero_torque_data = {
                "eval_epochs": np.ones(debug_array_lengths) * np.nan,
                "x_moment": np.ones(debug_array_lengths) * np.nan,
                "y_moment": np.ones(debug_array_lengths) * np.nan,
                "z_moment": np.ones(debug_array_lengths) * np.nan,
            }

    def update_sattelite_state(self):
        """
        updates the variables used in order to calculate the aerodynamic coefficients using the data
        from the Tudat propagation (this function is called at every timestep). If tudat propagation
        is doing its very first propagation step, states will not have been defined yet and thus
        the returned aerodynamic coefficents are all zero
        """

        self.first_iteration = False

        try:
            # satellite will not have a sate defines in very first propagation step
            self.sat_sate = self.sat_tudat_object.state
        except:
            self.first_iteration = True

        if not self.first_iteration:
            # updating relevant variables if they are available
            sat_spherical_state = element_conversion.cartesian_to_spherical(
                self.sat_sate)

            setattr(self.sat_properties_object, 'lat', sat_spherical_state[1])
            setattr(self.sat_properties_object, 'long', sat_spherical_state[2])
            setattr(self.sat_properties_object, 'body_fixed_to_inertial_frame',
                    self.sat_tudat_object.body_fixed_to_inertial_frame)
            setattr(self.sat_properties_object, 'inertial_to_body_fixed_frame',
                    self.sat_tudat_object.inertial_to_body_fixed_frame)
            setattr(self.sat_properties_object, 'inertial_angular_velocity',
                    self.sat_tudat_object.inertial_angular_velocity)
            setattr(self.sat_properties_object, 'state',
                    self.sat_tudat_object.state)

            if self.print_sim_progress:
                if time.time() - self.last_t_print_sim_progress > 5:
                    # more then 5 seconds have passed since last progress report, print a new one
                    t = self.sat_properties_object.epoch
                    progress = (t - self.simulation_start_epoch) / \
                        (self.simulation_end_epoch -
                         self.simulation_start_epoch) * 100
                    print(f"Progress = {progress:.1f} % \n", end="\r")
                    self.last_t_print_sim_progress = time.time()

        pass

    def get_aero_force_coeffs(self, tudat_inputs):
        """will return aerodynamic force coefficients (defined in body ref frame). This function gets called
        by tudat. Tudat inputs some dependent variables into the tudat_inputs variable which are used by the 
        functions which find the aerodynamic coefficients.

        Returns:
            (numpy.ndarray): array containing aerodynamic force coefficients in the body reference frame.
            Following order: Cx, Cy, Cz. (definition of x, y, z axes to be confirmed)
        """

        self.update_sattelite_state()

        if not self.first_iteration:

            altitude = np.linalg.norm(
                self.sat_properties_object.state[0:3]) - 6378137.0
            setattr(self.sat_properties_object, 'altitude', altitude)
            setattr(self.sat_properties_object, 'epoch', tudat_inputs[0])

        force_coeffs = np.zeros(3)
        if not self.first_iteration:
            force_coeffs = aero_funcs.find_CR(
                self.sat_properties_object
            )

        if self.debug_mode:
            self.get_aero_accel(tudat_inputs[0])

        return force_coeffs

    def get_aero_moment_coeffs(self, tudat_inputs):
        """will return aerodynamic moment coefficients (defined in body ref frame)

        Returns:
            (numpy.ndarray): array containing aerodynamic moment coefficients in the body reference frame,
            centered at the origin of the body reference frame.
            Following order: Cmx, Cmy, Cmz. (definition of x, y, z axes to be confirmed)
        """

        self.update_sattelite_state()

        if not self.first_iteration:

            altitude = np.linalg.norm(
                self.sat_properties_object.state[0:3]) - 6378137.0
            setattr(self.sat_properties_object, 'altitude', altitude)
            setattr(self.sat_properties_object, 'epoch', tudat_inputs[0])

        moment_coeffs = np.zeros(3)
        if not self.first_iteration:
            moment_coeffs = aero_funcs.find_CM(
                self.sat_properties_object
            )

        if self.debug_mode:
            self.get_aero_torque(tudat_inputs[0])

        return moment_coeffs

    def get_aero_accel(self, epoch):

        self.update_sattelite_state()

        self.aero_accel = np.zeros(3)

        if not self.first_iteration:

            altitude = np.linalg.norm(
                self.sat_properties_object.state[0:3]) - 6378137.0
            setattr(self.sat_properties_object, 'altitude', altitude)
            setattr(self.sat_properties_object, 'epoch', epoch)

            force = aero_funcs.find_R(self.sat_properties_object)

            self.aero_accel = force / self.sat_properties_object.mass

        if self.debug_mode:
            i = self.get_aero_force_eval_id
            self.get_aero_accel_data["eval_epochs"][i] = epoch
            self.get_aero_accel_data["x_accel"][i] = self.aero_accel[0]
            self.get_aero_accel_data["y_accel"][i] = self.aero_accel[1]
            self.get_aero_accel_data["z_accel"][i] = self.aero_accel[2]
            self.get_aero_force_eval_id += 1

        return self.aero_accel

    def get_aero_torque(self, epoch):

        self.update_sattelite_state()

        self.aero_torque = np.zeros(3)

        if not self.first_iteration:
            altitude = np.linalg.norm(
                self.sat_properties_object.state[0:3]) - 6378137.0
            setattr(self.sat_properties_object, 'altitude', altitude)
            setattr(self.sat_properties_object, 'epoch', epoch)

            moment = aero_funcs.find_M(self.sat_properties_object)

            self.aero_torque = moment

        if self.debug_mode:
            i = self.get_aero_torque_eval_id
            self.get_aero_torque_data["eval_epochs"][i] = epoch
            self.get_aero_torque_data["x_moment"][i] = self.aero_torque[0]
            self.get_aero_torque_data["y_moment"][i] = self.aero_torque[1]
            self.get_aero_torque_data["z_moment"][i] = self.aero_torque[2]
            self.get_aero_torque_eval_id += 1

        return self.aero_torque

    def get_applied_aero_accel(self):
        return self.aero_accel

    def get_applied_aero_torque(self):
        return self.aero_torque

    def get_body_fixed_angular_velocity(self):

        self.update_sattelite_state()

        body_fixed_angular_velocity = np.zeros(3)

        if not self.first_iteration:

            body_fixed_angular_velocity = self.sat_tudat_object.body_fixed_angular_velocity

        return body_fixed_angular_velocity

    def get_rad_pressure_constants(self):
        """_summary_

        Returns:
            _type_: _description_
        """

        emissivities = np.zeros(6)               # emissivity of each panel
        areas = np.zeros(6)                      # area of each panel
        # diffusion coefficient of each panel
        diffusion_coefficients = np.zeros(6)
        # normals of each panel surfaces in body-fixed reference frame
        panel_surface_normals = np.zeros((6, 3))

        out_dic = {
            "emissivities": emissivities,
            "areas": areas,
            "diffusion_coefficients": diffusion_coefficients,
            "panel_surface_normals": panel_surface_normals
        }

        return out_dic

    def save_debug_data(self):
        if self.debug_mode:

            print(self.get_aero_accel_data)

            accel_df = pd.DataFrame(self.get_aero_accel_data)
            torque_df = pd.DataFrame(self.get_aero_torque_data)

            accel_df.to_csv(self.save_dir + "/get_aero_accel_debug_data.csv")
            torque_df.to_csv(self.save_dir + "/get_aero_torque_debug_data.csv")


class Orbit:
    def __init__(self, satellite, init_kepler_elements):
        """_summary_

        Args:
            satellite (class): contains all necessary satellite information, created by the
            initializer.py function.
        """
        self.satellite = satellite
        self.init_kepler_elements = init_kepler_elements

        pass

    def simulate(self,
                 sim_properties_dict,
                 show_propagation_start_end=False,
                 save_results=True,
                 print_sim_progress=False
                 ):
        """_summary_

        Args:
            show_propagation_start_end (bool, optional): whether to print start and end informaiton to console. Defaults to False.

        Returns:
            object: contains propagation results 
        """

        spice.load_standard_kernels()
        # Set simulation time

        simulation_start_epoch = (time_conversion.calendar_date_to_julian_day(
            sim_properties_dict["sim_start_datetime"]) - constants.JULIAN_DAY_ON_J2000) * constants.JULIAN_DAY

        # seconds since J2000
        simulation_end_epoch = simulation_start_epoch + \
            sim_properties_dict["sim_duration"]

        ###########################################################
        #
        # -------------- Environment Setup ---------------------
        #
        ###########################################################

        bodies_to_create = ["Earth", "Sun", "Moon"]
        global_frame_origin = "Earth"
        global_frame_orientation = "J2000"

        # Get the default properties of the planetary bodies from the SPICE library
        body_settings = environment_setup.get_default_body_settings(bodies_to_create,
                                                                    global_frame_origin,
                                                                    global_frame_orientation)

        # Define atmosphere
        body_settings.get("Earth").atmosphere_settings = environment_setup.atmosphere.nrlmsise00(
            space_weather_file=init.get_space_weather_data_txt_file_path()
        )

        # Create the system of bodies
        bodies = environment_setup.create_system_of_bodies(body_settings)

        # Add the spacecraft to the system of bodies
        bodies.create_empty_body(self.satellite.name)

        ############################################################
        # Assigning properties of artifical satellite
        ############################################################

        bodies.get(self.satellite.name).mass = self.satellite.mass

        bodies.get(self.satellite.name).inertia_tensor = self.satellite.moi_array

        rigid_body_settings = environment_setup.rigid_body.constant_rigid_body_properties(
            self.satellite.mass, np.zeros(3), self.satellite.moi_array)

        environment_setup.add_mass_properties_model(
            bodies, self.satellite.name, rigid_body_settings)

        # Add rotational model to DVS

        # rotation_model_settings = environment_setup.rotation_model.aerodynamic_angle_based(
        #     central_body = "Earth",
        #     base_frame = f"{self.satellite.name}-fixed",
        #     target_frame = "J2000",
        #     )

        # environment_setup.add_rotation_model(bodies, self.satellite.name, rotation_model_settings)

        environment_setup.add_flight_conditions(
            bodies, self.satellite.name, "Earth")

        aero_functions = aerodynamics(self.satellite, bodies, sim_properties_dict, debug_mode=True, save_dir=sim_properties_dict["save_directory"],
                                      print_sim_progress=print_sim_progress)

        # print(dir(environment))

        # sys.exit()

        # Create aerodynamic coefficient interface settings, and add to vehicle
        aero_coefficient_settings = environment_setup.aerodynamic_coefficients.custom_aerodynamic_force_and_moment_coefficients(
            force_coefficient_function=aero_functions.get_aero_force_coeffs,
            moment_coefficient_function=aero_functions.get_aero_moment_coeffs,
            reference_length=self.satellite.lref,
            reference_area=self.satellite.Aref,
            moment_reference_point=[0, 0, 0],
            independent_variable_names=[
                environment.AerodynamicCoefficientsIndependentVariables.time_dependent
            ],
            force_coefficients_frame=environment.AerodynamicCoefficientFrames.positive_body_fixed_frame_coefficients,
            moment_coefficients_frame=environment.AerodynamicCoefficientFrames.positive_body_fixed_frame_coefficients
        )
        environment_setup.add_aerodynamic_coefficient_interface(
            bodies, self.satellite.name, aero_coefficient_settings)

        # Create radiation pressure settings, and add to vehicle
        # radiation_pressure_settings = environment_setup.radiation_pressure.panelled(
        #     source_body=["Sun"],
        #     emissivities=self.satellite.get_rad_pressure_constants()["emissivities"],
        #     areas=self.satellite.get_rad_pressure_constants()["areas"],
        #     diffusion_coefficients=self.satellite.get_rad_pressure_constants()["diffusion_coefficients"],
        #     surface_normals_in_body_fixed_frame=self.satellite.get_rad_pressure_constants()["panel_surface_normals"],
        #     occulting_bodies=["Earth", "Moon"]
        # )
        # environment_setup.add_radiation_pressure_interface(
        #     bodies, self.satellite.name, radiation_pressure_settings)

        # --------------- Propagation Setup ---------------------

        # Define the central body and the body which is propagated around it
        bodies_to_propagate = [self.satellite.name]
        central_bodies = ["Earth"]

        # Create the translational acceleration settings

        acceleration_settings_on_vehicle = dict(
            Earth=[
                propagation_setup.acceleration.spherical_harmonic_gravity(4, 4)],

            Sun=[propagation_setup.acceleration.point_mass_gravity()],

            Moon=[propagation_setup.acceleration.point_mass_gravity()]
        )

        # acceleration_settings_on_vehicle[self.satellite.name] = [propagation_setup.acceleration.custom_acceleration(aero_functions.get_aero_accel)]

        # Apply the settings to the s/c
        acceleration_settings = {
            self.satellite.name: acceleration_settings_on_vehicle}

        # Create the acceleration models
        acceleration_models = propagation_setup.create_acceleration_models(
            bodies,
            acceleration_settings,
            bodies_to_propagate,
            central_bodies)

        # Define torque models
        # Define torque settings acting on spacecraft
        torque_settings_spacecraft = dict(
            Earth=[propagation_setup.torque.aerodynamic()])

        # print((propagation_setup.torque.custom_torque(aero_functions.get_aero_torque)))

        # sys.exit()

        # torque_settings_spacecraft = {self.satellite.name: [propagation_setup.torque.custom_torque(aero_functions.get_aero_torque)]}

        torque_settings = {self.satellite.name: torque_settings_spacecraft}

        # sys.exit()
        # Create torque models.
        torque_models = propagation_setup.create_torque_models(
            bodies, torque_settings, bodies_to_propagate)
        # sys.exit()
        # --------------- Define initial conditions -------------------

        earth_gravitational_parameter = bodies.get(
            "Earth").gravitational_parameter

        initial_state_translational = element_conversion.keplerian_to_cartesian_elementwise(
            gravitational_parameter=earth_gravitational_parameter,
            semi_major_axis=self.init_kepler_elements[0],
            eccentricity=self.init_kepler_elements[1],
            inclination=self.init_kepler_elements[2],
            argument_of_periapsis=self.init_kepler_elements[3],
            longitude_of_ascending_node=self.init_kepler_elements[4],
            true_anomaly=self.init_kepler_elements[5]
        )

        # Below, we define the initial state in a somewhat trivial manner (body axes along global frame
        # axes; no initial rotation). A real application should use a more realistic initial rotational state
        # Set initial rotation matrix (identity matrix)
        initial_rotation_matrix = np.eye(3)
        # Set initial orientation by converting a rotation matrix to a Tudat-compatible quaternion
        initial_state_rotational = element_conversion.rotation_matrix_to_quaternion_entries(
            initial_rotation_matrix)
        # Complete initial state by adding angular velocity vector (zero in this case)
        initial_state_rotational = np.hstack(
            (initial_state_rotational, [np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]))

        # ----- DEFINE DEPENDENT VARIABLES TO SAVE

        eps = EPS(bodies, self.satellite)

        dependent_variables_to_save = [
            propagation_setup.dependent_variable.total_acceleration(
                self.satellite.name),
            propagation_setup.dependent_variable.keplerian_state(
                self.satellite.name, "Earth"),
            propagation_setup.dependent_variable.latitude(
                self.satellite.name, "Earth"),
            propagation_setup.dependent_variable.longitude(
                self.satellite.name, "Earth"),
            propagation_setup.dependent_variable.tnw_to_inertial_rotation_matrix(
                self.satellite.name, "Earth"),
            propagation_setup.dependent_variable.total_torque(
                self.satellite.name),
            propagation_setup.dependent_variable.inertial_to_body_fixed_rotation_frame(
                self.satellite.name),
            propagation_setup.dependent_variable.heading_angle(
                self.satellite.name, "Earth"),
            propagation_setup.dependent_variable.flight_path_angle(
                self.satellite.name, "Earth"),
            propagation_setup.dependent_variable.angle_of_attack(
                self.satellite.name, "Earth"),
            propagation_setup.dependent_variable.sideslip_angle(
                self.satellite.name, "Earth"),
            propagation_setup.dependent_variable.bank_angle(
                self.satellite.name, "Earth"),
            propagation_setup.dependent_variable.custom_dependent_variable(
                aero_functions.get_applied_aero_accel, 3),
            propagation_setup.dependent_variable.custom_dependent_variable(
                aero_functions.get_applied_aero_torque, 3),
            propagation_setup.dependent_variable.custom_dependent_variable(
                aero_functions.get_body_fixed_angular_velocity, 3),
            propagation_setup.dependent_variable.relative_position(
                self.satellite.name, "Sun"),
            propagation_setup.dependent_variable.custom_dependent_variable(
                eps.compute_power_generation, 1)
        ]

        # Create propagation settings.
        termination_settings = propagation_setup.propagator.time_termination(
            simulation_end_epoch)
        translational_propagator_settings = propagation_setup.propagator.translational(
            # (list[str]) – List of central bodies with respect to which the bodies to be integrated are propagated.
            central_bodies=central_bodies,
            # (AccelerationMap) – Set of accelerations acting on the bodies to propagate, provided as acceleration models.
            acceleration_models=acceleration_models,
            # (list[str]) – List of bodies to be numerically propagated, whose order reflects the order of the central bodies.
            bodies_to_integrate=bodies_to_propagate,
            initial_states=initial_state_translational,  # (numpy.ndarray) – Initial states of the bodies to integrate (one initial state for each body, concatenated into a single array), provided in the same order as the bodies to integrate. The initial states must be expressed in Cartesian elements, w.r.t. the central body of each integrated body. The states must be defined with the same frame orientation as the global frame orientation of the environment (specified when creating a system of bodies, see for instance get_default_body_settings() and create_system_of_bodies()). Consequently, for N integrated bodies, this input is a vector with size size 6N.
            # initial_time (float) – Initial epoch of the numerical propagation
            initial_time=simulation_start_epoch,
            # integrator_settings (IntegratorSettings) – Settings defining the numerical integrator that is to be used for the propagation
            integrator_settings=sim_properties_dict["integrator_settings"],
            # (PropagationTerminationSettings) – Generic termination settings object to check whether the propagation should be ended.
            termination_settings=termination_settings,
            # (TranslationalPropagatorType, default=cowell) – Type of translational propagator to be used (see TranslationalPropagatorType enum).
            propagator=propagation_setup.propagator.TranslationalPropagatorType.cowell,
            # (list[SingleDependentVariableSaveSettings], default=[]) – Class to define settings on how the numerical results are to be used, both during the propagation (printing to console) and after propagation (resetting environment)
            output_variables=[]
        )

        rotational_propagator_settings = propagation_setup.propagator.rotational(
            # (TorqueModelMap) – Set of torques acting on the bodies to propagate, provided as torque models.
            torque_models=torque_models,
            # (list[str]) – List of bodies to be numerically propagated, whose order reflects the order of the central bodies.
            bodies_to_integrate=bodies_to_propagate,
            # (numpy.ndarray) – Initial rotational states of the bodies to integrate (one initial state for each body), provided in the same order as the bodies to integrate. Regardless of the propagator that is selected, the initial rotational state is always defined as four quaternion entries, and the angular velocity of the body, as defined in more detail here: https://docs.tudat.space/en/latest/_src_user_guide/state_propagation/environment_setup/frames_in_environment.html#definition-of-rotational-state.
            initial_states=initial_state_rotational,
            # (float) – Initial epoch of the numerical propagation
            initial_time=simulation_start_epoch,
            # (IntegratorSettings) – Settings defining the numerical integrator that is to be used for the propagation
            integrator_settings=sim_properties_dict["integrator_settings"],
            # (PropagationTerminationSettings) – Generic termination settings object to check whether the propagation should be ended.
            termination_settings=termination_settings,
            # (RotationalPropagatorType, default=quaternions) – Type of rotational propagator to be used (see RotationalPropagatorType enum).
            propagator=propagation_setup.propagator.RotationalPropagatorType.quaternions,
            # (list[SingleDependentVariableSaveSettings], default=[]) – Class to define settings on how the numerical results are to be used, both during the propagation (printing to console) and after propagation (resetting environment)
            output_variables=[]
        )

        multiple_propagator_settings = propagation_setup.propagator.multitype(
            propagator_settings_list=[
                translational_propagator_settings, rotational_propagator_settings],
            integrator_settings=sim_properties_dict["integrator_settings"],
            initial_time=simulation_start_epoch,
            termination_settings=termination_settings,
            output_variables=dependent_variables_to_save
        )

        # sys.exit()
        multiple_propagator_settings.print_settings.print_initial_and_final_conditions = show_propagation_start_end

        ###########################################################################
        # PROPAGATE ORBIT #########################################################
        ###########################################################################

        # Create simulation object and propagate dynamics.
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            bodies, multiple_propagator_settings)

        # Retrieve all data produced by simulation
        propagation_results = dynamics_simulator.propagation_results

        save_dir = sim_properties_dict["save_directory"]

        if save_results:
            save2txt(solution=propagation_results.state_history,
                     filename="PropagationHistory.dat",
                     directory=save_dir
                     )

            save2txt(solution=propagation_results.dependent_variable_history,
                     filename="PropagationHistory_DependentVariables.dat",
                     directory=save_dir
                     )

            save_dependent_variable_info(
                f"{save_dir}dependent_variable_info.txt", propagation_results.dependent_variable_ids)

            aero_functions.save_debug_data()

        return propagation_results


def run_simple_sim():
    sim_properties_dic = {
        "sim_start_datetime": datetime.datetime(2025, 2, 1),
        "sim_duration": 12 * 60**2,
        "save_directory": f"data/simple_sim",
        "integrator_name": "rkf_45",
        "integrator_dt": 1,
    }

    DVS = satellite("dvs_properties.csv")
    init_kepler_elements = [
        6880000,  # semi_major axis
        0.01,  # eccentricity
        np.pi/2 + 9/180 * np.pi,  # inclination
        0,  # argument_of_periapsis
        0,  # longitude_of_ascending_node
        0,  # true_anomaly
    ]
    orbit = Orbit(DVS, init_kepler_elements)

    if sim_properties_dic["integrator_name"] == "rkf_45":
        coefficient_set = propagation_setup.integrator.rkf_45
    elif sim_properties_dic["integrator_name"] == "rkf_56":
        coefficient_set = propagation_setup.integrator.rkf_56

    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        time_step=sim_properties_dic["integrator_dt"],
        coefficient_set=coefficient_set)
    sim_properties_dic["integrator_settings"] = integrator_settings

    orbit.simulate(sim_properties_dic,
                   show_propagation_start_end=False, print_sim_progress=True)


def analyse_simple_sim():

    # os.mkdir(f"{init.folder_root}/data/simple_sim")
    data = np.genfromtxt(
        f"{init.folder_root}/data/simple_sim/PropagationHistory_DependentVariables.dat").T

    time_hours = (data[0] - data[0][0]) / 60**2

    plt.plot(time_hours, data[-1])

    print("Avg power generation:", np.mean(data[-1]))

    plt.grid()

    plt.xlabel("Time since propagation start [hours]")

    plt.ylabel("Power gen [W]")

    plt.show()


if __name__ == "__main__":

    run_simple_sim()
    analyse_simple_sim()
