import pandas as pd
import numpy as np
import datetime
import sys
import initializer as init
from pprint import pprint
import DVS_aerodynamics as aero_funcs


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


def get_accelerations(inputs):
    
    return np.array([1e-6, 1e-6, 1e-6])

def get_torques(inputs):
    
    return np.array([1e-7, 7e-7, 4e-7])

def get_accelerations_depvar():
    
    return np.array([1e-6, 1e-6, 1e-6])

def get_torques_depvar():
    
    return np.array([1e-7, 7e-7, 4e-7])





spice.load_standard_kernels()
# Set simulation time
simulation_start_epoch = 0

# seconds since J2000 
simulation_end_epoch = 0 + 5 * 24*60**2

###########################################################
#
# -------------- Environment Setup ---------------------
#
###########################################################

bodies_to_create = ["Earth"]
global_frame_origin = "Earth"
global_frame_orientation = "J2000"

# Get the default properties of the planetary bodies from the SPICE library
body_settings = environment_setup.get_default_body_settings(bodies_to_create,
                                                            global_frame_origin,
                                                            global_frame_orientation)

# Create the system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# Add the spacecraft to the system of bodies
bodies.create_empty_body("DVS")

############################################################
# Assigning properties of artifical satellite
############################################################

bodies.get("DVS").mass = 1000

bodies.get("DVS").inertia_tensor = np.array([
    [0.5, 0, 0],
    [0, 0.5, 0],
    [0, 0, 0.5]])



# --------------- Propagation Setup ---------------------

# Define the central body and the body which is propagated around it
bodies_to_propagate = ["DVS"]
central_bodies = ["Earth"]

# Create the translational acceleration settings

acceleration_settings_on_vehicle = dict(
    Earth=[propagation_setup.acceleration.point_mass_gravity()]
    )

acceleration_settings_on_vehicle["DVS"] = [propagation_setup.acceleration.custom_acceleration(get_accelerations)]

# Apply the settings to the s/c
acceleration_settings = {"DVS": acceleration_settings_on_vehicle}

# Create the acceleration models
acceleration_models = propagation_setup.create_acceleration_models(
    bodies, 
    acceleration_settings,
    bodies_to_propagate, 
    central_bodies)

torque_settings_spacecraft = {}

torque_settings_spacecraft["DVS"] = [propagation_setup.torque.custom_torque(get_torques)]


torque_settings = {"DVS": torque_settings_spacecraft}

# Create torque models.
torque_models = propagation_setup.create_torque_models( 
    bodies, torque_settings, bodies_to_propagate) 

# --------------- Define initial conditions -------------------

earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter

initial_state_translational = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=earth_gravitational_parameter,
    semi_major_axis=6378000+550000,
    eccentricity=0.1,
    inclination=0.01,
    argument_of_periapsis=1,
    longitude_of_ascending_node=1,
    true_anomaly=1
    )

# Below, we define the initial state in a somewhat trivial manner (body axes along global frame
# axes; no initial rotation). A real application should use a more realistic initial rotational state
# Set initial rotation matrix (identity matrix)
initial_rotation_matrix = np.eye(3)
# Set initial orientation by converting a rotation matrix to a Tudat-compatible quaternion
initial_state_rotational = element_conversion.rotation_matrix_to_quaternion_entries(initial_rotation_matrix)
# Complete initial state by adding angular velocity vector (zero in this case)
initial_state_rotational = np.hstack((initial_state_rotational, [0,0,0]))


# ----- DEFINE DEPENDENT VARIABLES TO SAVE


dependent_variables_to_save = [
    propagation_setup.dependent_variable.total_acceleration("DVS"),
    propagation_setup.dependent_variable.keplerian_state("DVS", "Earth"),
    propagation_setup.dependent_variable.latitude("DVS", "Earth"),
    propagation_setup.dependent_variable.longitude("DVS", "Earth"),
    propagation_setup.dependent_variable.tnw_to_inertial_rotation_matrix("DVS", "Earth"),
    propagation_setup.dependent_variable.total_torque("DVS"),
    propagation_setup.dependent_variable.inertial_to_body_fixed_rotation_frame("DVS"),
    propagation_setup.dependent_variable.heading_angle("DVS", "Earth"),
    propagation_setup.dependent_variable.flight_path_angle("DVS", "Earth"),
    propagation_setup.dependent_variable.angle_of_attack("DVS", "Earth"),
    propagation_setup.dependent_variable.sideslip_angle("DVS", "Earth"),
    propagation_setup.dependent_variable.bank_angle("DVS", "Earth"),
    propagation_setup.dependent_variable.custom_dependent_variable(get_accelerations_depvar, 3),
    propagation_setup.dependent_variable.custom_dependent_variable(get_torques_depvar, 3),
    ]

# ----- CREATE THE INTEGRATOR SETTINGS
fixed_step_size = 5.0
# integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
#     initial_time_step=1.0,
#     coefficient_set = propagation_setup.integrator.CoefficientSets.rkdp_87,
#     minimum_step_size=np.finfo(float).eps,
#     maximum_step_size=np.inf,
#     relative_error_tolerance= 1e-10,
#     absolute_error_tolerance= 1e-10
#     )
integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
    fixed_step_size,
    propagation_setup.integrator.CoefficientSets.rkf_45)

# Create propagation settings.
termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)
translational_propagator_settings = propagation_setup.propagator.translational(
    central_bodies=central_bodies, # (list[str]) – List of central bodies with respect to which the bodies to be integrated are propagated.
    acceleration_models=acceleration_models, # (AccelerationMap) – Set of accelerations acting on the bodies to propagate, provided as acceleration models.
    bodies_to_integrate=bodies_to_propagate, # (list[str]) – List of bodies to be numerically propagated, whose order reflects the order of the central bodies.
    initial_states=initial_state_translational, # (numpy.ndarray) – Initial states of the bodies to integrate (one initial state for each body, concatenated into a single array), provided in the same order as the bodies to integrate. The initial states must be expressed in Cartesian elements, w.r.t. the central body of each integrated body. The states must be defined with the same frame orientation as the global frame orientation of the environment (specified when creating a system of bodies, see for instance get_default_body_settings() and create_system_of_bodies()). Consequently, for N integrated bodies, this input is a vector with size size 6N.
    initial_time=simulation_start_epoch, # initial_time (float) – Initial epoch of the numerical propagation
    integrator_settings=integrator_settings, # integrator_settings (IntegratorSettings) – Settings defining the numerical integrator that is to be used for the propagation
    termination_settings=termination_settings, # (PropagationTerminationSettings) – Generic termination settings object to check whether the propagation should be ended.
    propagator=propagation_setup.propagator.TranslationalPropagatorType.cowell, # (TranslationalPropagatorType, default=cowell) – Type of translational propagator to be used (see TranslationalPropagatorType enum).
    output_variables=[] # (list[SingleDependentVariableSaveSettings], default=[]) – Class to define settings on how the numerical results are to be used, both during the propagation (printing to console) and after propagation (resetting environment)
)

rotational_propagator_settings = propagation_setup.propagator.rotational(
    torque_models=torque_models, # (TorqueModelMap) – Set of torques acting on the bodies to propagate, provided as torque models.
    bodies_to_integrate=bodies_to_propagate, # (list[str]) – List of bodies to be numerically propagated, whose order reflects the order of the central bodies.
    initial_states=initial_state_rotational, # (numpy.ndarray) – Initial rotational states of the bodies to integrate (one initial state for each body), provided in the same order as the bodies to integrate. Regardless of the propagator that is selected, the initial rotational state is always defined as four quaternion entries, and the angular velocity of the body, as defined in more detail here: https://docs.tudat.space/en/latest/_src_user_guide/state_propagation/environment_setup/frames_in_environment.html#definition-of-rotational-state.
    initial_time=simulation_start_epoch, #  (float) – Initial epoch of the numerical propagation
    integrator_settings=integrator_settings, # (IntegratorSettings) – Settings defining the numerical integrator that is to be used for the propagation
    termination_settings=termination_settings, # (PropagationTerminationSettings) – Generic termination settings object to check whether the propagation should be ended.
    propagator=propagation_setup.propagator.RotationalPropagatorType.exponential_map, # (RotationalPropagatorType, default=quaternions) – Type of rotational propagator to be used (see RotationalPropagatorType enum).
    output_variables=[]  # (list[SingleDependentVariableSaveSettings], default=[]) – Class to define settings on how the numerical results are to be used, both during the propagation (printing to console) and after propagation (resetting environment)
)

multiple_propagator_settings = propagation_setup.propagator.multitype(
    propagator_settings_list=[translational_propagator_settings, rotational_propagator_settings],
    integrator_settings=integrator_settings,
    initial_time=simulation_start_epoch, 
    termination_settings=termination_settings,
    output_variables=dependent_variables_to_save
)

multiple_propagator_settings.print_settings.print_initial_and_final_conditions = True

###########################################################################
# PROPAGATE ORBIT #########################################################
###########################################################################

# Create simulation object and propagate dynamics.
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, multiple_propagator_settings )

# Retrieve all data produced by simulation
propagation_results = dynamics_simulator.propagation_results

save_dir = "data/torque_testing"

save2txt(solution=propagation_results.state_history,
        filename="PropagationHistory.dat",
        directory=save_dir
        )

save2txt(solution=propagation_results.dependent_variable_history,
        filename="PropagationHistory_DependentVariables.dat",
        directory=save_dir
        )

save_dependent_variable_info(f"{save_dir}/dependent_variable_info.txt", propagation_results.dependent_variable_ids)









