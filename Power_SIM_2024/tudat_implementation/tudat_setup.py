import numpy as np
import math

from tudatpy.interface import spice
from tudatpy.util import result2array
from tudatpy.data import save2txt
from tudatpy import numerical_simulation
from tudatpy.dynamics import (environment, environment_setup, \
    propagation_setup)
from tudatpy.astro import element_conversion, time_conversion
from tudatpy.util import result2array

from datetime import datetime

# Load tudat spice kernels.
spice.load_standard_kernels()

# Defines common directories. 
data_dir = "./data/"

# Defines origin of global reference frame. 
global_frame_origin = "Earth"
global_frame_orientation = "J2000"

# Defines central body of simulation. 
central_body = ["Earth"]

# TODO: Understand what variables are actually necessary here. 
def create_bodies(
    sc_mass: float,
    starting_time: float,
    time_step: float,
    atmo_sim= False,
    constant_attitude= False,
    varying_attitude= False,
    rotation :bool=False,
    initial_att= np.zeros([3,3]),
    rot_rate=0.0,

    ):
    """This function creates the environment (as a system of bodies) used for the 
    simulation. Has the option of using an exponential Earth atmosphere model.
    ------
    Parameters:
    - atmo_sim: defines whether Earth's atmosphere is simulated.

    ------
    Return
    Set of bodies, stored in a SystemOfBodies, that comprises the environment
    """
    # Defines which bodies to include in environment. 
    bodies_to_create = [
        "Earth",
        "Moon",
        "Sun"
    ]

    # Creates default body settings. 
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation
    )

    # Adds atmospheric modelling. 
    if atmo_sim: 
        # Earth Exponential Atmosphere parameters
        # TODO: Choose these properly. Currently using example values. 
        density_scale_height = 7200.0
        density_at_sea = 1.225

        body_settings.get("Earth").atmosphere_settings = (
            environment_setup.atmosphere.exponential(
                scale_height= density_scale_height, 
                surface_density= density_at_sea
            )
        )

    # Create vehicle object.
    body_settings.add_empty_settings("davinci")
    body_settings.get("davinci").constant_mass = sc_mass # [kg]

    # Defines values for drag behavior. 
    # TODO: Define these properly. Currently eyeballed. 
    # TODO: Replace these with parameters that can be changed in main. 
    spacecraft_reference_area = 1
    drag_coefficient = 1.2

    # Defines solar occulting bodies.
    occulting_bodies = dict()
    occulting_bodies["Sun"] = ["Earth"]

    # Defines vehicle solar radiation pressure properties. 
    # NOTE: Not necessary for occultation. Using compute_shadow_function is, 
    # for the moment, annoying. Might make sense to use later. 
    ref_area_srp = 100
    rad_press_coeff = 1.2
    vehicle_target_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
        ref_area_srp, rad_press_coeff, occulting_bodies
        )
    body_settings.get("davinci").radiation_pressure_target_settings = vehicle_target_settings

    # Creates drag settings. 
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        spacecraft_reference_area, [drag_coefficient, 0, 0]
    )
    body_settings.get("davinci").aerodynamic_coefficient_settings = (
        aero_coefficient_settings
    )      

    # Creates environment bodies. 
    bodies = environment_setup.create_system_of_bodies(body_settings)

    return bodies


def create_prop_settings(
    bodies: environment.SystemOfBodies,
    initial_state: np.ndarray,
    initial_time: float,
    termination_condition: propagation_setup.propagator.PropagationTerminationSettings,
    fixed_step_size: float, 
    dependent_variables:list=[],
    atmo_sim:bool=False
) -> propagation_setup.propagator.SingleArcPropagatorSettings:
    """Creates the propagator settings for a perturbed trajectory. Uses rk4
    fixed step integration. 

    ------
    Parameters:
    - bodies: environment.SystemOfBodies. Body objects defining the physical 
    simulation environment
    - initial_state: 6x1 np.ndarray. Cartesian initial state of the vehicle in 
    the simulation
    - initial_time: float. Epoch since J2000 at which the propagation starts.
    - termination_condition: propagation_setup.propagator.PropagationTerminationSettings
    Settings for condition upon which the propagation will be terminated
    - fixed_step_size: float. Time step to use during propagation. (fixed)
    - dependent_variables: list. Defines which dependent variables should be 
    saved. 
    - atmos_sim: bool. Defines whether atmospheric simulation should be 
    performed. 

    ------
    Return
    Propagation settings of the perturbed trajectory.
    """

    # Define bodies that are propagated, and their central bodies of propagation.
    bodies_to_propagate = ["davinci"]
    
    # TODO: Figure out better spherical harmonic degrees to use. 
    # Checks whether atmospheric sims will be performed. 
    match atmo_sim:
        case True:
            earth_prop_settings = [
                propagation_setup.acceleration.spherical_harmonic_gravity(4,4),
                propagation_setup.acceleration.aerodynamic()
            ]
        case False:
            earth_prop_settings = [
                propagation_setup.acceleration.spherical_harmonic_gravity(4,4)
            ]

    # Defines accelerations acting on vehicle.
    acceleration_settings_on_vehicle = dict(
        Sun=[
            propagation_setup.acceleration.point_mass_gravity(),
            # TODO: Figure out if I need srp acceleration for solar flux and 
            # occultation stuff. 
            propagation_setup.acceleration.radiation_pressure()
        ],
        Earth=earth_prop_settings,
        Moon=[
            propagation_setup.acceleration.point_mass_gravity()
        ],       
    )

    # Create global accelerations dictionary.
    acceleration_settings = {"davinci": acceleration_settings_on_vehicle}

    # Create acceleration models.
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_body
    )

    # Create numerical integrator settings.
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        fixed_step_size, 
        coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
    )

    # Create propagation settings.
    propagator_settings = propagation_setup.propagator.translational(
        central_body,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        initial_time,
        integrator_settings,
        termination_condition,
        output_variables=dependent_variables,
    )

    return propagator_settings


def create_rotational_settings(
        bodies: environment.SystemOfBodies, time_step: float):
    
    # Creates the attitude control object. 
    attitude_control_object = nadir_pointing(bodies)
    
    # Creates rotational model settings. 
    # TODO: Vary which custom rotation matrix gets used depending on mode of
    # operation. 
    rotation_model_settings = environment_setup.rotation_model.custom_rotation_model(
            base_frame= "J2000",
            target_frame= "davinci_frame",
            custom_rotation_matrix_function= attitude_control_object.get_nadir_rotation_frame,
            finite_difference_time_step= time_step
        )
    
    # Adds settings to existing environment. 
    environment_setup.add_rotation_model(
        bodies, 'davinci', rotation_model_settings)
    
    return bodies


def propagate_orbit(
        propDurationTime: float,
        timeStep: float, 
        stateStartKep: np.array,
        propStartTime,
        bodies

):

    """ # Converts to Mean Equinocial Elems to sort out eccentricity = 0 issues. 
    stateStartMee = element_conversion.keplerian_to_mee(
        keplerian_elements= stateStartKep
    )
 """
    # Defines initial state from keplerian orbital parameters. 
    stateStartCartesian = element_conversion.keplerian_to_cartesian(
        keplerian_elements= stateStartKep, 
        gravitational_parameter= bodies.get("Earth").gravitational_parameter
    )

    # Creates propagation termination settings. 
    time_termination_settings = propagation_setup.propagator.time_termination(
        termination_time= propStartTime + propDurationTime*60**2
    )

    # Defines dependent variables. 
    dependent_variables = [
        # Stores received irradiance in [W/m^2]. Considers eclypse as 
        # on/off, no penumbra. Dependent column 1 (Zero is time)
        propagation_setup.dependent_variable.received_irradiance_shadow_function(
            target_body= "davinci",
            source_body= "Sun"
        ),
        # Stores position vector of spacecraft relative to the sun in the
        # Earth-centered coordinate frame. Dependent columns (2 to 4)
        propagation_setup.dependent_variable.relative_position(
            body= "davinci", 
            relative_body= "Sun"
        ),
        # Stores the rotation matrix for converting from body_fixed to 
        # inertial reference frame (J2000). Dependent columns (5 to 13)
        propagation_setup.dependent_variable.inertial_to_body_fixed_rotation_frame(
            body= "davinci"
        ),

        ### REMOVABLE DEPENDENTS TODO: REMOVE ONCE UNNECESSARY
        # Stores kepler elements. Columns (-1 to -6) 
        # NOTE: Verification purposes only.  
        propagation_setup.dependent_variable.keplerian_state(
            body= "davinci",
            central_body= "Earth"
        )

    ]

    # Retrieves propagation settings.
    propagation_settings = create_prop_settings(
        bodies= bodies,
        initial_state= stateStartCartesian,
        initial_time= propStartTime,
        termination_condition= time_termination_settings,
        fixed_step_size= timeStep,
        dependent_variables= dependent_variables,
        atmo_sim= False
    )

    # Propagate dynamics. 
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, 
        propagation_settings
    )
    
    # Extract state history and dependent variables. 
    stateHistory = dynamics_simulator.propagation_results.state_history
    dependentHistory = dynamics_simulator.propagation_results.dependent_variable_history

    stateArr = result2array(stateHistory)
    dependentArr = result2array(dependentHistory)

    # Might be useful if you want to save this data at some point in the future. 
    """ # Saves values to data files.  
    save2txt(
        solution= stateHistory, 
        filename= "state_data.csv",
        directory= data_dir
    )
    save2txt(
        solution= dependentHistory,
        filename= "dependent_data.csv",
        directory= data_dir
    ) """

    return stateHistory, dependentHistory, stateArr, dependentArr

    


# TODO: Might be a good idea to place all these (As different modes of ops) 
# in their own python file. 
class nadir_pointing:
    def __init__(self, bodies: environment.SystemOfBodies):
        
        # Extracts bodies of vehicle and Earth. 
        self.vehicle = bodies.get_body("davinci")
        self.earth = bodies.get_body("Earth")
        self.sun = bodies.get_body("Sun")

        # Sets current time as NaN. Used for updating attitude. 
        self.current_time = float("NaN")

    def get_nadir_rotation_frame(self, current_time: float):

        # Checks if time has changed, updates attitude. 
        self.update_guidance(current_time)

        # Creates the body to inertial rotation matrix. 
        self.body_to_inertial_matrix = np.array(
            [self.nadir_unit_vector,
             self.angular_unit_vector,
             self.completing_unit_vector]
        ).T

        # Returns the rotation frame for constant nadir pointing. 
        return self.body_to_inertial_matrix
    
    def update_guidance(self, current_time: float):
        
        if( math.isnan( current_time ) ):
            # Set the model's current time to NaN, 
            # indicating that it needs to be updated. 
            self.current_time = float("NaN")

        elif( current_time != self.current_time ):
            # Extracts vehicle position and velocity. 
            position = self.vehicle.position
            velocity = self.vehicle.velocity

            # Calculates the three unit vectors for a nadir pointing coordinate
            # frame. x:Nadir - y:Angular Momentum - z:Completes right hand.
            self.nadir_unit_vector = - position / np.linalg.norm(position)
            self.angular_unit_vector = np.cross(position, velocity) / \
                np.linalg.norm(np.cross(position, velocity))
            self.completing_unit_vector = np.cross(
                self.nadir_unit_vector, self.angular_unit_vector
            ) 

            # Updates time. 
            self.current_time = current_time
    
