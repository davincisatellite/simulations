import numpy as np
import DVS_aerodynamics as aero

from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

import custom_propagation_environment as env
import initializer as init

import datetime
import pathlib
import sys
import time
import matplotlib.pyplot as plt
import multiprocessing as mp

import json

from tudatpy.kernel.astro.element_conversion import keplerian_to_cartesian

def make_log(results, len_data):
    def logger(evaluation):
        results.append(evaluation)
        print(f"{len(results)} / {len_data} completed, {len_data - len(results)} remaining")
    return logger


class satellite:
    def __init__(
        self, 
        satellite_properties_file: str
        ):
        """init of cubesat class.

        Args:
            satellite_properties_file: (str): path csv file containing properties of satellite
        """

        sat_properties = init.extract_values_from_csv(satellite_properties_file)
        
        for key, value in sat_properties.items():
            setattr(self, key, value)
        
        self.moi_array = np.array([[self.Ixx, 0,        0       ],
                                   [0,        self.Iyy, 0       ],
                                   [0,        0,        self.Izz]])
        
        self.panel_pos_vectors = np.zeros((6,3))
        self.panel_pos_vectors[0][0] = 0.101/2
        self.panel_pos_vectors[1][1] = 0.101/2
        self.panel_pos_vectors[2][2] = 0.222/2
        self.panel_pos_vectors[3][0] = -0.101/2
        self.panel_pos_vectors[4][1] = -0.101/2
        self.panel_pos_vectors[5][2] = -0.222/2
        
        self.panel_areas = np.array([self.A_0, self.A_1, self.A_2, self.A_3, self.A_4, self.A_5])

        self.r_cg_body = np.array([self.x_cg_body, self.y_cg_body, self.z_cg_body])
        
    def get_power_production(T_body_to_inertial):
        
        
        
        pass


class Orbit:
    def __init__(self, satellite, init_kepler_elements):
        """_summary_

        Args:
            satellite (class): contains all necessary satellite information, created by the
            initializer.py function.
        """
        self.satellite = satellite
        self.init_kepler_elements = init_kepler_elements
        
        self.init_cartesian_elements = keplerian_to_cartesian(init_kepler_elements, 3.986004418e14)
        
        setattr(self.satellite, 'epoch', None)
        
        pass     
           
    def update_sattelite_attributes_from_state(self, t, x):
        if t != self.satellite.epoch:
            state = x[0:6]      
            q = x[6:10]
            
            Rot_inertial_to_body = Rotation.from_quat(q)
            T_inertial_to_body = Rot_inertial_to_body.as_matrix()
            
            Rot_body_to_inertial = Rot_inertial_to_body.inv()
            T_body_to_inertial = Rot_body_to_inertial.as_matrix()
            
            lat, long = env.find_long_lat_form_state(t, state[0:3])
            
            altitude = np.linalg.norm(state[0:3]) - 6378137.0    
            
            setattr(self.satellite, 'lat', lat)
            setattr(self.satellite, 'long', long)    
            setattr(self.satellite, 'body_fixed_to_inertial_frame', T_body_to_inertial)
            setattr(self.satellite, 'inertial_to_body_fixed_frame', T_inertial_to_body)
            setattr(self.satellite, 'state', state)
            setattr(self.satellite, 'altitude', altitude)
            setattr(self.satellite, 'epoch', t)
            
        
    def find_aero_accel(self, t, x):
        
        self.update_sattelite_attributes_from_state(t, x)
        
        a_aero = aero.find_R(self.satellite) / self.satellite.mass
        
        return a_aero
    
    def find_aero_torque(self, t, x):
        
        self.update_sattelite_attributes_from_state(t, x)
        
        M_aero = aero.find_M(self.satellite)
        
        return M_aero
    
    def evaluate_dependent_vars(
        self, 
        dependent_variable_functions, 
        t_history, 
        x_history,
        save_dir=None):
        
        # determine size of data
        no_cols = 1
        no_rows = len(t_history)
        func_data_lengths = np.zeros(len(dependent_variable_functions))
        for i, func in enumerate(dependent_variable_functions):
            func_data_len = len(func(t_history[0], x_history[0]))
            func_data_lengths[i] = func_data_len
            no_cols += func_data_len
            
        data = np.zeros((no_rows, no_cols))
        
        data[:, 0] = t_history
            
        for i, func in enumerate(dependent_variable_functions):
            start_id = int(func_data_lengths[:i].sum()) + 1
            end_id = int(start_id + func_data_lengths[i]) 
            
            for j, (t, x) in enumerate(zip(t_history, x_history)):
                data[j][start_id:end_id] = func(t, x)
                
        if save_dir:   

            path = pathlib.Path(save_dir + "/dependent_variable_history.dat")
            path.parent.mkdir(parents=True, exist_ok=True)
            
            np.savetxt(save_dir + "/dependent_variable_history.dat", data)
    
    def save_propagation_output(self, out, save_dir):
        
        t_history = out.t
        x_history = out.y.T
        
        no_cols = 1 + len(x_history[0])
        no_rows = len(t_history)
        
        data = np.zeros((no_rows, no_cols))
        
        data[:, 1:] = x_history
        data[:, 0] = t_history
        
        num_states_path = save_dir + "/numerical_states_history.dat"
        dict_path = save_dir + "/propagation_info.dat"
        
        path = pathlib.Path(num_states_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savetxt(num_states_path, data)
        
        data_dict = {
            "teval": out.teval,
            "nfev": out.nfev,
            "njev": out.njev,
            "nlu": out.nlu,
            "status": out.status,
            "success": out.success,
            "message": out.message,
        }
        
        with open(dict_path, "w") as fp:
            json.dump(data_dict, fp, indent=4, separators=(',', ': '))
        
    def save_propagation_setup(self, sim_properties_dic, save_dir):
        
        dict_path = save_dir + "/propagation_setup.dat"
        
        path = pathlib.Path(dict_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dict_path, "w") as fp:
            json.dump(sim_properties_dic, fp, indent=4, separators=(',', ': '))
        
    def terminate_on_CPU_time(self, t, y):
        current_run_time = time.time() - self.start_time
        if current_run_time >= self.max_time:
            return 0

        return 1     
    terminate_on_CPU_time.terminal = True
    
    def simulate(self, sim_properties_dict):
        
        J = self.satellite.moi_array
        J_inv = np.linalg.inv(J)
        
        def dx_dt(t, x):
                       
            omega = x[10:13]
            
            o1, o2, o3 = omega[0], omega[1], omega[2]
            
            q = x[6:10]
            
            q = q / np.linalg.norm(q)
            
            q1, q2, q3, q4 = q[0], q[1], q[2], q[3]
            
            Rot_inertial_to_body = Rotation.from_quat(q)
            T_inertial_to_body = Rot_inertial_to_body.as_matrix()
            
            Rot_body_to_inertial = Rot_inertial_to_body.inv()
            T_body_to_inertial = Rot_body_to_inertial.as_matrix()
            
            self.update_sattelite_attributes_from_state(t, x)
            
            a_earth = env.point_mass_earth_accel(x[0:3]) 
            
            a_aero = self.find_aero_accel(t, x)
                        
            M = self.find_aero_torque(t, x)
            
            M = T_body_to_inertial @ M
            
            Q = np.array([
                [q4, -q3, q2, q1],
                [q3, q4, -q1, q1],
                [-q2, q1, q4, q3],
                [-q1, -q2, -q3, q4]
            ])
            
            OQ = np.array([
                [0, o3, -o2, o1],
                [-o3, 0, o1, o2],
                [o2, -o1, 0, o3],
                [-o1, -o2, -o3, 0]
            ])
            
            O = np.array([
                [0, -o3, o2],
                [o3, 0, -o1],
                [-o2, o1, 0]
            ])
            
            dx_dt = np.zeros(13)
            
            dx_dt[0:3] = x[3:6]
            
            dx_dt[3:6] = a_earth + a_aero
            
            dx_dt[6:10] = 0.5 * OQ @ q
            
            dx_dt[10:13] = -J_inv @ O @ J @ omega + J_inv @ M
            
            return dx_dt
        
        x_0 = np.zeros(13)
        
        x_0[0:6] = self.init_cartesian_elements
        
        init_rot = Rotation.from_euler('zyx', [0, 0, 0])
        
        init_quat = init_rot.as_quat()
        
        x_0[6:10] = init_quat
        
        x_0[10:13] = np.ones(3) * np.deg2rad(0.1)
        
        self.start_time = time.time()
        
        self.max_time = sim_properties_dict["max_time"]
                
        try:
            out = solve_ivp(
                dx_dt, 
                [0, sim_properties_dict["sim_duration"]], 
                x_0, 
                rtol=sim_properties_dict["rtol"], 
                atol=sim_properties_dict["atol"],
                events=self.terminate_on_CPU_time,
                method=sim_properties_dict["method"],
                max_step=sim_properties_dict["max_step"]
                ) 
            
            
            duration = time.time() - self.start_time   
            setattr(out, "teval", duration)
                
            t_history = out.t
            x_history = out.y.T
            
            dependent_var_funcs = [self.find_aero_accel, self.find_aero_torque]
            
            self.evaluate_dependent_vars(dependent_var_funcs, t_history, x_history, save_dir=sim_properties_dict["save_directory"])
            
            self.save_propagation_output(out, sim_properties_dict["save_directory"])
            
            self.save_propagation_setup(sim_properties_dict, sim_properties_dict["save_directory"])
        except:
            pass
                
def start_propagation_for_mp(sim_properties_dict):
    
    DVS = satellite("dvs_properties.csv")
    
    init_kepler_elements = [
        6378137 + 550000, # semi_major axis
        0, # eccentricity
        np.pi/2 + 1/180 * np.pi, # inclination
        0, # argument_of_periapsis
        0, # longitude_of_ascending_node
        0, # true_anomaly
    ]
                
    orbit = Orbit(DVS, init_kepler_elements)
    
    orbit.simulate(sim_properties_dict)
        
        
def investigate_propagation(mp_nodes=None):
    
    tols = 10.**np.arange(-2, -15, -1)
    
    methods = ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]
        
    inputs = []
    
    for method in methods:
        for rtol in tols:
            for atol in tols:
                inputs.append(
                    {
                        "sim_duration": 14 * 24 * 60**2 ,
                        "max_time": 5 * 60,
                        "rtol": rtol,
                        "atol": atol,
                        "method": method,
                        "save_directory": f"data/custom_propagator_investigation/{method}/atol{atol:.0e}_rtol{rtol:.0e}",
                        "max_step": np.inf,                        
                    }                    
                )
    
    if mp_nodes:    
        pool = mp.Pool(mp_nodes)
        results = []
        for fun_in in inputs:
            pool.apply_async(start_propagation_for_mp, args=[fun_in], callback=make_log(results, len(inputs)))
        pool.close()
        pool.join()  
    else:
        i = 0
        for fun_in in inputs:
            start_propagation_for_mp(fun_in)
            i += 1
            print(f"{i} / {len(inputs)} completed")
        
def gen_benchmark(mp_nodes=None):
    
    steps = np.array([0.125, 0.25, 0.5])
    
    methods = ["RK45"]
    
    inputs = []
    
    
    for method in methods:
        for step in steps:
            inputs.append(
                {
                    "sim_duration": 14 * 24 * 60**2 ,
                    "max_time": 16 * 60 * 60,
                    "rtol": 50,
                    "atol": 50,
                    "method": method,
                    "save_directory": f"data/custom_propagator_benchmarks/{method}/step_{step}",
                    "max_step": step,                        
                }                    
            )
    
    if mp_nodes:    
        if mp_nodes > len(inputs):
            mp_nodes = len(inputs)
            print(mp_nodes)
        pool = mp.Pool(mp_nodes)
        results = []
        for fun_in in inputs:
            pool.apply_async(start_propagation_for_mp, args=[fun_in], callback=make_log(results, len(inputs)))
        pool.close()
        pool.join()  
    else:
        i = 0
        for fun_in in inputs:
            start_propagation_for_mp(fun_in)
            i += 1
            print(f"{i} / {len(inputs)} completed")




if __name__ == "__main__":
    
    default_sim_properties_dict = {
        "sim_duration": 2 * 24 * 60**2 ,
        "max_time": 30,
        "rtol": 50,
        "atol": 50,
        "method": "RK45",
        "save_directory": f"data/propagation_with_energy",
        "max_step": 30,                        
    }  
    
    gen_benchmark(20)
    