import numpy as np
import matplotlib.pyplot as plt
import orbit as o
import initializer as init
import datetime
from tudatpy.kernel.numerical_simulation import propagation_setup
import multiprocessing as mp
import sys    

def make_log(results, len_data):
    def logger(evaluation):
        results.append(evaluation)
        print(f"{len(results)} / {len_data} completed, {len_data - len(results)} remaining")
    return logger

def print_error(exeption):
    
    print(exeption)

def bench_sim_wrapper(sim_properties_dic):
    
    DVS = o.satellite("dvs_properties.csv")    
    init_kepler_elements = [
        6378137 + 550000, # semi_major axis
        0, # eccentricity
        np.pi/2 + 1/180 * np.pi, # inclination
        0, # argument_of_periapsis
        0, # longitude_of_ascending_node
        0, # true_anomaly
    ]    
    orbit = o.Orbit(DVS, init_kepler_elements)    
    
    if sim_properties_dic["integrator_name"] == "rkf_45":
        coefficient_set = propagation_setup.integrator.rkf_45
    elif sim_properties_dic["integrator_name"] == "rkf_56":
        coefficient_set = propagation_setup.integrator.rkf_56
    
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
    time_step = sim_properties_dic["integrator_dt"],
    coefficient_set = coefficient_set)
    sim_properties_dic["integrator_settings"] = integrator_settings
    
    orbit.simulate(sim_properties_dic, show_propagation_start_end=False, print_sim_progress=False)
    
    
def generate_benchmarks(mp_nodes=None):
    
    input_list = []
    
    time_steps = 2**np.arange(1,5,1)
        
    for dt in time_steps:
        
        
        sim_properties_dic = {
            "sim_start_datetime": datetime.datetime(2021, 1, 1),
            "sim_duration": 5 * 24 * 60**2,
            "save_directory": f"benchmark_data/dt={dt}/rkf_45",
            "integrator_name": "rkf_45",
            "integrator_dt": dt,
        }
        
        input_list.append(sim_properties_dic.copy())
                
        sim_properties_dic["save_directory"] = f"benchmark_data/dt={dt}/rkf_56"
        sim_properties_dic["integrator_name"] = "rkf_56"
        sim_properties_dic["integrator_dt"] = dt
        
        input_list.append(sim_properties_dic.copy())
            
    if mp_nodes:    
        if mp_nodes > len(input_list):
            mp_nodes = len(input_list)
        pool = mp.Pool(mp_nodes)
        results = []
        for fun_in in input_list:
            pool.apply_async(bench_sim_wrapper, args=[fun_in], callback=make_log(results, len(input_list)), error_callback=print_error)
        pool.close()
        pool.join()  
    else:
        i = 0
        for fun_in in input_list:
            bench_sim_wrapper(fun_in)
            i += 1
            print(f"{i} / {len(input_list)} completed")

def investigate_benchmarks(time_steps):
    
    angular_vel_error_norm_lst = []
    time_lst = []
    
    for dt in time_steps:
        
        data_folder1 = f"{init.folder_root}/benchmark_data/dt={dt}/rkf_45"
        data_folder2 = f"{init.folder_root}/benchmark_data/dt={dt}/rkf_56"
        
        array1 = np.genfromtxt(data_folder1 + "/PropagationHistory_DependentVariables.dat").T
        array2 = np.genfromtxt(data_folder2 + "/PropagationHistory_DependentVariables.dat").T
        
        angular_vel1 = array1[44:47]
        angular_vel2 = array2[44:47]
        
        angular_vel_error = angular_vel2 - angular_vel1
        
        angular_vel_error_norm = np.linalg.norm(angular_vel_error, axis=0)
        
        angular_vel_error_norm_lst.append(angular_vel_error_norm)
        
        time_hours = (array1[0] - array1[0][0]) / (60**2)

        time_lst.append(time_hours)

        plt.plot(time_hours, angular_vel_error_norm, label=dt)

    plt.legend()

    plt.show()

    pass



if __name__ == "__main__":
    
    investigate_benchmarks([16, 32,64,128,256])