import numpy as np
import data_visualizer as dv
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
from scipy.interpolate import interp1d
import multiprocessing as mp
import sys


def make_log(results, len_data):
    def logger(evaluation):
        results.append(evaluation)
        print(f"{len(results)} / {len_data} completed, {len_data - len(results)} remaining")
    return logger



def compare_benchmarks(regen_data):#=False):
    
    
    if regen_data:
        
        
        steps = np.array([0.125, 0.25, 0.5])
        
        time_in_days_lst = []
        dt_array_lst = []
        mag_omega_lst = []
        labels = []
        
        for step in steps:
            
            labels.append(f"Step={step}s")
            
            directory = f"data/custom_propagator_benchmarks/RK45/step_{step}"
            
            num_sates = np.genfromtxt(directory + "/numerical_states_history.dat").T
            
            time_in_days = num_sates[0] / (24 * 60**2)
            
            dt_array = np.diff(num_sates[0], prepend=0)
            omega = num_sates[11:14]
            mag_omega = np.linalg.norm(omega, axis=0)
            
            time_in_days_lst.append(time_in_days)
            dt_array_lst.append(dt_array)
            mag_omega_lst.append(mag_omega)
    
    
    dv.plot_arrays(time_in_days_lst, dt_array_lst, keep_in_memory=True, legend=labels)
    dv.plot_arrays(time_in_days_lst, mag_omega_lst, keep_in_memory=True, legend=labels)
    
    plt.show()
    
def gen_data(filepath, benchmark_interpolated_num_states, benchmark_interpolated_dep_vars, evaluation_time_array, recalc_error):
               
        if not os.path.isfile(filepath + "/numerical_states_history.dat"):
            return False
        
        if recalc_error:
                
            num_states = np.genfromtxt(filepath + "/numerical_states_history.dat").T
            dep_vars = np.genfromtxt(filepath + "/dependent_variable_history.dat").T
        
            if num_states[0][-1] < evaluation_time_array[-1]:
                return False
        
            interp_num_states = interp1d(num_states[0, :-1], num_states[1:, :-1], kind="cubic")(evaluation_time_array)
            interp_dep_vars = interp1d(dep_vars[0, :-1], dep_vars[1:, :-1], kind="cubic")(evaluation_time_array)
            
            num_states_error = np.zeros((len(interp_num_states)+1, len(evaluation_time_array)))
            num_states_error[0] = evaluation_time_array
            num_states_error[1:] = interp_num_states - benchmark_interpolated_num_states
            
            dep_vars_error = np.zeros((len(interp_dep_vars)+1, len(evaluation_time_array)))
            dep_vars_error[0] = evaluation_time_array
            dep_vars_error[1:] = interp_dep_vars - benchmark_interpolated_dep_vars
        
            np.savetxt(filepath + "/interpolated_numerical_states_history.dat", interp_num_states.T)
            np.savetxt(filepath + "/interpolated_dependent_variable_history.dat", interp_dep_vars.T)
            np.savetxt(filepath + "/numerical_states_history_error.dat", num_states_error.T)
            np.savetxt(filepath + "/dependent_variable_history_error.dat", dep_vars_error.T)
        
        else:
            if not os.path.isfile(filepath + "/numerical_states_history_error.dat"):
                return False
            
            interp_num_states = np.genfromtxt(filepath + "/interpolated_numerical_states_history.dat").T
            interp_dep_vars = np.genfromtxt(filepath + "/interpolated_dependent_variable_history.dat").T
            num_states_error = np.genfromtxt(filepath + "/numerical_states_history_error.dat").T
            dep_vars_error = np.genfromtxt(filepath + "/dependent_variable_history_error.dat").T
        
        
        omega_error = num_states_error[11:14]
        mag_omega_error = np.linalg.norm(omega_error, axis=0)
        bench_omega = benchmark_interpolated_num_states[11:14]
        mag_omega_p_error = mag_omega_error / np.linalg.norm(bench_omega, axis=0) * 100
        
        
        arrays_to_stack = [
            evaluation_time_array,
            mag_omega_error,
            mag_omega_p_error
        ]
        
        plot_data = np.vstack(arrays_to_stack)
        np.savetxt(filepath + "/error_plot_data.dat", plot_data.T)
        
        return True
    
def evaluate_integrators(benchmark_path, evaluation_time_array, mp_nodes=None, recalc_error=False):
    
    tols = 10.**np.arange(-2, -15, -1)
    
    methods = ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]
    
    inputs = []
    
    if recalc_error:
    
        bench_num_states = np.genfromtxt(benchmark_path + "/numerical_states_history.dat").T
        bench_dep_vars = np.genfromtxt(benchmark_path + "/dependent_variable_history.dat").T
            
        benchmark_interpolated_num_states = interp1d(bench_num_states[0, :-1], bench_num_states[1:, :-1], kind="cubic")(evaluation_time_array)
        benchmark_interpolated_dep_vars = interp1d(bench_dep_vars[0, :-1], bench_dep_vars[1:, :-1], kind="cubic")(evaluation_time_array)
        
        np.savetxt(benchmark_path + "/interpolated_numerical_states_history.dat", benchmark_interpolated_num_states.T)
        np.savetxt(benchmark_path + "/interpolated_dependent_variable_history.dat", benchmark_interpolated_dep_vars.T)
    else:
    
        benchmark_interpolated_num_states = np.genfromtxt(benchmark_path + "/interpolated_numerical_states_history.dat").T
        benchmark_interpolated_dep_vars = np.genfromtxt(benchmark_path + "/interpolated_dependent_variable_history.dat").T
    
    for method in methods:
        for atol in tols:
            for rtol in tols:
                inputs.append([f"data/custom_propagator_investigation/{method}/atol{atol:.0e}_rtol{rtol:.0e}",
                               benchmark_interpolated_num_states,
                               benchmark_interpolated_dep_vars,
                               evaluation_time_array,
                               recalc_error
                               ])

    if mp_nodes:    
        pool = mp.Pool(mp_nodes)
        results = []
        for fun_in in inputs:
            pool.apply_async(gen_data, args=fun_in, callback=make_log(results, len(inputs)))
        pool.close()
        pool.join()  
    else:
        i = 0
        for fun_in in inputs:
            a = gen_data(*fun_in)           
            i += 1
            print(f"{i} / {len(inputs)} completed")
    
    
    
def plot_omega_error_vs_tols():
    tols = 10.**np.arange(-2, -15, -1)
    
    methods = ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]
    # methods = ["RK45"]
        
    for method in methods:
        
        error_data = np.zeros((len(tols), len(tols))) * np.nan
        error_p_data = np.zeros((len(tols), len(tols))) * np.nan
        
        for i, rtol in enumerate(tols):
            print(i+1, "/", len(tols))
            for j, atol in enumerate(tols):
                filepath = f"data/custom_propagator_investigation/{method}/atol{atol:.0e}_rtol{rtol:.0e}"
                if os.path.isfile(filepath + "/error_plot_data.dat"):
                    error_plot_data = np.genfromtxt(filepath + "/error_plot_data.dat").T
                    error_data[i][j] = np.max(error_plot_data[1])
                    
                    if np.max(error_plot_data[2]) < 50:
                        error_p_data[i][j] = np.max(error_plot_data[2])
        
        ######################################
        # max error plots   
        ######################################
        
        fig, ax = plt.subplots()
        im = ax.imshow(error_data, 
                        norm=colors.LogNorm(vmin=np.nanmin(error_data), 
                                            vmax=np.nanmax(error_data)))
        
        fig.suptitle(method)
        
        fig.set_figheight(4.8)
        fig.set_figwidth(5)
                
        ax.grid(which='major', linewidth=0.2)
        
        ax.set_xlabel("Absolute tolerance [-]")
        ax.set_ylabel("Relative tolerance [-]")
        
        ax.set_xticks(np.arange(len(tols)), labels=[f"{tols[i]:.0e}" for i in range(len(tols))], rotation=45)
        ax.set_yticks(np.arange(len(tols)), labels=[f"{tols[i]:.0e}" for i in range(len(tols))])
                
        cbar = fig.colorbar(im, orientation='vertical', fraction=0.046, pad=0.04)
        # cbar.set_ticks(cbar_values) # set the tick locations
        # cbar.set_ticklabels([np.format_float_scientific(cbar_values[i], precision=2) for i in range(len(cbar_values))]) # set the tick labels
        cbar.set_label(r'Maximum $\omega$ error [rad/s]', rotation=270, labelpad=12) # set the colorbar label

        fig.tight_layout()          
        fig.savefig(f"data/custom_propagator_investigation/{method}_tolerances_vs_omega_error.pdf", bbox_inches='tight')   
        
        fig, ax = plt.subplots()
        im = ax.imshow(error_p_data)
        
        # percentage error
        
        fig.suptitle(method)
        
        fig.set_figheight(4.8)
        fig.set_figwidth(5)
                
        ax.grid(which='major', linewidth=0.2)
        
        ax.set_xlabel("Absolute tolerance [-]")
        ax.set_ylabel("Relative tolerance [-]")
        
        ax.set_xticks(np.arange(len(tols)), labels=[f"{tols[i]:.0e}" for i in range(len(tols))], rotation=45)
        ax.set_yticks(np.arange(len(tols)), labels=[f"{tols[i]:.0e}" for i in range(len(tols))])
                       
                
        cbar = fig.colorbar(im, orientation='vertical', fraction=0.046, pad=0.04)
        
        cbar.set_label(r'Maximum $\omega$ error [%]', rotation=270, labelpad=12) # set the colorbar label

        fig.tight_layout()          
        fig.savefig(f"data/custom_propagator_investigation/{method}_tolerances_vs_omega_p_error.pdf", bbox_inches='tight')   

    plt.show()
    
    
def plot_data(folders, labels):
    
    
    time_lst = []
    mag_omega_array = []
    dt_lst = []
    
    mag_pos_array = []
    
    
    for folder in folders:
    
        num_sates = np.genfromtxt(folder + "/numerical_states_history.dat").T
        
        time_in_days = num_sates[0] / (24 * 60**2)
        
        dt_array = np.diff(num_sates[0], prepend=0)
        omega = num_sates[11:14]
        mag_omega = np.linalg.norm(omega, axis=0)
        
        pos_mag_array = np.linalg.norm(num_sates[1:4], axis=0)
        
        time_lst.append(time_in_days)
        dt_lst.append(dt_array)
        mag_omega_array.append(mag_omega * np.rad2deg(1))
        mag_pos_array.append(pos_mag_array/1000)
        
    
    
    dv.plot_arrays(time_lst, dt_lst, keep_in_memory=True, x_label="Time [days]", y_label="Timestep [s]", legend=labels)
    dv.plot_arrays(time_lst, mag_omega_array, keep_in_memory=True, x_label="Time [days]", y_label=r"Body rate ($\omega$) [deg/s]", legend=labels)
    dv.plot_arrays(time_lst, mag_pos_array, keep_in_memory=True, x_label="Time [days]", y_label=r"Radial position [km]", legend=labels)
    
    plt.show()
    

if __name__ == "__main__":

    compare_benchmarks(regen_data=True)
    
    evaluate_integrators("data/custom_propagator_benchmarks/RK45/step_0.5", np.arange(0, 0.5*24*60**2+1, 10), mp_nodes=6, recalc_error=True)
    
    plot_omega_error_vs_tols()
    
    plot_data([
        "data/custom_propagator_investigation/BDF/atol1e-12_rtol1e-09",
        "data/custom_propagator_investigation/LSODA/atol1e-11_rtol1e-09",
        "data/custom_propagator_investigation/DOP853/atol1e-13_rtol1e-10",
        "data/custom_propagator_investigation/DOP853/atol1e-13_rtol1e-13",
        "data/custom_propagator_investigation/DOP853/atol1e-14_rtol1e-14"
        ],
        [
            "BDF atol-12, rtol-09",
            "LSODA atol-11, rtol-09",
            "DOP853 atol-13, rtol-10",
            "DOP853 atol-13, rtol-13",
            "DOP853 atol-14, rtol-14",
        ])
    
    pass
    
