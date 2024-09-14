import initializer as init
import orbit as orb
import numpy as np
import data_visualizer as dv
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation  
import sys


def investiage_orbit_1(path): 

    array = np.genfromtxt(path + "PropagationHistory_DependentVariables.dat").T
    
    # dv.plot_keplerian_elements(array, start_col=4, plot_all_in_one=True, plot_deviation=False)
    
    # array = np.genfromtxt(path + "PropagationHistory.dat").T
    
    # dv.plot_single_dependent_vars(array, [35, 36, 37], path_to_save=False, y_label=None, title=None, log=False, 
    #                           x_lim=None, y_lim=None, scale_factor=np.rad2deg(1), init_offset=False, 
    #                           legend=["Aoa", "sideslip", "bank angle"])
    
    dv.plot_single_dependent_vars(array, [41, 42, 43, 21, 22, 23], path_to_save=False, y_label=None, title=None, log=False, 
                              x_lim=None, y_lim=None, scale_factor=1, init_offset=False, 
                              legend=["my Torque xbody", "my Torque ybody", "my Torque zbody", "Torque xbody", "Torque ybody", "Torque zbody"])
    
    dv.plot_single_dependent_vars(array, [38, 39, 40], path_to_save=False, y_label=None, title=None, log=False, 
                              x_lim=None, y_lim=None, scale_factor=1, init_offset=False, 
                              legend=["ax", "ay", "az"])
    
    dv.plot_single_dependent_vars(array, [44, 45, 46], path_to_save=False, y_label=r"Body angular velocity [$deg/s$]", title=None, log=False, 
                              x_lim=None, y_lim=None, scale_factor=np.rad2deg(1), init_offset=False,
                              legend=[r"$\omega_1$", r"$\omega_2$", r"$\omega_3$"])
    

investiage_orbit_1("debug_data/test/")

dv.plt_debug_data("debug_data/test/")

plt.show()