import matplotlib.pyplot as plt
import numpy as np
from tudatpy.kernel import constants
import os
import pandas as pd
import pathlib

def plot_keplerian_elements(array: np.ndarray, 
                            start_col=1, 
                            plot_deviation=True, 
                            plot_all_in_one=False, 
                            path_to_save=None, 
                            show_max_perc_change=False, 
                            filter_theta=False):
    """generates plot(s) of keplerian elements

    Args:
        array (numpy.ndarray): array containing relevant data
        start_col (int, optional): start of column with keplerian elements. Defaults to 1.
        plot_deviation (bool, optional): whether to plot the deviation w.r.t. the starting value. Defaults to True.
        plot_all_in_one (bool, optional): whether to plot all keplerian elements as subplots in 1 figure. Defaults to False.
        path_to_save (bool, optional): whether to save plots in a specific path in addition to the default one. Defaults to False.
        show_max_perc_change (str, optional): whether to show some text on the plot with the max deviation w.r.t. start value. Defaults to None.
        filter_theta (bool, optional): whether to replace values of theta close to 360 with 0, useful when plotting deviation from initial state. Defaults to False.
    """
    
    data_names = ["a", "e", "i", "omega", "RAAN", "theta"]
    data_labels = ["semi major axis", "eccentricity", "inclination", "argument of periapsis", "RAAN", "true anomaly"]
    scale_labels = ["km", "-", "degrees", "degrees", "degrees", "degrees"]
    
    time = array[0]
    
    time_days = [ t / constants.JULIAN_DAY - array[0][0] / constants.JULIAN_DAY for t in time ]
        
    scale_factor = 1
    
    if filter_theta:
        
        true_anomaly = array[start_col+5] 
        
        for i in range(len(true_anomaly)):
            if true_anomaly[i] - true_anomaly[i-1] > (35/18)*np.pi:
                true_anomaly[i] -= 2*np.pi
                
        array[start_col+5] = true_anomaly
    
            
    if not plot_all_in_one:
    
        for i in range(6):
            
            plt.figure()
            
            percentage_deviation = find_max_perc_change(array[start_col+i])
            
            mean_percentage_deviation = find_mean_perc_change(array[start_col+i])
            
            if plot_deviation:
                array[start_col+i] -= array[start_col+i][0]
                plt.title(f"Change in {data_labels[i]} w.r.t. initial conditions")
            else:
                plt.title(data_labels[i])
            
            
            if i in [2, 3, 4, 5]:
                scale_factor = 180/np.pi
            if i == 0:
                scale_factor = 1/1000
            
            plt.plot(time_days, array[start_col+i] * scale_factor)
            
            plt.ticklabel_format(useOffset=False)
            
            plt.ylabel(scale_labels[i])
            plt.xlabel("days")
            plt.grid()
            
            if show_max_perc_change:

                plt.text(0, min(array[start_col+i] * scale_factor), 
                         f'Max deviation = {np.format_float_scientific(percentage_deviation, 2, sign=True)} % \nMean deviation = {np.format_float_scientific(mean_percentage_deviation, 2, sign=True)} % ', 
                         fontsize = 8,  
                         verticalalignment='bottom',
                         horizontalalignment='left',
                         bbox = dict(facecolor = 'white', alpha = 0.6))
            
            plt.tight_layout()  
            
            if path_to_save:      
                plt.savefig(path_to_save + f"{data_names[i]}.pdf")
            
            scale_factor = 1
    else:    
        fig, _ = plt.subplots(3, 2, figsize=(9, 12))
        if plot_deviation:
            fig.suptitle('Change of Kepler elements w.r.t. initial conditions')
        else:
            fig.suptitle('Evolution of Kepler elements')
            
        axs = fig.get_axes()

        for i in range(6):
            percentage_deviation = find_max_perc_change(array[start_col+i])
            mean_percentage_deviation = find_mean_perc_change(array[start_col+i])
            if plot_deviation:
                array[start_col+i] -= array[start_col+i][0]
                
            axs[i].title.set_text(data_labels[i])
            
            if i in [2, 3, 4, 5]:
                scale_factor = 180/np.pi
            if i == 0:
                scale_factor = 1/1000
            
            axs[i].plot(time_days, array[start_col+i] * scale_factor, linewidth=0.6)
            axs[i].set_ylabel(scale_labels[i])
            
            axs[i].set_xlabel('days')
            axs[i].grid()
        
            axs[i].ticklabel_format(useOffset=False)
            
            if show_max_perc_change:                
                axs[i].text(0, min(array[start_col+i] * scale_factor), 
                            f'Max deviation = {np.format_float_scientific(percentage_deviation, 2, sign=True)} % \nMean deviation = {np.format_float_scientific(mean_percentage_deviation, 2, sign=True)} % ', 
                            fontsize = 8,  
                            verticalalignment='bottom',
                            horizontalalignment='left',
                            bbox = dict(facecolor = 'white', alpha = 0.6))
        plt.tight_layout()  
        if path_to_save:
            plt.savefig(path_to_save + "all_kepler_elements.pdf")
        
        
def find_max_perc_change(array):
        
    start_value = array[0]

    max_deviation_id = np.argmax(np.abs(array - start_value))

    percentage_deviation = (array[max_deviation_id] - start_value) / start_value * 100

    return percentage_deviation

def find_mean_perc_change(array):
        
    start_value = array[0]

    mean_deviation = (array - start_value).mean()

    percentage_deviation = mean_deviation / start_value * 100

    return percentage_deviation

def plot_3d_orbits(
    cartesian_arrays, points=None, legend=None, path_to_save=None, 
    title=None, equal_axis=False, line_styles=None, line_colors=None,
    point_styles=None):
    
    # Define a 3D figure using pyplot
    fig = plt.figure(figsize=(6,6), dpi=125)
    ax = fig.add_subplot(111, projection='3d')
    if title:
        ax.set_title(title)

    # Plot the positional state history
    
    for i, array in enumerate(cartesian_arrays):
        line_plot, = ax.plot(array[0], array[1], array[2])
        if line_styles:
            plt.setp(line_plot, linestyle=line_styles[i])
        if line_colors:
            plt.setp(line_plot, color=line_colors[i])
            
    
    if points:
        for i, point in enumerate(points):
            point_plot, = ax.scatter(point[1], point[2], point[3])
            if point_styles:
                pass
            



    # Add the legend and labels, then show the plot
    if legend:
        ax.legend(legend)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    
    if equal_axis:
        max_value = np.max(np.array(cartesian_arrays)*1.05)
        min_value = np.min(np.array(cartesian_arrays)*1.05)
        
        max_value = np.around(max_value, -int(np.floor(np.log10(abs(max_value)))))
        min_value = np.around(min_value, -int(np.floor(np.log10(abs(min_value)))))

        ax.set_xlim((min_value, max_value))
        ax.set_ylim((min_value, max_value))
        ax.set_zlim((min_value, max_value))
    
    if path_to_save:
        plt.savefig(path_to_save)

def plot_single_dependent_vars(array, columns, path_to_save=False, y_label=None, title=None, log=False, 
                              x_lim=None, y_lim=None, scale_factor=1, init_offset=False, legend=False):
    """
    
    

    Args:
        array (_type_): _description_
        columns (_type_): _description_
        path_to_save (bool, optional): _description_. Defaults to False.
        y_label (_type_, optional): _description_. Defaults to None.
        title (_type_, optional): _description_. Defaults to None.
        log (bool, optional): _description_. Defaults to False.
        x_lim (_type_, optional): _description_. Defaults to None.
        y_lim (_type_, optional): _description_. Defaults to None.
        scale_factor (int, optional): _description_. Defaults to 1.
        init_offset (bool, optional): _description_. Defaults to False.
        legend (bool, optional): _description_. Defaults to False.
    """
       
    time = array[0]
    
    time_days = [ t / constants.JULIAN_DAY - time[0] / constants.JULIAN_DAY for t in time ]
    
    plt.figure()
    
    for column in columns:
    
        if init_offset:
            array[column] = array[column] - array[column][0]
        
        if log:
            plt.semilogy(time_days, array[column]* scale_factor)
        else:
            plt.plot(time_days, array[column]* scale_factor)
        
    if title:
        plt.title(title)
    
    plt.grid()
    plt.xlabel("days")
    if y_label:
        plt.ylabel(y_label)
    
    if x_lim:
        plt.xlim(x_lim)
    
    if y_lim:
        plt.ylim(y_lim)
        
    if legend:
        plt.legend(legend)
    
    plt.tight_layout()
    
    if path_to_save:
        directory_path = os.path.dirname(path_to_save)
        
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)
        
        plt.savefig(path_to_save)
        
        
def plt_debug_data(folder_path):
    
    accel_df = pd.read_csv(folder_path + "/get_aero_accel_debug_data.csv")
    torque_df = pd.read_csv(folder_path + "/get_aero_torque_debug_data.csv")
    
    # accel_df.sort_values(by=['eval_epochs'])
    # torque_df.sort_values(by=['eval_epochs'])
    
    
    plt.figure()
    
    plt.title("Debug data")
    
    plt.plot(torque_df["eval_epochs"], torque_df["x_moment"], label="x")
    plt.plot(torque_df["eval_epochs"], torque_df["y_moment"], label="y")
    plt.plot(torque_df["eval_epochs"], torque_df["z_moment"], label="z")
    
    plt.legend()
    
    
    
    plt.figure()
    
    plt.title("Debug data")
    
    plt.plot(accel_df["eval_epochs"], accel_df["x_accel"], label="x")
    plt.plot(accel_df["eval_epochs"], accel_df["y_accel"], label="y")
    plt.plot(accel_df["eval_epochs"], accel_df["z_accel"], label="z")
    
    plt.legend()
    
    
def plot_arrays(
    x_arrays, y_arrays, path_to_save=False, title=None, x_label=None, y_label=None, scale_factor=None,
    legend=None, grid=True, x_log=False, linestyles=None, linewiths=None, plot_size=None, colors=None, x_lim=None, legend_pos=None,
    force_xticks=False, force_sci_notation=False, custom_legend_entries=None, custom_markings=None, markers=None, marker_colors=None,
    markerfacecolors=None, marker_sizes=None, keep_in_memory=False, y_log=False, markings=None):
    
    if type(x_arrays[0]) not in [list, np.ndarray]:
        x_arrays = [x_arrays] * len(y_arrays)
    
    plt.figure()
    
    if plot_size:
        fig = plt.gcf()
        fig.set_size_inches(plot_size[0], plot_size[1])
    
    if scale_factor:
        y_arrays = y_arrays*scale_factor
    
    if title:
        plt.title(title)
        
    if x_label:
        plt.xlabel(x_label)

    if y_label:
        plt.ylabel(y_label)
    
    i = 0
    for x_array, y_array in zip(x_arrays, y_arrays):
        line_plot, = plt.plot(x_array, y_array)
        if linestyles:
            plt.setp(line_plot, linestyle=linestyles[i])
        if linewiths:
            plt.setp(line_plot, linewidth=linewiths[i])            
        if colors:
            if len(colors) == 1:
                colors = colors * len(y_arrays)
            plt.setp(line_plot, color=colors[i])
        if markings:
            if not type(markings) == list:
                markings = ["."] * len(y_arrays)
            else:
                if len(markings) == 1:
                    markings = markings * len(y_arrays)
            plt.setp(line_plot, marker=markings[i])
            
        i+=1 
        
    plt.ticklabel_format(useOffset=False)
        
    if custom_markings:
        if markerfacecolors == None:
            markerfacecolor=['none'] * len(custom_markings)
        elif len(markerfacecolors) == 1:
            markerfacecolors = markerfacecolors * len(custom_markings)
            
        if markers == None:
            markers = ["o"] * len(custom_markings)
        elif len(markers) == 1:
            markers = markers * len(custom_markings)
            
        if marker_colors == None:
            marker_colors = ["black"] * len(custom_markings)
        elif len(marker_colors) == 1:
            marker_colors = marker_colors * len(custom_markings)
        
        if marker_sizes == None:
            marker_sizes = [5] * len(custom_markings)
        elif len(marker_sizes) == 1:
            marker_sizes = marker_sizes * len(custom_markings)
            
        for i, mark in enumerate(custom_markings):
            plt.plot(mark[0], mark[1], marker=markers[i], markerfacecolor=markerfacecolor[i], color=marker_colors[i],
                     markersize=marker_sizes[i], markeredgewidth=2)
            
    
    if force_sci_notation:
        if force_sci_notation == "x":
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        if force_sci_notation == "y":
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        
    if legend:
        if custom_legend_entries:
            plt.legend(custom_legend_entries, legend)
        else:
            plt.legend(legend)
        if legend_pos:
            if legend_pos == "below":
                plt.legend(legend, loc='upper center', bbox_to_anchor=(0.5, -0.18))
            if legend_pos == "right":
                plt.legend(legend, loc='center left', bbox_to_anchor=(1, 0.5))
            if legend_pos == "inside_top_left":
                plt.legend(legend, loc='upper left')
                
                
        
    if grid:
        plt.grid()
        
    if x_log:
        plt.xscale('log')
        
    if y_log:
        plt.yscale('log')
        
    if x_lim:
        plt.xlim(x_lim[0], x_lim[1])
    
    if force_xticks:
        plt.xticks(force_xticks)
    
    plt.tight_layout()
        
    if path_to_save:
        
        
        
        path = pathlib.Path(path_to_save)
        path.parent.mkdir(parents=True, exist_ok=True)
            
        plt.savefig(path_to_save, bbox_inches='tight')

    if keep_in_memory == False:
        plt.close()