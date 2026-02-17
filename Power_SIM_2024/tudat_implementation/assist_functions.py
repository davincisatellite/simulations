import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import numpy.typing as npt

# Plot style lists.
plotStyles      = ["-", "--", "-.", ":"]
plotColors      = ["r", "g", "b", "k"]

def read_files():
            """Returns specific case data as arrays. Specifically for Q2.
            """
            filename = f"dependent_data.csv"
            dependents = pd.read_csv(
                "./data/"+filename, delimiter="\t", header= None)
            dependent_arr = dependents.to_numpy()

            filename = f"state_data.csv"
            states = pd.read_csv(
                "./data/"+filename, delimiter="\t", header= None)
            state_array = states.to_numpy()

            return state_array, dependent_arr

def plot_average_heatmap(
        eccVals: npt.NDArray,
        incVals: npt.NDArray,
        semiMajorVals: npt.NDArray,
        dataDir: str,
        runCount: int,
        powerReq: float= 4.0,
        showUncompliant: bool= False,
):
    valuesDir = dataDir + f"run_num_{runCount}/orbit_averages/"
    plotsDir = dataDir + f"run_num_{runCount}/plots/"

    plt.rcParams.update({'font.size': 14})

    for i, eccentricity in enumerate(eccVals):
        # Reads data from saved csv files.
        importDir = valuesDir + f"orbit_avg_eccentricity{eccentricity}.csv"
        data = np.genfromtxt(importDir, delimiter=",")

        # Creates array of ratio to required power.
        alphaArr = data/powerReq
        alphaArr[alphaArr < 1.0] = 0
        alphaArr[alphaArr >= 1.0] = 1

        fig, ax = plt.subplots()

        # Checks whether we want to see uncompliant (<100% of 4W) results.
        if showUncompliant:
            im = ax.imshow(data / powerReq * 100)
        else:
            im = ax.imshow(data/powerReq * 100, vmin= 100, alpha= alphaArr)

        # Creates xticks for semiMajor. Assigns every third an empty space to make it more readable.
        yTicks = np.array(semiMajorVals*1e-3, dtype=str)       # km
        yTicks[1::2] = ""
        ax.set_yticks(range(len(yTicks)), labels=yTicks,
                      rotation=45, ha="right", rotation_mode="anchor")
        # Creates xticks for inclination. Assigns every third an empty space to make it more readable.
        xTicks = np.array(np.round(incVals, decimals=3), dtype=str)
        xTicks[1::2] = ""
        ax.set_xticks(range(len(xTicks)), labels=xTicks,
                      rotation=45, ha="right", rotation_mode="anchor")

        # Axis labels
        ax.set_xlabel(r"Inclination [º]")
        ax.set_ylabel(r"Semi Major Axis [km]")

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("% of 4W Average", rotation=-90, va="bottom")

        fig.suptitle(f"Eccentricity {eccentricity}")

        fig.tight_layout()

        fig.savefig(plotsDir + f"eccentricity{eccentricity}_orbitAvg.png")

def plot_battery_charge(
        dataDir: str,
        battMax: float,
        runCount: list,
):
    """

    Args:
        dataDir:
        battMax:
        runCount:

    Returns:

    """

    plt.rcParams.update({'font.size': 14})

    fig, ax = plt.subplots()

    for idx, run in enumerate(runCount):
        # Run directory addresses.
        runDir = dataDir + f"run_num_{run}/"
        propsDir = runDir + "propagation/"
        outputsDir = runDir + "outputs/"

        outputArr = np.loadtxt(outputsDir + 'battery_charge.txt', delimiter=',')
        dependentArr = np.loadtxt(propsDir + 'dependent_vals.txt', delimiter=',')

        times = (outputArr[:, 0] - outputArr[0, 0]) / 60  # Min

        ax.plot(times, dependentArr[:, 1] * battMax, ":k")
        ax.plot(times, outputArr[:, 1], plotStyles[idx], plotColors[idx],
                label=f"Run {run}")

    ax.grid()

    # Axis ticks.
    ax.set_ylim(bottom= -5.0, top= battMax + 5.0)

    # Axis labels
    ax.set_xlabel(r"Propagation Time [min]")
    ax.set_ylabel(r"Battery Charge [W*h]")

    fig.legend()

    fig.savefig(dataDir + f"battery_sim_combined.png")

    return None