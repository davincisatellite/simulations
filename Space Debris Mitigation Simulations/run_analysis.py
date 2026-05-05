from datetime import datetime

import numpy as np

from lifetime_functions import run_lifetime_analysis, plot_results
import numpy

# ===== INPUTS =====
output_files = {
    "latest": "lifetime_latest_prediction.csv",
    "ecss": "lifetime_ecss.csv",
    "montecarlo": "lifetime_montecarlo.csv"
}
summary_output = "lifetime_summary.csv"

semi_major_axes = [6895.0] #6830, 6840, 6850, 6860, 6870, 6880
eccentricities = [0.0] # 0.015
inclinations = [60.0] # Power sim constraint - 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0

start_date = datetime(2027, 1, 1)
end_date = datetime(2038, 1, 1)
step_days = 366//2

spacecraft_area = 0.02725090
spacecraft_mass = 2.2
drag_coefficient = 2.2
reflectivity_coefficient = 1.2

montecarlo_cycles = 5

# ===== CONTROL FLAGS =====
RUN_ANALYSIS = True     # Run OSCAR simulation
PLOT_RESULTS = True      # Generate plots

# ===== MAIN =====
if __name__ == "__main__":
    if RUN_ANALYSIS:
        run_lifetime_analysis(
            semi_major_axes=semi_major_axes,
            eccentricities=eccentricities,
            inclinations=inclinations,
            start_date=start_date,
            end_date=end_date,
            step_days=step_days,
            output_files=output_files,
            summary_output=summary_output,
            spacecraft_area=spacecraft_area,
            spacecraft_mass=spacecraft_mass,
            drag_coefficient=drag_coefficient,
            reflectivity_coefficient=reflectivity_coefficient,
            montecarlo_cycles=montecarlo_cycles
        )

    if PLOT_RESULTS:
        plot_results(lifetime_vs_epoch=False, lifetime_heatmap=True)
