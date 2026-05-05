import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import csv
import os
from datetime import timedelta
from statistics import median
from collections import defaultdict
from drama import oscar

def run_lifetime_analysis(
    semi_major_axes,
    eccentricities,
    inclinations,
    start_date,
    end_date,
    step_days,
    output_files,
    summary_output,
    spacecraft_area,
    spacecraft_mass,
    drag_coefficient,
    reflectivity_coefficient,
    montecarlo_cycles
):
    # === Ensure output/data directory exists ===
    data_dir = os.path.join("output", "data")
    os.makedirs(data_dir, exist_ok=True)

    # Prepend output/data/ to filenames
    output_files = {
        key: os.path.join(data_dir, os.path.basename(path))
        for key, path in output_files.items()
    }
    summary_output = os.path.join(data_dir, os.path.basename(summary_output))

    # === Define solar activity scenarios ===
    scenarios = {
        "montecarlo": {
            "solarAndGeomagneticActivityScenario": 5,
            "monteCarloSamplingCycles": montecarlo_cycles},
        "latest": {"solarAndGeomagneticActivityScenario": 1},
        "ecss": {"solarAndGeomagneticActivityScenario": 4},
    }

    # === Build list of begin dates ===
    begin_dates = []
    date = start_date
    while date < end_date:
        begin_dates.append(date)
        date += timedelta(days=step_days)

    data = {key: [] for key in scenarios}

    # === Run OSCAR for each solar scenario ===
    for scenario_key, scenario_config in scenarios.items():
        try:
            config = {
                "semiMajorAxis": semi_major_axes,
                "eccentricity": eccentricities,
                "inclination": inclinations,
                "rightAscensionOfTheAscendingNode": 20,
                "argumentOfPerigee": 0,
                "meanAnomaly": 0,
                "spacecraftCrossSectionArea": spacecraft_area,
                "spacecraftMass": spacecraft_mass,
                "dragCoefficient": drag_coefficient,
                "reflectivityCoefficient": reflectivity_coefficient,
                "beginDate": begin_dates,
            }
            config.update(scenario_config)

            print(
                f"Running {scenario_key} scenario with "
                f"{len(semi_major_axes)} SMA × {len(eccentricities)} ecc × "
                f"{len(inclinations)} inc × {len(begin_dates)} dates = "
                f"{len(semi_major_axes)*len(eccentricities)*len(inclinations)*len(begin_dates)} cases..."
            )

            results = oscar.run(config)

            if not results or "results" not in results:
                print(f"No results returned for {scenario_key}")
                continue

            total = len(results["results"])
            for idx, res in enumerate(results["results"], start=1):
                if not isinstance(res, dict):
                    continue
                if "lifetime" not in res or "config" not in res:
                    continue

                lifetime = res["lifetime"]
                cfg = res["config"]
                sma = cfg["semiMajorAxis"]
                ecc = cfg["eccentricity"]
                inc = cfg["inclination"]
                bdate = cfg["beginDate"]

                data[scenario_key].append((bdate, sma, ecc, inc, lifetime))

                print(
                    f"[{idx}/{total}] {scenario_key} | SMA={sma} km | Ecc={ecc} | "
                    f"Inc={inc}° | BeginDate={bdate} | Lifetime={lifetime:.2f} yrs"
                )

        except Exception as e:
            print(f"Error for {scenario_key}: {e}")

    # === Write per-scenario CSVs (duplicate-safe append) ===
    for scenario_key, values in data.items():
        if not values:
            continue

        file_exists = os.path.isfile(output_files[scenario_key])
        existing_combos = set()

        # Load existing combos if file already exists
        if file_exists and os.path.getsize(output_files[scenario_key]) > 0:
            with open(output_files[scenario_key], newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    combo = (
                        row["Date"],
                        float(row["SemiMajorAxis_km"]),
                        float(row["Eccentricity"]),
                        float(row["Inclination_deg"])
                    )
                    existing_combos.add(combo)

        # Append new unique rows
        with open(output_files[scenario_key], mode='a', newline='') as file:
            writer = csv.writer(file)

            if not file_exists or os.path.getsize(output_files[scenario_key]) == 0:
                writer.writerow([
                    "Date",
                    "SemiMajorAxis_km",
                    "Eccentricity",
                    "Inclination_deg",
                    "Lifetime_yrs"
                ])

            for (bdate, sma, ecc, inc, lifetime) in values:
                combo = (bdate.strftime("%Y-%m-%d"), sma, ecc, inc)
                if combo in existing_combos:
                    print(f"combo ({scenario_key}, {sma}, {ecc}, {inc}, {bdate.date()}) already written → skipped")
                    continue

                writer.writerow([
                    bdate.strftime("%Y-%m-%d"),
                    sma,
                    ecc,
                    inc,
                    f"{lifetime:.3f}"
                ])
                existing_combos.add(combo)

    # === Compute grouped medians ===
    grouped = defaultdict(list)
    for scenario_key, values in data.items():
        for date, sma, ecc, inc, lifetime in values:
            grouped[(scenario_key, sma, ecc, inc)].append(lifetime)

    # === Write summary CSV (duplicate-safe append) ===
    file_exists = os.path.isfile(summary_output)
    existing_summary = set()

    # Load existing summary combos
    if file_exists and os.path.getsize(summary_output) > 0:
        with open(summary_output, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                combo = (
                    row["Scenario"],
                    float(row["SemiMajorAxis_km"]),
                    float(row["Eccentricity"]),
                    float(row["Inclination_deg"])
                )
                existing_summary.add(combo)

    with open(summary_output, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header if file is new or empty
        if not file_exists or os.path.getsize(summary_output) == 0:
            writer.writerow([
                "Scenario",
                "SemiMajorAxis_km",
                "Eccentricity",
                "Inclination_deg",
                "MedianLifetime_yrs",
                "MinLifetime_yrs",
                "MaxLifetime_yrs"
            ])

        for (scenario_key, sma, ecc, inc), lifetimes in grouped.items():
            combo = (scenario_key, sma, ecc, inc)
            if combo in existing_summary:
                print(f"summary combo ({scenario_key}, {sma}, {ecc}, {inc}) already written → skipped")
                continue

            writer.writerow([
                scenario_key,
                sma,
                ecc,
                inc,
                f"{median(lifetimes):.3f}",
                f"{min(lifetimes):.3f}",
                f"{max(lifetimes):.3f}"
            ])
            existing_summary.add(combo)




def plot_results(lifetime_vs_epoch=False, lifetime_heatmap=False):
    # === Define directories ===
    data_dir = "output/data"
    output_dir_epoch = "output/figures/lifetime_vs_epoch"
    output_dir_heatmap = "output/figures/lifetime_heatmaps"
    os.makedirs(output_dir_epoch, exist_ok=True)
    os.makedirs(output_dir_heatmap, exist_ok=True)

    # === File paths ===
    files = {
        "latest": os.path.join(data_dir, "lifetime_latest_prediction.csv"),
        "ecss": os.path.join(data_dir, "lifetime_ecss.csv"),
        "montecarlo": os.path.join(data_dir, "lifetime_montecarlo.csv")
    }
    summary_file = os.path.join(data_dir, "lifetime_summary.csv")

    # === Color settings ===
    scenario_colors = {
        "latest": "#1f77b4",      # blue
        "ecss": "#ff7f0e",        # orange
        "montecarlo": "#2ca02c"   # green
    }

    # ----------------------------------------------------------------------
    # 1️⃣ LIFETIME VS EPOCH PLOTS
    # ----------------------------------------------------------------------
    if lifetime_vs_epoch:
        print("Generating lifetime vs epoch plots...")

        data = {}
        for scenario, path in files.items():
            if not os.path.isfile(path):
                print(f"Warning: {path} not found, skipping {scenario}")
                continue
            df = pd.read_csv(path, parse_dates=["Date"])
            data[scenario] = df

        if not os.path.isfile(summary_file):
            print(f"Warning: summary file not found at {summary_file}")
            return
        summary = pd.read_csv(summary_file)

        combos = sorted({
            (float(row["SemiMajorAxis_km"]),
             float(row["Eccentricity"]),
             float(row["Inclination_deg"]))
            for df in data.values() for _, row in df.iterrows()
        })

        for (sma, ecc, inc) in combos:
            plt.figure(figsize=(8, 5))
            plt.title(f"Lifetime vs Epoch\nSMA={sma:.0f} km | Ecc={ecc} | Inc={inc}°")

            for scenario, df in data.items():
                subset = df[
                    (df["SemiMajorAxis_km"] == sma) &
                    (df["Eccentricity"] == ecc) &
                    (df["Inclination_deg"] == inc)
                ].sort_values("Date")

                if subset.empty:
                    continue

                color = scenario_colors.get(scenario, None)

                # Solid lifetime curve
                plt.plot(
                    subset["Date"], subset["Lifetime_yrs"],
                    label=f"{scenario} (data)",
                    color=color,
                    linewidth=1.8
                )

                # Dashed median line with same color
                median_row = summary[
                    (summary["Scenario"] == scenario) &
                    (summary["SemiMajorAxis_km"] == sma) &
                    (summary["Eccentricity"] == ecc) &
                    (summary["Inclination_deg"] == inc)
                ]
                if not median_row.empty:
                    med_val = median_row["MedianLifetime_yrs"].values[0]
                    plt.axhline(
                        med_val, linestyle="--", alpha=0.8,
                        color=color,
                        linewidth=1.5,
                        label=f"{scenario} median = {med_val:.2f} yr"
                    )

            plt.xlabel("Epoch (Start Date)")
            plt.ylabel("Orbital Lifetime [years]")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            plt.tight_layout()

            filename = f"SMA{sma:.0f}km_ECC{ecc}_INC{inc}deg.png".replace(".", "p")
            save_path = os.path.join(output_dir_epoch, filename)
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Saved: {save_path}")

        print("All lifetime vs epoch plots generated successfully.")

    # ----------------------------------------------------------------------
    # 2️⃣ HEATMAPS: MEDIAN LIFETIME VS SMA & ECC (per scenario, per inclination)
    # ----------------------------------------------------------------------
    # if lifetime_heatmap:
    #     print("Generating median lifetime heatmaps...")
    #
    #     if not os.path.isfile(summary_file):
    #         print(f"Summary file not found at {summary_file}")
    #         return
    #     summary = pd.read_csv(summary_file)
    #
    #     scenarios = summary["Scenario"].unique()
    #     inclinations = sorted(summary["Inclination_deg"].unique())
    #
    #     # === Custom colormap ===
    #     colors = [
    #         (0.0, "#2ca02c"),  # green
    #         (0.7, "#ffff00"),  # yellow
    #         (1.0, "#ff0000")   # red
    #     ]
    #     custom_cmap = LinearSegmentedColormap.from_list("lifetime_cmap", colors)
    #     custom_cmap.set_under('black')
    #     custom_cmap.set_over('black')
    #
    #     # === Normalization (clip everything above 5 years) ===
    #     vmin = 2.0
    #     vmax = 5.0
    #     norm = Normalize(vmin=vmin, vmax=vmax, clip=False)
    #
    #     for i, scenario in enumerate(scenarios, start=1):
    #         print(f"[{i}/{len(scenarios)}] Processing scenario: {scenario}")
    #
    #         fig, axes = plt.subplots(
    #             1, len(inclinations),
    #             figsize=(5 * len(inclinations), 5),
    #             constrained_layout=True
    #         )
    #
    #         if len(inclinations) == 1:
    #             axes = [axes]
    #
    #         for ax, inc in zip(axes, inclinations):
    #             subset = summary[
    #                 (summary["Scenario"] == scenario) &
    #                 (summary["Inclination_deg"] == inc)
    #             ]
    #             if subset.empty:
    #                 continue
    #
    #             pivot = subset.pivot_table(
    #                 index="SemiMajorAxis_km",
    #                 columns="Eccentricity",
    #                 values="MedianLifetime_yrs"
    #             ).sort_index(ascending=True)
    #
    #             im = ax.imshow(
    #                 pivot,
    #                 origin="lower",
    #                 cmap=custom_cmap,
    #                 aspect="auto",
    #                 norm=norm
    #             )
    #
    #             ax.set_title(f"Inc = {inc}°")
    #             ax.set_xlabel("Eccentricity")
    #             ax.set_ylabel("Semi-major Axis [km]")
    #
    #             ax.set_xticks(np.arange(len(pivot.columns)))
    #             ax.set_xticklabels([f"{c:.3f}" for c in pivot.columns])
    #             ax.set_yticks(np.arange(len(pivot.index)))
    #             ax.set_yticklabels([f"{r:.0f}" for r in pivot.index])
    #
    #         # === Shared colorbar ===
    #         cbar = fig.colorbar(im, ax=axes, shrink=0.9, extend="both")
    #         cbar.set_label("Median Orbital Lifetime [years]")
    #
    #         fig.suptitle(f"Median Lifetime Heatmap – {scenario.upper()} Model", fontsize=14)
    #
    #         save_path = os.path.join(output_dir_heatmap, f"heatmap_{scenario}.png")
    #         plt.savefig(save_path, dpi=300)
    #         plt.close()
    #         print(f"  → Saved: {save_path}")
    #
    #     print("All heatmaps generated successfully.")


    if lifetime_heatmap:
        print("Generating median lifetime heatmaps...")

        if not os.path.isfile(summary_file):
            print(f"Summary file not found at {summary_file}")
            return
        summary = pd.read_csv(summary_file)

        inclinations = sorted(summary["Inclination_deg"].unique())

        # === Custom colormap ===
        colors = [
            (0.0, "#2ca02c"),  # green
            (0.7, "#ffff00"),  # yellow
            (1.0, "#ff0000")   # red
        ]
        custom_cmap = LinearSegmentedColormap.from_list("lifetime_cmap", colors)
        custom_cmap.set_under('white')
        custom_cmap.set_over('white')

        # === Normalization (clip everything above 5 years) ===
        vmin = 2.0
        vmax = 5.0
        norm = Normalize(vmin=vmin, vmax=vmax, clip=False)

        n_rows = 4
        n_cols = int(np.ceil(len(inclinations) / n_rows))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5 * n_cols, 4 * n_rows),
            constrained_layout=True
        )

        if len(inclinations) == 1:
            axes = [axes]

        axes = axes.flatten()

        def conditional_min_max(x, threshold):
            return x.min() if x.mean() < threshold else x.max()

        for ax, inc in zip(axes, inclinations):
            subset = summary[(summary["Inclination_deg"] == inc)]

            if subset.empty:
                ax.axis("off")
                continue

            pivot = subset.pivot_table(
                index="SemiMajorAxis_km",
                columns="Eccentricity",
                values="MedianLifetime_yrs",
                aggfunc=lambda x: conditional_min_max(x, threshold=2.3)
            ).sort_index(ascending=True)

            im = ax.imshow(
                pivot,
                origin="lower",
                cmap=custom_cmap,
                aspect="auto",
                norm=norm
            )

            ax.set_title(f"Inc = {inc}°")
            ax.set_xlabel("Eccentricity")
            ax.set_ylabel("Semi-major Axis [km]")

            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([f"{c:.3f}" for c in pivot.columns])
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels([f"{r:.0f}" for r in pivot.index])

        for ax in axes[len(inclinations):]:
            ax.axis("off")

        # === Shared colorbar ===
        cbar = fig.colorbar(im, ax=axes, shrink=0.9, extend="both")
        cbar.set_label("Median Orbital Lifetime [years]")

        # fig.suptitle(f"Median Lifetime Heatmap", fontsize=14)

        save_path = os.path.join(output_dir_heatmap, f"heatmap.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  → Saved: {save_path}")

        print("All heatmaps generated successfully.")