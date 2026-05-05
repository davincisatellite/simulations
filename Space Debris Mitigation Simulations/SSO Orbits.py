import numpy as np
import os
import csv

def sso_sma(i, e):
    J_2 = 0.00108263
    omega_dot = (360 / 365.242199) * (np.pi / 180) * (1 / 86400)
    a_e = 6378.1363 * 10**3
    mu_e = 398600.442 * 10**9
    i = i * np.pi / 180

    a = ((-3 / 2) * J_2 * (a_e / (1 - e ** 2)) ** 2 * (np.sqrt(mu_e) / omega_dot) * np.cos(i)) ** (2 / 7)
    a = a / 1000
    return a


if __name__ == "__main__":
    inclinations = list(np.arange(50, 100, 5))
    eccentricities = [0.0]

    data_dir = "output/data"
    sso_file = os.path.join(data_dir, "conceivable_sso.csv")
    file_exists = os.path.isfile(sso_file)

    with open(sso_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        if os.path.getsize(sso_file) == 0:
            writer.writerow([
                "Inclination [deg]",
                "Semi-major Axis [km]",
            ])
        for i in inclinations:
            e = eccentricities[0]
            a = sso_sma(i, e)
            writer.writerow([f"{i:.2f}", f"{a:.2f}"])

