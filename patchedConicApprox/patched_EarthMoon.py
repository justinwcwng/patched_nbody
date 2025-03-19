import numpy as np
import pandas as pd

import time
from tabulate import tabulate

from patchedConicApprox.keplerian import compute_a_e, elliptical, detect_soi_moon, orbital_elements, hyperbolic

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constants import mu_earth, mu_moon, r_LEO, r_moon, soi_moon

def patched_sim(
        rp: float = r_LEO,
        ra: float = r_moon,
        t_sim: float = 600_000
    ) -> float:

    """
    Main simulation function for the Patched Conic Approximation method.
    Currently trialling an Earth-Moon trajectory, patched at the edge of the Moon's sphere of influence.
    """

    ### START OF SIMULATION ###

    start_time = time.time()

    a, e = compute_a_e(ra, rp)
    t, x, y, z, r, vx, vy, vz, v = elliptical(mu = mu_earth,
                                              a = a,
                                              e = e,
                                              n_points = 1_000
                                             )

    entered_soi_moon = detect_soi_moon(soi_moon = soi_moon,
                                       x = x,
                                       y = y,
                                       z = z
                                       )
    x0 = x[entered_soi_moon]
    y0 = y[entered_soi_moon]
    z0 = z[entered_soi_moon]
    r0 = r[entered_soi_moon]
    vx0 = vx[entered_soi_moon]
    vy0 = vy[entered_soi_moon]
    vz0 = vz[entered_soi_moon]
    v0 = v[entered_soi_moon]

    with open("output/patchPoint.txt", "w") as f:
        f.write(str(entered_soi_moon)) # store index as str

    a_h, e_h, i, Omega, omega, theta_moon = orbital_elements(mu_moon,
                                                             np.array([x0]),
                                                             np.array([y0]),
                                                             np.array([z0]),
                                                             np.array([vx0]),
                                                             np.array([vy0]),
                                                             np.array([vz0])
                                                             )
    
    useful_data = pd.DataFrame({
        "Var": ["Patch Point Index", "x0", "y0", "z0", "r0", "vx0", "vy0", "vz0", "v0", "a_h", "e_h", "inclination", "longitude of asc node", "argument of periapsis", "true anomaly"],
        "Value": [entered_soi_moon, x0, y0, z0, r0, vx0, vy0, vz0, v0, a_h, e_h, i, Omega, omega, np.degrees(theta_moon)],
        "Unit": [None, "km", "km", "km", "km", "km/s", "km/s", "km/s", "km/s", "km", None, "deg", "rad", "rad", "deg"]
    })

    def printer():
        table = tabulate(useful_data,
                         headers=useful_data.columns,
                         tablefmt="fancy_grid",
                         numalign="center",
                         stralign="center",
                         colalign=("center", "center", "center"))
        print()
        print(table)

    printer()

    t1, x1, y1, z1, r1, v1 = hyperbolic(mu = mu_moon,
                                        a = a_h,
                                        e = e_h,
                                        theta = (2*np.pi-theta_moon),
                                        omega = omega,
                                        t_max = (t_sim - t[entered_soi_moon]),
                                        n_points = 1_000
                                        )

    end_time = time.time()
    duration = end_time - start_time

    ### END OF SIMULATION ###

    # create pre-patch dataframe
    df = pd.DataFrame({
        "t (s)": t,
        "x (km)": x,
        "y (km)": y,
        "z (km)": z,
        "r (km)": r,
        "vx (km/s)": vx,
        "vy (km/s)": vy,
        "vz (km/s)": vz,
        "v (km/s)": v
    })

    # create post-patch dataframe
    df1 = pd.DataFrame({
        "t (s)": t1 + t[entered_soi_moon],
        "x (km)": x1,
        "y (km)": y1,
        "z (km)": z1,
        "r (km)": r1,
        "vx (km/s)": np.zeros_like(x1),
        "vy (km/s)": np.zeros_like(x1),
        "vz (km/s)": np.zeros_like(x1),
        "v (km/s)": v1
    })

    # combine dataframes
    df_out = pd.concat([df[:entered_soi_moon], df1])

    print()
    print(f"patched conic simulation successful [{duration*1e3:5f} ms]")
    print()
    print("data shape:", df_out.shape)
    df_out.to_csv("output/patched_EarthMoon.csv")
    print()
    print("csv export completed")
    print()

    return duration

if __name__ == "__main__":
    patched_sim()