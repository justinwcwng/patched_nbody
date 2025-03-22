import numpy as np
import pandas as pd

import time
from tabulate import tabulate

from patchedConicApprox.keplerian import compute_a_e, elliptical, orbital_elements, hyperbolic
from analytics import detect_soi_moon

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constants import mu_earth, mu_moon, r_LEO, r_moon, soi_moon
from celestial_xyz import earth_xyz, moon_xyz

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

    print()
    print("- "*27)
    print()
    print("patched conic simulation initiated")

    start_time = time.time()

    a, e = compute_a_e(ra, rp)
    t, x, y, z, r, vx, vy, vz, v = elliptical(mu = mu_earth,
                                              a = a,
                                              e = e,
                                              theta0 = 0,
                                              omega = 0,
                                              t_max = (np.pi * np.sqrt(a**3 / mu_earth)),
                                              n_points = 1_000
                                              )

    entered_soi_moon = detect_soi_moon(soi_moon = soi_moon,
                                       x = x,
                                       y = y,
                                       z = z
                                       )
    t_entered = t[entered_soi_moon]
    x_entered = x[entered_soi_moon]
    y_entered = y[entered_soi_moon]
    z_entered = z[entered_soi_moon]
    r_entered = r[entered_soi_moon]
    vx_entered = vx[entered_soi_moon]
    vy_entered = vy[entered_soi_moon]
    vz_entered = vz[entered_soi_moon]
    v_entered = v[entered_soi_moon]

    with open("output/point_entered.txt", "w") as f:
        f.write(str(entered_soi_moon)) # store index as str

    a_h, e_h, _, _, omega, theta_moon, pre_patch_elements = orbital_elements(mu_moon,
                                                                             np.array([x_entered]),
                                                                             np.array([y_entered]),
                                                                             np.array([z_entered]),
                                                                             np.array([vx_entered]),
                                                                             np.array([vy_entered]),
                                                                             np.array([vz_entered]),
                                                                             moon_xyz
                                                                             )

    print()
    print("   -   PATCH POINT 1   -")
    print()
    print(pre_patch_elements)

    t1, x1, y1, z1, r1, vx1, vy1, vz1, v1 = hyperbolic(mu = mu_moon,
                                                       a = a_h,
                                                       e = e_h,
                                                       theta0 = (2*np.pi-theta_moon),
                                                       omega = omega,
                                                       t_max = (t_sim - t_entered),
                                                       n_points = 5_000
                                                       )
    t1 += t_entered

    exited_soi_moon = detect_soi_moon(soi_moon = soi_moon,
                                      x = x1,
                                      y = y1,
                                      z = z1
                                      )
    t_exited = t1[exited_soi_moon]
    x_exited = x1[exited_soi_moon]
    y_exited = y1[exited_soi_moon]
    z_exited = z1[exited_soi_moon]
    r_exited = r1[exited_soi_moon]
    vx_exited = vx1[exited_soi_moon]
    vy_exited = vy1[exited_soi_moon]
    vz_exited = vz1[exited_soi_moon]
    v_exited = v1[exited_soi_moon]

    with open("output/point_exited.txt", "w") as f:
        f.write(str(exited_soi_moon)) # store index as str

    a1, e1, _, _, omega1, theta1, post_patch_elements = orbital_elements(mu_earth,
                                                                         np.array([x_exited]),
                                                                         np.array([y_exited]),
                                                                         np.array([z_exited]),
                                                                         np.array([vx_exited]),
                                                                         np.array([vy_exited]),
                                                                         np.array([vz_exited]),
                                                                         earth_xyz
                                                                         )

    print()
    print("   -   PATCH POINT 2   -")
    print()
    print(post_patch_elements)

    t2, x2, y2, z2, r2, vx2, vy2, vz2, v2 = elliptical(mu = mu_earth,
                                                       a = a1,
                                                       e = e1,
                                                       theta0 = theta1,
                                                       omega = omega1,
                                                       t_max = (t_sim - t_exited),
                                                       n_points = 500
                                                       )
    t2 += t_exited

    end_time = time.time()
    duration = end_time - start_time

    ### END OF SIMULATION ###

    # create pre-patch1 dataframe
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

    # create post-patch1 dataframe
    df1 = pd.DataFrame({
        "t (s)": t1,
        "x (km)": x1,
        "y (km)": y1,
        "z (km)": z1,
        "r (km)": r1,
        "vx (km/s)": vx1,
        "vy (km/s)": vy1,
        "vz (km/s)": vz1,
        "v (km/s)": v1
    })

    # create post-patch2 dataframe
    df2 = pd.DataFrame({
        "t (s)": t2,
        "x (km)": x2,
        "y (km)": y2,
        "z (km)": z2,
        "r (km)": r2,
        "vx (km/s)": vx2,
        "vy (km/s)": vy2,
        "vz (km/s)": vz2,
        "v (km/s)": v2
    })

    # combine dataframes
    df_out = pd.concat([df.iloc[:entered_soi_moon], df1.iloc[:exited_soi_moon], df2], axis=0)

    print()
    print("- "*27)
    print()
    print(f"patched conic simulation successful [{duration*1e3:5f} ms]")
    print()
    print("data shape:", df_out.shape)
    df_out.to_csv("output/patched_EarthMoon.csv")
    print()
    print("csv export completed")
    print()
    print("="*53)

    return duration

if __name__ == "__main__":
    patched_sim()