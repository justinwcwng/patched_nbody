import numpy as np
import pandas as pd
from tabulate import tabulate

from celestial_xyz import moon_xyz
from constants import soi_moon

def detect_soi_moon(
        soi_moon: float,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> int:

    """
    Returns the index of the position-time vector when the spacecraft first enters or exits the Moon's sphere of influence.
    """

    distanceFromMoon = np.sqrt((x - moon_xyz[0])**2 + (y - moon_xyz[1])**2 + (z - moon_xyz[2])**2)
    soi_bool = (distanceFromMoon < soi_moon)

    # ONLY INCLUDE IF YOU WANT PATCH POINT 2
    if 1 == 0:
        soi_bool = np.array(soi_bool)
        i = 500_000
        soi_bool = soi_bool[i:-1]

    switch_index = np.where(soi_bool != soi_bool[0])[0][0]# if soi_bool[0] != soi_bool[-1] else -1

    return switch_index

def read_csvs(
        filenames: list
    ) -> tuple[list]:

    """
    Reads and returns simulation data for further processing or plotting.
    File(s) must reside in the output folder.
    """

    t = []
    x = []
    y = []
    z = []
    r = []
    vx = []
    vy = []
    vz = []
    v = []

    for filename in filenames:
        df = pd.read_csv(f"output/{filename}")
        t += [df["t (s)"]]
        x += [df["x (km)"]]
        y += [df["y (km)"]]
        z += [df["z (km)"]]
        r += [df["r (km)"]]
        vx += [df["vx (km/s)"]]
        vy += [df["vy (km/s)"]]
        vz += [df["vz (km/s)"]]
        v += [df["v (km/s)"]]

    return t, x, y, z, r, vx, vy, vz, v

def process_and_export() -> None:

    """
    Matches the shapes of the patched and nbody output DataFrames, and calculates the differences in simulated data over time.
    """

    filenames = ["patched_EarthMoon.csv", "nBody_EarthMoon.csv"]
    t, x, y, z, r, vx, vy, vz, v = read_csvs(filenames)

    x_nbody_interp = np.interp(t[0], t[1], x[1])
    y_nbody_interp = np.interp(t[0], t[1], y[1])
    z_nbody_interp = np.interp(t[0], t[1], z[1])
    r_nbody_interp = np.interp(t[0], t[1], r[1])
    vx_nbody_interp = np.interp(t[0], t[1], vx[1])
    vy_nbody_interp = np.interp(t[0], t[1], vy[1])
    vz_nbody_interp = np.interp(t[0], t[1], vz[1])
    v_nbody_interp = np.interp(t[0], t[1], v[1])
    
    x_diff = np.abs(x_nbody_interp - x[0])
    y_diff = np.abs(y_nbody_interp - y[0])
    z_diff = np.abs(z_nbody_interp - z[0])
    r_diff = np.abs(r_nbody_interp - r[0])
    vx_diff = np.abs(vx_nbody_interp - vx[0])
    vy_diff = np.abs(vy_nbody_interp - vy[0])
    vz_diff = np.abs(vz_nbody_interp - vz[0])
    v_diff = np.abs(v_nbody_interp - v[0])

    diff = pd.DataFrame({
        "t (s)": t[0],
        "x (km)": x_diff,
        "y (km)": y_diff,
        "z (km)": z_diff,
        "r (km)": r_diff,
        "vx (km/s)": vx_diff,
        "vy (km/s)": vy_diff,
        "vz (km/s)": vz_diff,
        "v (km/s)": v_diff
    })

    x_percent_diff = np.abs((x_nbody_interp - x[0]) / x[0])
    y_percent_diff = np.abs((y_nbody_interp - y[0]) / y[0])
    z_percent_diff = np.abs((z_nbody_interp - z[0]) / z[0])
    r_percent_diff = np.abs((r_nbody_interp - r[0]) / r[0])
    vx_percent_diff = np.abs((vx_nbody_interp - vx[0]) / vx[0])
    vy_percent_diff = np.abs((vy_nbody_interp - vy[0]) / vy[0])
    vz_percent_diff = np.abs((vz_nbody_interp - vz[0]) / vz[0])
    v_percent_diff = np.abs((v_nbody_interp - v[0]) / v[0])

    percent_diff = pd.DataFrame({
        "t (s)": t[0],
        "x (km)": x_percent_diff,
        "y (km)": y_percent_diff,
        "z (km)": z_percent_diff,
        "r (km)": r_percent_diff,
        "vx (km/s)": vx_percent_diff,
        "vy (km/s)": vy_percent_diff,
        "vz (km/s)": vz_percent_diff,
        "v (km/s)": v_percent_diff
    })

    diff.to_csv("output/diff.csv")
    percent_diff.to_csv("output/percent_diff.csv")

def patch_point_vectors():

    patched = pd.read_csv("output/patched_EarthMoon.csv")
    nbody = pd.read_csv("output/nbody_EarthMoon.csv")

    # r and v vectors from patched
    with open("output/point_entered.txt", "r") as f:
        patch_point = int(f.read().strip()) - 1
    patched_row = patched.iloc[patch_point]

    # r and v vectors from nbody
    entered_soi_nbody = detect_soi_moon(soi_moon = soi_moon,
                                        x = nbody["x (km)"],
                                        y = nbody["y (km)"],
                                        z = nbody["z (km)"])
    nbody_row = nbody.iloc[entered_soi_nbody]

    # concatenate & compare
    at_the_edge = pd.concat([patched_row, nbody_row], axis=1)
    at_the_edge["diff"] = nbody_row - patched_row
    at_the_edge["percent_diff"] = 100 * (nbody_row - patched_row) / patched_row
    at_the_edge = at_the_edge.iloc[1:-12]

    table = tabulate(at_the_edge,
                     headers=["quantity", "patched", "nbody", "diff", "%diff"],
                     tablefmt="rounded_grid"
                     )

    print(table)

if __name__ == "__main__":
    #process_and_export()
    patch_point_vectors()