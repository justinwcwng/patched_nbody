import numpy as np
import pandas as pd
from tabulate import tabulate

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

    with open("output/patchPoint.txt", "r") as f:
        patch_point = int(f.read().strip()) - 1

    t_patch = patched["t (s)"][patch_point]

    patched_row = patched.iloc[(patched["t (s)"] - t_patch).abs().idxmin()]
    nbody_row = nbody.iloc[(nbody["t (s)"] - t_patch).abs().idxmin()]

    at_the_edge = pd.concat([patched_row, nbody_row], axis=1)

    at_the_edge["diff"] = nbody_row - patched_row

    print(tabulate(at_the_edge, tablefmt="rounded_grid"))

if __name__ == "__main__":
    process_and_export()
    patch_point_vectors()