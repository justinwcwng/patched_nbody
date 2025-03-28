import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from constants import R_earth, R_moon, soi_moon

from analytics import read_csvs

def plot_vx_vy(
        patched: bool = False
    ) -> None:

    """
    Plots simulated x and y velocity data over time.
    """

    filenames = ["patched_EarthMoon.csv"] if patched else ["patched_EarthMoon.csv", "nBody_EarthMoon.csv"]
    t, *_, vx, vy, vz, v = read_csvs(filenames)

    colours = ["red", "blue"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13,10))

    for i in range(len(filenames)):
        ax1.plot(t[i], vx[i], c=colours[i])
        ax2.plot(t[i], vy[i], c=colours[i])

    ax1.set_ylabel("x velocity (km/s)")
    ax2.set_ylabel("y velocity (km/s)")
    ax2.set_xlabel("time elapsed (s)")

    for ax in [ax1, ax2]:
        ax.legend(["patched", "nbody"])
        ax.grid()
    
    plt.savefig("plots/vx_vy_vs_t.png")

def plot_r_v_diff() -> None:

    """
    Plots the difference in simulated position and velocity data over time.
    """

    filename = ["percent_diff.csv"]
    t, _, _, _, r_diff, vx_diff, vy_diff, _, v_diff = read_csvs(filename)

    t = t[0]
    r_diff = r_diff[0]
    v_diff = v_diff[0]

    vx_diff = vx_diff[0]
    vy_diff = vy_diff[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13,12))

    ax1.plot(t, r_diff, c="grey")
    ax1.set_ylabel("difference in distance from Earth (km)")
    
    ax1.set_xlim([0, 500_000])
    #ax1.set_ylim([0, 0.005])

    ax2.plot(t, v_diff, c="orange")
    ax2.set_xlabel("time elapsed (s)")
    ax2.set_ylabel("difference in velocity (km/s)")

    ax1.set_xlim([0, 500_000])
    #ax2.set_ylim([0, 0.005])

    ax1.grid()
    ax2.grid()

    plt.savefig("plots/r_v_diff.png")

def plot_xy(
        patched: bool = False
    ) -> None:

    """
    Plots simulated position data in 2D space.
    """
    
    filenames = ["patched_EarthMoon.csv"] if patched else ["patched_EarthMoon.csv", "nBody_EarthMoon.csv"]
    t, x, y, z, r, vx, vy, vz, v = read_csvs(filenames)

    colours = ["yellow", "orange"]
    legend = ["Earth", "Moon's SOI", "Moon", "Patched Conic Approx", "n-Body Problem Solution"]

    fig, ax = plt.subplots(figsize=(13,10))
    ax.set_facecolor("#050505")

    # Earth Patch
    drawEarth = Circle((0, 0), R_earth, edgecolor="#50C878", facecolor="#89CFF0", linewidth=1)
    ax.add_patch(drawEarth)

    # Moon's SOI Patch
    drawMoonSOI = Circle((-384400, 0), soi_moon, edgecolor="#FF3333", facecolor="#490505", linewidth=1)
    ax.add_patch(drawMoonSOI)

    # Moon Patch
    drawMoon = Circle((-384400, 0), R_moon, edgecolor="#555555", facecolor="#333333", linewidth=1)
    ax.add_patch(drawMoon)

    # Trajectory Connection Point
    #with open("output/point_entered.txt", "r") as f:
    #    patch_point = int(f.read().strip()) - 1
    #x_patch, y_patch = x[0][patch_point], y[0][patch_point]
    #drawPatchPoint = Circle((x_patch, y_patch), 5000, edgecolor="#FFFFFF", facecolor="#000000", linewidth=3)
    #ax.add_patch(drawPatchPoint)

    # set x and y limits
    margin = 5e4
    xmin = min(x[0]) - margin
    xmax = max(x[0]) + margin
    ymin = min(y[0]) - margin
    ymax = max(y[0]) + margin

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    for i in range(len(filenames)):
        #ax.scatter(x[i], y[i], c=colours[i], s=1)
        ax.plot(x[i], y[i], c=colours[i], lw=1)
        plt.axis("equal")
        plt.legend(legend)
        plt.savefig("plots/xy.png")

def plot_r_v(
        patched: bool = False
    ) -> None:

    """
    Plots simulated position data on the top and velocity data on the bottom over time.
    """

    filenames = ["patched_EarthMoon.csv"] if patched else ["patched_EarthMoon.csv", "nBody_EarthMoon.csv"]
    t, x, y, z, r, vx, vy, vz, v = read_csvs(filenames)

    colours = ["red", "blue"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13,12))

    ax1.set_ylabel("distance from earth (km)")
    ax2.set_ylabel("velocity (km/s)")
    ax2.set_xlabel("time elapsed (s)")

    for i in range(len(filenames)):
        ax1.plot(t[i], r[i], c=colours[i])
        ax2.plot(t[i], v[i], c=colours[i])
    ax1.legend(["patched", "nbody"])
    ax1.grid()
    ax2.grid()

    plt.savefig("plots/r_v_vs_t.png")

def plot_acc() -> None:

    """
    Plots the simulated acceleration data over time.
    """
    
    nbody = pd.read_csv("output/nbody_EarthMoon.csv")

    t = nbody["t (s)"]
    #ax = nbody["ax (km/s2)"]
    #ay = nbody["ay (km/s2)"]
    #az = nbody["az (km/s2)"]
    a = nbody["a (km/s2)"]
    ae = nbody["ae (km/s2)"]
    am = nbody["am (km/s2)"]

    #ax = np.abs(1000 * ax)
    #ay = np.abs(1000 * ay)
    #az = np.abs(1000 * az)
    a = np.abs(1000 * a)
    ae = np.abs(1000 * ae)
    am = np.abs(1000 * am)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13,10))

    ax1.plot(t, a, c="red")
    ax2.plot(t, ae, c="blue")
    ax3.plot(t, am, c="grey")

    ax3.set_xlabel("time elapsed (s)")
    ax1.set_ylabel("total acceleration (m/s2)")
    ax2.set_ylabel("Earth acceleration (m/s2)")
    ax3.set_ylabel("Moon acceleration (m/s2)")

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim([0, np.max(t)])
        ax.set_xticks(np.arange(0, np.max(t)+1, 50000))
        ax.set_ylim([0, 0.1])
        ax.set_yticks(np.arange(0, 0.16, 0.05))
        ax.grid()

    plt.savefig("plots/acc_vs_t.png")

if __name__ == "__main__":
    #plot_xy(patched = False)
    #plot_r_v(patched = False)
    #plot_vx_vy()
    plot_r_v_diff()
    plot_acc()