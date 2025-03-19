import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from constants import R_earth, R_moon, soi_moon

from analytics import read_csvs

def plot_v(
        patched: bool = False
    ) -> None:

    """
    Plots simulated velocity data over time.
    """

    filenames = ["patched_EarthMoon.csv"] if patched else ["patched_EarthMoon.csv", "nBody_EarthMoon.csv"]
    t, x, y, z, r, vx, vy, vz, v = read_csvs(filenames)

    colours = ["red", "green"]
    legend = ["velocity magnitude", "x velocity", "y velocity"]

    fig, ax = plt.subplots(figsize=(15,8))

    ax.set_xlabel("time elapsed (s)")
    ax.set_ylabel("velocity (km/s)")

    for i in range(len(filenames)):
        ax.plot(t[i], v[i], c=colours[i])
        #ax.plot(t[i], vx[i])
        #ax.plot(t[i], vy[i])
    ax.legend(["patched", "nBody"])
    #ax.legend(legend)
    ax.grid()
    plt.savefig("plots/current_v.png")

def plot_r_v_diff() -> None:

    """
    Plots the difference in simulated position and velocity data over time.
    """

    filename = ["diff.csv"]
    t, _, _, _, r_diff, _, _, _, v_diff = read_csvs(filename)

    t = t[0]
    r_diff = r_diff[0]
    v_diff = v_diff[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13,12))

    ax1.plot(t, r_diff, c="grey")
    ax1.set_ylabel("difference in distance from earth (km)")
    
    ax1.set_ylim([0, 6000])

    ax2.plot(t, v_diff, c="orange")
    ax2.set_xlabel("time elapsed (s)")
    ax2.set_ylabel("difference in velocity (km/s)")

    ax2.set_ylim([0, 0.2])

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
    legend = ["Earth", "Moon's SOI", "Moon", "Patch Point", "Patched Conic Approx", "n-Body Problem Solution"]

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
    with open("output/patchPoint.txt", "r") as f:
        patch_point = int(f.read().strip()) - 1
    x_patch, y_patch = x[0][patch_point], y[0][patch_point]
    drawPatchPoint = Circle((x_patch, y_patch), 5000, edgecolor="#FFFFFF", facecolor="#000000", linewidth=3)
    ax.add_patch(drawPatchPoint)

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
        plt.savefig("plots/current_xy.png")

def plot_r(
        patched: bool = False
    ) -> None:

    """
    Plots simulated position data over time.
    """

    filenames = ["patched_EarthMoon.csv"] if patched else ["patched_EarthMoon.csv", "nBody_EarthMoon.csv"]
    t, x, y, z, r, vx, vy, vz, v = read_csvs(filenames)

    colours = ["red", "green"]
    legend = ["Distance from Earth"]

    fig, ax = plt.subplots(figsize=(15,8))

    ax.set_xlabel("time elapsed (s)")
    ax.set_ylabel("position r (km)")

    for i in range(len(filenames)):
        ax.plot(t[i], r[i], c=colours[i])
    ax.legend(["patched", "nBody"])
    #ax.legend(legend)
    ax.grid()
    plt.savefig("plots/current_r.png")

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
    ax1.legend(["patched", "nBody"])
    ax1.grid()
    ax2.grid()

    plt.savefig("plots/current_r_v.png")

if __name__ == "__main__":
    #pass
    #plot_xy(patched = False)
    plot_r_v(patched = False)
    plot_r_v_diff()