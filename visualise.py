import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from constants import R_earth, R_moon, soi_moon

def read_csvs(
        filenames: list
    ) -> tuple[list]:

    """
    Reads and returns simulation data for plotting.
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

    # set x and y limits
    #xmin = -5
    #xmax = 5
    #ymin = 0
    #ymax = 12
    #ax.set_xlim([xmin, xmax])
    #ax.set_ylim([ymin, ymax])

    #ax.set_xticks(range(0, 458000, 40000))
    #ax.set_yticks(range(-2, 13, 1))

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

    fig, ax = plt.subplots(figsize=(16,9))
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

if __name__ == "__main__":
    #pass
    plot_v(patched = False)
    plot_xy(patched = False)