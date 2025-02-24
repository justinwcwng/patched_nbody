import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D

from constants import R_earth, R_moon, SOI_moon

filenames = ["nBody_EarthMoon.csv"] #"patched_EarthMoon.csv"
colours = ["yellow", "green"]
legend = ["Earth", "Moon's SOI", "Moon", "Patched Conic Approx", "n-Body Problem Solution"]

x = []
y = []
z = []

for filename in filenames:
    df = pd.read_csv(f"output/{filename}")
    x += [df["x (km)"]]
    y += [df["y (km)"]]
    z += [df["z (km)"]]

fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111, projection="3d")
#ax.set_facecolor("#050505")

# Earth sphere
phi, theta = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
xEarth = R_earth * np.sin(theta) * np.cos(phi)
yEarth = R_earth * np.sin(theta) * np.sin(phi)
zEarth = R_earth * np.cos(theta)

# set x and y limits
margin = 5e4
xmin = min(x[0]) - margin
xmax = max(x[0]) + margin
ymin = min(y[0]) - margin
ymax = max(y[0]) + margin
zmin = min(z[0]) - margin
zmax = max(z[0]) + margin

ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

# plot

def ani():
    for i in range(len(x)):
        ax.scatter(x[i], y[i], c="yellow", s=2)
        plt.draw()
        plt.pause(0.05)
    plt.axis("equal")
    plt.show()

def static():
    # plot Earth sphere
    ax.plot_surface(xEarth, yEarth, zEarth, color="#89CFF0")
    for i in range(len(filenames)):
        ax.plot(x[i], y[i], z[i], c=colours[i], lw=1)
    plt.axis("equal")
    #plt.legend(legend)
    plt.show()
    #plt.savefig("current_viz_3d.png")

static()