import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

df = pd.read_csv("data/as16_pancam_state_vectors.csv")

df["date"] = df["utc_time_str"].str[:10]
df["time"] = df["utc_time_str"].str[11:]

df["date"] = pd.to_datetime(df["date"])
df["time"] = pd.to_datetime(df["time"])

df["datetime"] = pd.to_datetime(df["utc_time_str"], errors="coerce")
df.sort_values(by="datetime", inplace=True)

df = df[df["date"] == pd.Timestamp("1972-04-21")]

df2 = df[["datetime", "date", "time", "x1950_x", "x1950_y", "x1950_z", "x1950_xdot", "x1950_ydot", "x1950_zdot"]]

date_counts = df2["date"].value_counts()
print(date_counts)

df2.to_csv("data/current_data.csv")

dt = df2["datetime"]
d = df2["date"]
t = df2["time"]
x = df2["x1950_x"]
y = df2["x1950_y"]
z = df2["x1950_z"]
r = np.sqrt(x**2 + y**2 + z**2)
vx = df2["x1950_xdot"]
vy = df2["x1950_ydot"]
vz = df2["x1950_zdot"]
v = np.sqrt(vx**2 + vy**2 + vz**2)

def plot_xyz():

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #for i in range(len(x)):
    #    ax.scatter(x[i], y[i], z[i], c="black", s=2)
    #    plt.draw()
    #    plt.pause(0.1)

    ax.scatter(x, y, z, marker='o', alpha=1, linewidths=0, edgecolors=None, s=2)
    plt.axis("equal")
    plt.savefig("plots/as16.png")

def plot_r_v():

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(t, r)
    ax1.set_ylabel("distance from Earth")
    ax1.grid()

    ax2.plot(t, v)
    ax2.set_ylabel("velocity")
    ax2.set_xlabel("time (UTC)")
    ax2.grid()

    plt.savefig("plots/as16_r_v_over_t")

def animator():
    # Create a figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set up the line (empty for now) and the point (for trajectory)
    line, = ax.plot([], [], [], color='b', label='Trajectory')
    point, = ax.plot([], [], [], 'ro')  # Red point for the current position

    # Setting up axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set up the limit for the axes
    ax.set_xlim([df['x1950_x'].min(), df['x1950_x'].max()])
    ax.set_ylim([df['x1950_y'].min(), df['x1950_y'].max()])
    ax.set_zlim([df['x1950_z'].min(), df['x1950_z'].max()])

    # Function to initialize the animation (sets the background)
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return line, point

    # Function to animate the trajectory
    def update(frame):
        # Get the current time step's x, y, z values
        x_data = df['x1950_x'][:frame]
        y_data = df['x1950_y'][:frame]
        z_data = df['x1950_z'][:frame]
        
        # Update the trajectory line and the point
        line.set_data(x_data, y_data)
        line.set_3d_properties(z_data)
        point.set_data(df['x1950_x'][frame:frame+1], df['x1950_y'][frame:frame+1])
        point.set_3d_properties(df['x1950_z'][frame:frame+1])
        
        return line, point

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(df), init_func=init, blit=True, interval=50)

    # Show the plot (animation)
    plt.show()

#plot_xyz()
plot_r_v()