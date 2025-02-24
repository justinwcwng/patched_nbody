import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

df = pd.read_csv("data/as16_pancam_state_vectors.csv")

df["utc_time_str"] = pd.to_datetime(df["utc_time_str"], errors="coerce")
df2 = df[["utc_time_str", "x1950_x", "x1950_y", "x1950_z"]]
df2.sort_values(by="utc_time_str", inplace=True)

df2.to_csv("data/current_data.csv")

x = df2["x1950_x"]
y = df2["x1950_y"]
z = df2["x1950_z"]

def plotter():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #for i in range(len(x)):
    #    ax.scatter(x[i], y[i], z[i], c="black", s=2)
    #    plt.draw()
    #    plt.pause(0.1)

    ax.scatter(x, y, z, marker='o', alpha=1, linewidths=0, edgecolors=None, s=2)
    plt.axis("equal")
    plt.savefig("plots/plot.png")

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

plotter()