import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, FFMpegWriter
from constants import R_earth

def ani(r, x, y, v, fps, numFrame, timescale, xArray, yArray):

    fig, ax = plt.subplots(figsize=(16,9))
    ax.set_facecolor("#050505") # background colour is space gray
    ax.set_xticks([]) # hide x-ticks
    ax.set_yticks([]) # hide y-ticks
    # ax.set_xlabel("distance (x10e3 km)")

    drawEarth = Circle((0, 0), R_earth, edgecolor="#50C878", facecolor="#89CFF0", linewidth=3.5) # edgecolor is emerald green, facecolor is baby blue
    ax.add_patch(drawEarth) # Earth represented as a patch

    # initiate animated objects
    path, = ax.plot([], [], lw=2, color="#FFD700") # trajectory colour is golden yellow
    sqSat, = ax.plot([], [], "rD", markersize=7) # satellite represented as a red square
    txTime = ax.text(0.93, 0.07, "", transform=ax.transAxes, ha="right", va="bottom", color="#FFFFFF")
    
    def init():
        ax.set_xlim(-3.2*r, 3.2*r)
        ax.set_ylim(-1.8*r, 1.8*r)
        ax.set_aspect("equal")
        return sqSat, path, txTime

    def update(frame):
        
        sqSat.set_data(xArray[frame:frame+1], yArray[frame:frame+1]) # update instantaneous satellite position
        
        path.set_data(xArray[:frame], yArray[:frame]) # update path taken by satellite so far
        
        hrsElapsed = (timescale / 3600) * (frame / fps) # update elapsed satellite time in hours
        txTime.set_text(f"{r/1000000:.2f}e3 km, {v:.1f} km/s, {hrsElapsed:.2f} hrs")


        # adjust limits as necessary
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        margin = 2e5
        if x <= 0.8*xmin:
            ax.set_xlim(xmin-margin, xmax)
            ax.figure.canvas.draw()
        if x >= 0.8*xmax:
            ax.set_xlim(xmin, xmax+margin)
            ax.figure.canvas.draw()
        if y <= 0.8*ymin:
            ax.set_ylim(ymin-margin, ymax)
            ax.figure.canvas.draw()
        if y >= 0.8*ymax:
            ax.set_ylim(ymin, ymax+margin)
            ax.figure.canvas.draw()
        
        return sqSat, path, txTime

    ani = FuncAnimation(fig=fig, func=update, frames=numFrame, interval=1000/fps, init_func=init, blit=True)

    print("Generating Animation...")
    print()

    #HTML(ani.to_jshtml())
    #FFwriter = FFMpegWriter()
    ani.save('animation.gif', writer='pillow')

if __name__ == "__main__":
    pass
    # x, y, _, r, ... = nbody_sim(r0, r_target)
    # ani(r, x, y, v, fps, numFrame, timescale, xArray, yArray)