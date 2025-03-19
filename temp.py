import numpy as np
import matplotlib.pyplot as plt

def investigate_theta_F():
    e = np.arange(1.01, 3.1, 0.5)
    theta = np.linspace(0, 2*np.pi, 10000)

    for i in np.arange(len(e)):
        F0 = 2 * np.arctanh(np.sqrt((e[i] - 1) / (e[i] + 1)) * np.tan(theta / 2))
        plt.plot(theta, F0)
    plt.grid()
    plt.legend([_ for _ in e])
    plt.show()

#investigate_theta_F()