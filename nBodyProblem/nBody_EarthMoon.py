import numpy as np
import pandas as pd

import time

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constants import mu_earth, mu_moon, R_earth, r_LEO, r_moon

from calculations import hohmann

from nBodyProblem.numerical import acceleration, verlet_integrator

def nbody_sim(
        r0 = r_LEO,
        r_target = r_moon,
        t_sim: float = 600_000
    ) -> float:

    """
    Main simulation function for the n-Body Problem method.
    Currently trialling an Earth-Moon trajectory.
    """

    delta_v1, _ = hohmann(mu_earth, r0, r_target)

    v = np.sqrt(mu_earth / r0) # orbital velocity

    a = (r0 + r_target) / 2 # semi-major axis
    T = 2 * np.pi * np.sqrt(a**3 / mu_earth) # orbital period
    T = t_sim

    h = 0.000001 * T

    numFrame = T / h

    ### START OF SIMULATION ###

    start_time = time.time()

    acc, _ = acceleration(r0, 0) # initial acceleration

    # initialise state vectors [x, y, v_x, v_y]
    state = [r0, 0, 0, v + delta_v1, acc, 0] # initial state

    xArray = [state[0]]
    yArray = [state[1]]
    v_xArray = [state[2]]
    v_yArray = [state[3]]
    a_xArray = [state[4]]
    a_yArray = [state[5]]

    for i in np.arange(numFrame):
        state = verlet_integrator(state, h)
        x, y, v_x, v_y, a_x, a_y = state # unpack
        xArray += [x]
        yArray += [y]
        v_xArray += [v_x]
        v_yArray += [v_y]
        a_xArray += [a_x]
        a_yArray += [a_y]

    end_time = time.time()
    duration = end_time - start_time

    ### END OF SIMULATION ###

    print()
    if duration < 1:
        msg = f"n-body simulation successful [{duration*1e3:.5f} ms]"
    else:
        msg = f"n-body simulation successful [{duration:.5f} s]"
    print(msg)

    zArray = np.zeros(len(xArray))
    v_zArray = np.zeros(len(xArray))
    a_zArray = np.zeros(len(xArray))

    df_out = pd.DataFrame({
        "t (s)": np.arange(numFrame+1) * h,
        "x (km)": xArray,
        "y (km)": yArray,
        "z (km)": zArray,
        "r (km)": np.sqrt(np.array(xArray)**2 + np.array(yArray)**2 + np.array(zArray)),
        "vx (km/s)": v_xArray,
        "vy (km/s)": v_yArray,
        "vz (km/s)": v_zArray,
        "v (km/s)": np.sqrt(np.array(v_xArray)**2 + np.array(v_yArray)**2 + np.array(v_zArray)),
        "ax (km/s2)": a_xArray,
        "ay (km/s2)": a_yArray,
        "az (km/s2)": a_zArray,
        "a (km/s2)": np.sqrt(np.array(a_xArray)**2 + np.array(a_yArray)**2 + np.array(a_zArray))
    })

    print()
    print("data shape:", df_out.shape)
    df_out.to_csv("output/nBody_EarthMoon.csv")
    print("csv export completed")
    print()

    return duration

if __name__ == "__main__":
    nbody_sim()