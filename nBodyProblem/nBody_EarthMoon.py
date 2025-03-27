import numpy as np
import pandas as pd

import time
import psutil

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constants import mu_earth, mu_moon, R_earth, r_LEO, r_moon

from calculations import hohmann

from nBodyProblem.numerical import acceleration, verlet_integrator, integrator_investigator

def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2  # convert bytes to MB

def nbody_sim(
        r0 = r_LEO,
        r_target = r_moon,
        t_sim: float = 700_000
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

    print()
    print("- "*27)
    print()
    print("nbody problem simulation initiated")

    start_time = time.time()
    mem_before = memory_usage()

    # initial acceleration
    a_x_earth, a_y_earth, a_x_moon, a_y_moon = acceleration(r0, 0)
    a_x = a_x_earth + a_x_moon
    a_y = a_y_earth + a_y_moon

    # initialise state vectors [x, y, v_x, v_y, a_x, a_y, a_x_e, a_y_e, a_x_m, a_y_m]
    state = [r0, 0, 0, v + delta_v1, a_x, a_y, a_x_earth, a_y_earth, a_x_moon, a_y_moon]

    xArray = [state[0]]
    yArray = [state[1]]
    v_xArray = [state[2]]
    v_yArray = [state[3]]
    a_xArray = [state[4]]
    a_yArray = [state[5]]
    a_x_earthArray = [state[6]]
    a_y_earthArray = [state[7]]
    a_x_moonArray = [state[8]]
    a_y_moonArray = [state[9]]

    for i in np.arange(numFrame):
        state = integrator_investigator(state, h)
        x, y, v_x, v_y, a_x, a_y, a_x_earth, a_y_earth, a_x_moon, a_y_moon = state # unpack
        xArray += [x]
        yArray += [y]
        v_xArray += [v_x]
        v_yArray += [v_y]
        a_xArray += [a_x]
        a_yArray += [a_y]
        a_x_earthArray += [a_x_earth]
        a_y_earthArray += [a_y_earth]
        a_x_moonArray += [a_x_moon]
        a_y_moonArray += [a_y_moon]

    end_time = time.time()
    mem_after = memory_usage()
    duration = end_time - start_time
    memory_used = mem_after - mem_before

    ### END OF SIMULATION ###

    print()
    print("- "*27)
    print()
    if duration < 1:
        msg = f"n-body simulation successful [{duration*1e3:.5f} ms]"
    else:
        msg = f"n-body simulation successful [{duration:.5f} s]"
    print(msg)
    print()
    print(f"memory used: {memory_used:.5f} MB")

    zArray = np.zeros(len(xArray))
    v_zArray = np.zeros(len(xArray))
    a_zArray = np.zeros(len(xArray))
    a_z_earthArray = np.zeros(len(xArray))
    a_z_moonArray = np.zeros(len(xArray))

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
        "a (km/s2)": np.sqrt(np.array(a_xArray)**2 + np.array(a_yArray)**2 + np.array(a_zArray)),
        "axe (km/s2)": a_x_earthArray,
        "aye (km/s2)": a_y_earthArray,
        "aze (km/s2)": a_z_earthArray,
        "ae (km/s2)": np.sqrt(np.array(a_x_earthArray)**2 + np.array(a_y_earthArray)**2 + np.array(a_z_earthArray)),
        "axm (km/s2)": a_x_moonArray,
        "aym (km/s2)": a_y_moonArray,
        "azm (km/s2)": a_z_moonArray,
        "am (km/s2)": np.sqrt(np.array(a_x_moonArray)**2 + np.array(a_y_moonArray)**2 + np.array(a_z_moonArray))
    })

    print()
    print("data shape:", df_out.shape)
    df_out.to_csv("output/nBody_EarthMoon.csv")
    print()
    print("csv export completed")
    print()
    print("="*53)

    return duration

if __name__ == "__main__":
    nbody_sim()