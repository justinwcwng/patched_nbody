import numpy as np

# earth-centred, static moon system

earth_xyz = np.array([0, 0, 0]) # km
moon_xyz = np.array([-384400, 0, 0]) # km

# sun-centred, circular planetary orbits system

sun_xyz = np.array([0, 0, 0]) # km

def earth_xyz() -> tuple[np.ndarray]:
    t = np.linspace(0, 100, 100)
    earth_x = np.sin(t)
    earth_y = np.sin(t)
    earth_z = np.sin(t)
    return earth_x, earth_y, earth_z