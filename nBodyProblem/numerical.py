import numpy as np

from constants import mu_earth, mu_moon, r_moon, soi_moon
from celestial_xyz import earth_xyz

def moon_position(theta: float=0) -> list[float]:

    """
    Returns the Moon's instantaneous position as a list.
    """

    x = -384400 #r_moon * np.cos(theta)
    y = 0 #r_moon * np.sin(theta)
    z = 0
    return [x, y, z]

def acceleration(x, y) -> tuple[float]:

    """
    Computes the acceleration due to Earth and Moon only.
    """

    r_earth_mag = np.sqrt((x - earth_xyz[0])**2 + (y - earth_xyz[1])**2)
    a_earth = mu_earth / r_earth_mag**2 # instantaneous centripetal acceleration

    alpha = np.arctan2((y - earth_xyz[1]), (x - earth_xyz[0])) - np.pi
    a_x_earth = a_earth * np.cos(alpha)
    a_y_earth = a_earth * np.sin(alpha)

    r_moon = moon_position()
    r_moon_mag = np.sqrt((x - r_moon[0])**2 + (y - r_moon[1])**2)
    a_moon = mu_moon / r_moon_mag**2 # instantaneous centripetal acceleration

    alpha_moon = np.arctan2((y - r_moon[1]), (x - r_moon[0])) - np.pi
    a_x_moon = a_moon * np.cos(alpha_moon)
    a_y_moon = a_moon * np.sin(alpha_moon)

    # manual gravity override
    distanceFromMoon = np.sqrt((x - r_moon[0])**2 + (y - r_moon[1])**2)
    soi_bool = (distanceFromMoon < soi_moon)
    if soi_bool:
        acc = 0.0*a_x_earth, 0.0*a_y_earth, a_x_moon, a_y_moon
    else:
        acc = a_x_earth, a_y_earth, 0, 0

    #acc = a_x_earth + a_x_moon, a_y_earth + a_y_moon # if using verlet_integrator

    #acc = a_x_earth, a_y_earth, a_x_moon, a_y_moon # if using integrator_investigator

    return acc

def verlet_integrator(state, h):

    """
    Returns the state after one time step.
    State consists of x, y, v_x, v_y, a_x, a_y
    """

    x, y, v_x, v_y, a_x, a_y = state # unpack

    a_x_old, a_y_old = acceleration(x, y)
    
    # update position
    x += v_x * h + 0.5 * a_x_old * h**2
    y += v_y * h + 0.5 * a_y_old * h**2

    a_x_new, a_y_new = acceleration(x, y)

    a_x = 0.5 * (a_x_old + a_x_new)
    a_y = 0.5 * (a_y_old + a_y_new)
    
    # update velocity
    v_x += a_x * h
    v_y += a_y * h
    
    # update state
    state[0] = x
    state[1] = y
    state[2] = v_x
    state[3] = v_y
    state[4] = a_x
    state[5] = a_y
    
    return state

def integrator_investigator(state, h):

    """
    Returns the state after one time step.
    State consists of x, y, v_x, v_y, a_x, a_y
    """

    x, y, v_x, v_y, *_ = state # unpack

    a_x_earth_old, a_y_earth_old, a_x_moon_old, a_y_moon_old = acceleration(x, y)
    
    a_x_old = a_x_earth_old + a_x_moon_old
    a_y_old = a_y_earth_old + a_y_moon_old

    # update position
    x += v_x * h + 0.5 * a_x_old * h**2
    y += v_y * h + 0.5 * a_y_old * h**2

    a_x_earth_new, a_y_earth_new, a_x_moon_new, a_y_moon_new = acceleration(x, y)

    a_x_new = a_x_earth_new + a_x_moon_new
    a_y_new = a_y_earth_new + a_y_moon_new

    a_x = 0.5 * (a_x_old + a_x_new)
    a_y = 0.5 * (a_y_old + a_y_new)

    a_x_earth = 0.5 * (a_x_earth_old + a_x_earth_new)
    a_y_earth = 0.5 * (a_y_earth_old + a_y_earth_new)
    a_x_moon = 0.5 * (a_x_moon_old + a_x_moon_new)
    a_y_moon = 0.5 * (a_y_moon_old + a_y_moon_new)
    
    # update velocity
    v_x += a_x * h
    v_y += a_y * h
    
    # update state
    state[0] = x
    state[1] = y
    state[2] = v_x
    state[3] = v_y
    state[4] = a_x
    state[5] = a_y
    state[6] = a_x_earth
    state[7] = a_y_earth
    state[8] = a_x_moon
    state[9] = a_y_moon
    
    return state