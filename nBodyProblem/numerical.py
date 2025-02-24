import numpy as np

from constants import mu_earth, mu_moon, r_moon

def moon_position(theta: float=0) -> list[float]:

    """
    Returns the Moon's instantaneous position as a list.
    """

    x = -384400 #r_moon * np.cos(theta)
    y = 0 #r_moon * np.sin(theta)
    z = 0
    return [x, y, z]

def acceleration(x, y):

    """
    Computes the acceleration due to Earth and Moon only.
    """

    r_earth_mag = np.sqrt(x**2 + y**2)
    a_earth = mu_earth / r_earth_mag**2 # instantaneous centripetal acceleration

    alpha = np.arctan2(y, x) - np.pi
    a_x_earth = a_earth * np.cos(alpha)
    a_y_earth = a_earth * np.sin(alpha)

    r_moon = moon_position()
    r_moon_mag = np.sqrt((x - r_moon[0])**2 + (y - r_moon[1])**2)
    a_moon = mu_moon / r_moon_mag**2 # instantaneous centripetal acceleration

    alpha_moon = np.arctan2((y - r_moon[1]), (x - r_moon[0])) - np.pi
    a_x_moon = a_moon * np.cos(alpha_moon)
    a_y_moon = a_moon * np.sin(alpha_moon)

    return a_x_earth + a_x_moon, a_y_earth + a_y_moon #a_x_earth, a_y_earth

def verlet_integrator(state, h):

    """
    Returns the state after one time step.
    """
    
    x, y, v_x, v_y = state # unpack

    a_x_old, a_y_old = acceleration(x, y)
    
    # update position
    x += v_x * h + 0.5 * a_x_old * h**2
    y += v_y * h + 0.5 * a_y_old * h**2

    a_x_new, a_y_new = acceleration(x, y)
    
    # update velocity
    v_x += 0.5 * (a_x_old + a_x_new) * h
    v_y += 0.5 * (a_y_old + a_y_new) * h
    
    # update state
    state[0] = x
    state[1] = y
    state[2] = v_x
    state[3] = v_y
    
    return state