import numpy as np
from scipy.optimize import newton
from celestial_xyz import moon_xyz

def compute_a_e(
        ra: float,
        rp: float
    ) -> tuple[float]:

    """Calculates the semi-major axis (a) and eccentricity (e) using radius of apogee (ra) and radius of perigee (rp)."""

    # semi-major axis
    a = (ra + rp) / 2

    # eccentricity
    e = (ra - rp) / (ra + rp)

    return a, e

def solveE(M, e):

    """Solves Kepler's Equation for E given M and e using Newton's method."""
    
    return newton(lambda E: E - e * np.sin(E) - M, M, fprime=lambda E: 1 - e * np.cos(E))

def elliptical(
        mu: float,
        a: float,
        e: float,
        n_points: int = 10_000
    ) -> tuple[np.ndarray]:

    """
    Compute the position-time vector for an elliptical orbit.
    """

    T = np.pi * np.sqrt(a**3 / mu)  # time for half-orbit
    
    t = np.linspace(0, T, n_points) # generate time points
    n = np.sqrt(mu / a**3)          # mean motion
    M = n * t                       # mean anomaly

    # eccentric anomaly
    E = np.array([solveE(M_i, e) for M_i in M])

    # true anomaly
    theta = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    
    # obtain position components
    x = a * (np.cos(E) - e)
    y = a * np.sqrt(1 - e**2) * np.sin(E)
    z = np.zeros_like(x)
    r = np.sqrt(x**2 + y**2 + z**2)

    # obtain velocity components
    v = np.sqrt(mu * ((2 / r) - (1 / a))) # vis viva velocity
    v_r = (a * e * np.sin(E) * n) / (1 - e * np.cos(E)) # radial velocity
    v_t = np.sqrt(v**2 - v_r**2) # tangential velocity
    
    # convert to Cartesian
    vx = v_r * np.cos(theta) - v_t * np.sin(theta)
    vy = v_r * np.sin(theta) + v_t * np.cos(theta)
    vz = np.zeros_like(vx)

    return t, x, y, z, r, vx, vy, vz, v

def detect_soi_moon(
        soi_moon: float,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> np.ndarray[bool]:

    """
    Returns the index of the position-time vector when the spacecraft first enters the Moon's sphere of influence.
    """

    distanceFromMoon = np.sqrt((x-moon_xyz[0])**2 + (y-moon_xyz[1])**2 + (z-moon_xyz[2])**2)
    soi_bool = (distanceFromMoon < soi_moon)
    first_true_index = np.argmax(soi_bool) if np.any(soi_bool) else -1

    return first_true_index

def orbital_elements(
        mu: float,
        x: np.ndarray, y: np.ndarray, z: np.ndarray,
        vx: np.ndarray, vy: np.ndarray, vz: np.ndarray
    ) -> tuple:
    """
    Computes Keplerian orbital elements in 3D.
    
    Parameters:
        x, y, z     : np.ndarray - Position components (km)
        vx, vy, vz  : np.ndarray - Velocity components (km/s)
        mu          : float - Gravitational parameter of the celestial body (km^3/s^2)

    Returns:
        dict: Keplerian orbital elements {a, e, i, Omega, omega, theta}
    """

    # Compute relative position & velocity
    r_vec = np.stack((x, y, z), axis=-1) - moon_xyz  # Shape: (N, 3)
    v_vec = np.stack((vx, vy, vz), axis=-1)  # Shape: (N, 3)
    
    r = np.linalg.norm(r_vec, axis=1)  # Compute magnitude of r
    v = np.linalg.norm(v_vec, axis=1)  # Compute magnitude of v

    # Compute specific angular momentum h = r × v
    h_vec = np.cross(r_vec, v_vec)  # Shape: (N, 3)
    h = np.linalg.norm(h_vec, axis=1)  # Magnitude of angular momentum

    # Compute eccentricity vector e
    v_sq = np.sum(v_vec**2, axis=1)  # Equivalent to v⋅v
    r_dot_v = np.sum(r_vec * v_vec, axis=1)  # Equivalent to r⋅v

    e_vec = (1 / mu) * ((v_sq[:, None] - mu / r[:, None]) * r_vec - r_dot_v[:, None] * v_vec)
    e = np.linalg.norm(e_vec, axis=1)  # Magnitude of eccentricity
    e = float(e)

    # Compute semi-major axis a using Vis-Viva equation
    a = 1 / ((2 / r) - (v_sq / mu))
    a = float(a)

    # Compute inclination i
    i = np.arccos(h_vec[:, 2] / h)  # h_z component
    i = float(i)

    # Compute Longitude of Ascending Node (Ω)
    node_vec = np.cross(np.array([0, 0, 1]), h_vec)  # Node vector (N, 3)
    node_mag = np.linalg.norm(node_vec, axis=1)
    Omega = np.arctan2(node_vec[:, 1], node_vec[:, 0])  # atan2(ny, nx)
    Omega = float(Omega)

    # Compute Argument of Periapsis (ω)
    omega = np.arccos(np.sum(node_vec * e_vec, axis=1) / (node_mag * e))
    omega = np.where(e_vec[:, 2] < 0, 2 * np.pi - omega, omega)  # Adjust for sign
    if i == 0: omega = np.arctan2(e_vec[0,1], e_vec[0,0]) # 2D
    omega = float(omega)

    # Compute True Anomaly (θ)
    theta = np.arccos(np.sum(e_vec * r_vec, axis=1) / (e * r))
    theta = np.where(r_dot_v < 0, 2 * np.pi - theta, theta)  # Adjust for sign
    theta = float(theta)

    return a, e, i, Omega, omega, theta

def solveF(M, e):

    """Solves Kepler's Equation for F given M and e using Newton's method."""

    return newton(lambda F: e * np.sinh(F) - F - M, M, fprime=lambda F: e * np.cosh(F) - 1)

def hyperbolic(
        mu: float,
        a: float,
        e: float,
        theta: float,
        omega: float,
        t_max: float = 50000,
        n_points: int = 5000
    ) -> tuple[np.ndarray]:

    """
    Compute the position-time vector for a hyperbolic trajectory.
    Currently trialling a fixed Moon hyperbolic fly-by.
    """

    t = np.linspace(0, t_max, n_points) # time points

    # initial hyperbolic eccentric anomaly
    F0 = 2 * np.arctanh(np.sqrt((e - 1) / (e + 1)) * np.tan(theta / 2))

    M0 = e * np.sinh(F0) - F0   # initial mean anomaly
    n = np.sqrt(mu / abs(a)**3) # mean motion
    M = M0 - n * t              # mean anomaly

    # hyperbolic eccentric anomaly
    F = np.array([solveF(M_i, e) for M_i in M])

    # obtain position components using hyperbolic formulas
    x = a * (np.cosh(F) - e)
    y = a * np.sqrt(e**2 - 1) * np.sinh(F)
    z = np.zeros_like(x)

    # rotate by argument of periapsis
    x_rot = x * np.cos(omega) - y * np.sin(omega)
    y_rot = x * np.sin(omega) + y * np.cos(omega)

    # adjust to moon-centred frame
    x_rot += moon_xyz[0]
    y_rot += moon_xyz[1]
    z += moon_xyz[2]
    
    r = np.sqrt(x_rot**2 + y_rot**2 + z**2)

    # obtain velocity components
    v = np.sqrt(mu * (2 / r - 1 / a)) # vis viva velocity
    # vx, vy, vz to be found

    return t, x_rot, y_rot, z, r, v