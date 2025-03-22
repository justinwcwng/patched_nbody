def elliptical(
        mu: float,
        a: float,
        e: float,
        theta0: float,
        omega: float,
        t_max: float,
        n_points: int = 10_000
    ) -> tuple[np.ndarray]:

    """
    Compute the position-time vector for an elliptical orbit.
    """

    t = np.linspace(0, t_max, n_points) # generate time points

    # initial eccentric anomaly
    E0 = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(theta0 / 2))

    M0 = E0 - e * np.sin(E0)  # initial mean anomaly
    n = np.sqrt(mu / a**3)    # mean motion
    M = M0 + n * t            # mean anomaly

    # eccentric anomaly
    E = np.array([solveE(M_i, e) for M_i in M])

    # true anomaly
    theta = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))

    # obtain position components
    x = a * (np.cos(E) - e)
    y = a * np.sqrt(1 - e**2) * np.sin(E)
    z = np.zeros_like(x)

    # rotate by argument of periapsis
    x_rot = x * np.cos(omega) - y * np.sin(omega)
    y_rot = x * np.sin(omega) + y * np.cos(omega)

    r = np.sqrt(x_rot**2 + y_rot**2 + z**2)

    # obtain velocity components
    v = np.sqrt(mu * ((2 / r) - (1 / a)))                # vis viva velocity
    v_r = (a * e * np.sin(E) * n) / (1 - e * np.cos(E))  # radial velocity
    v_t = np.sqrt(v**2 - v_r**2)                         # tangential velocity
    
    # convert to Cartesian
    vx = v_r * np.cos(theta) - v_t * np.sin(theta)
    vy = v_r * np.sin(theta) + v_t * np.cos(theta)
    vz = np.zeros_like(vx)

    return t, x_rot, y_rot, z, r, vx, vy, vz, v




    h = r_moon * v # angular momentum
    #v_r = (mu / h) * (e * np.sin(theta))
    v_r = (a * e * np.sinh(F) * n) / (e * np.cosh(F) - 1) # radial velocity
    v_t = np.sqrt(v**2 - v_r**2) # tangential velocity

    # convert to Cartesian
    theta = theta0
    vx = v_r * np.cos(theta) - v_t * np.sin(theta)
    vy = v_r * np.sin(theta) + v_t * np.cos(theta)
    vz = np.zeros_like(vx)



















