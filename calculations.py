import numpy as np

from constants import mu_earth, r_LEO, r_moon

def acceleration() -> float:

    return 0

def circular_v(
        mu: float,
        r: float
    ) -> float:

    v = np.sqrt(mu / r)

    return v

def elliptical_v(
        mu: float,
        r: float,
        r1: float,
        r2: float
    ) -> float:

    v = np.sqrt((2 * mu / r) - (2 * mu / (r1 + r2)))

    return v

def hohmann(
        mu: float,
        r1: float,
        r2: float
    ) -> tuple[float]:

    v1_minus = circular_v(mu, r1)
    v1_plus = elliptical_v(mu, r1, r1, r2)
    delta_v1 = v1_plus - v1_minus

    v2_minus = elliptical_v(mu, r2, r1, r2)
    v2_plus = circular_v(mu, r2)
    delta_v2 = v2_plus - v2_minus

    return delta_v1, delta_v2

if __name__ == "__main__":

    v1, v2 = hohmann(mu = mu_earth,
                     r1 = r_LEO+30000,
                     r2 = r_moon+15000)
    
    print()
    print(f"1st delta v = {v1:.5f} km/s")
    print(f"2nd delta v = {v2:.5f} km/s")
    print()

    #ans = # custom calculation
    #print(f"{ans:.3f}")