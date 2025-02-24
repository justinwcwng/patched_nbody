from patchedConicApprox.patched_EarthMoon import patched_sim
from nBodyProblem.nBody_EarthMoon import nbody_sim
from visualise import plot_v, plot_xy

from constants import r_LEO, r_moon

def main():

    r1 = r_LEO
    r2 = r_moon

    t_sim = 600_000 # s

    # simulation
    t_patched = patched_sim(r1, r2, t_sim)
    t_nbody = nbody_sim(r1, r2, t_sim)

    # visualisation
    plot_xy(patched = False)
    plot_v(patched = False)

    print()
    print(f"patched conic method: {t_patched:.5f} s")
    print(f"nbody method: {t_nbody:.5f} s")
    print()

if __name__ == "__main__":
    main()