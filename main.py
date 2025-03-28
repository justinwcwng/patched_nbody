from patchedConicApprox.patched_EarthMoon import patched_sim
from nBodyProblem.nBody_EarthMoon import nbody_sim
from analytics import process_and_export
from visualise import plot_xy, plot_r_v, plot_r_v_diff, plot_vx_vy, plot_acc

from constants import R_earth, r_moon

def main():

    r1 = R_earth + 5_000 # km
    r2 = r_moon + 5_000 # km
    t_sim = 500_000 # s

    # simulation
    t_patched, RAM_patched = patched_sim(r1, r2, t_sim)
    t_nbody, RAM_nbody  = nbody_sim(r1, r2, t_sim)

    # match size and compare
    process_and_export()

    # visualisation
    plot_xy()
    plot_r_v()
    plot_r_v_diff()
    plot_vx_vy()
    plot_acc()

    print()
    print(f"patched conic method: {t_patched:.5f} s, {RAM_patched:.5f} MB")
    print(f"nbody method: {t_nbody:.5f} s, {RAM_nbody:.5f} MB")
    print()
    print(f"nbody runs for {(t_nbody-t_patched)/t_patched:.0f}x longer")
    print(f"nbody uses {(RAM_nbody-RAM_patched)/RAM_patched:.0f}x more RAM")
    print()
    print("="*53)
    print()

if __name__ == "__main__":
    main()