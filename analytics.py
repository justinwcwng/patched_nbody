import numpy as np
import pandas as pd

patched = pd.read_csv("output/patched_EarthMoon.csv")
nbody = pd.read_csv("output/nbody_EarthMoon.csv")

with open("output/patchPoint.txt", "r") as f:
    patch_point = int(f.read().strip()) - 1

patched_x = patched["x (km)"][patch_point]
patched_y = patched["y (km)"][patch_point]
nbody_x = nbody["x (km)"][patch_point]
nbody_y = nbody["y (km)"][patch_point]

print(patched_x, patched_y)
print(nbody_x, nbody_y)