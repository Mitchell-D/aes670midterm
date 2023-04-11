"""
Quick and easy theoretical atmospheric transmittance in .9um absorption range
"""

import numpy as np
import json
from pathlib import Path
from scipy.interpolate import interp1d

rsr_dict = json.load(Path("data/modis_rsrs.json").open("r"))['19']
atran_path = Path("atran_860-1300.dat")
rsr_wl = rsr_dict["wavelength"]
rsr = rsr_dict["rsr"]

with atran_path.open("r") as atranfp:
    lines = atranfp.readlines()
    atran_wl = []
    atran_t = []
    for _,wl,t in map(str.split, lines):
        atran_wl.append(float(wl))
        atran_t.append(float(t))
    atran_wl = np.asarray(atran_wl)
    atran_t = np.asarray(atran_t)

print(np.amin(atran_wl), np.amax(atran_wl))
print(np.amin(rsr_wl), np.amax(rsr_wl))
print(np.amin(atran_t), np.amax(atran_t))
print(np.amin(rsr), np.amax(rsr))

# Since transmission is scaled 0 to 1, the average value is total transmission
print(np.average(interp1d(atran_wl, atran_t)(rsr_wl)*rsr))
