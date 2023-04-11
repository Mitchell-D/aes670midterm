"""
Generates custom RGBs based on user-selected values and prompts the user to
choose classification samples using the RGBs.

band  range          reason
3     459-479nm      blue
4     545-565nm      green
1     620-670nm      near-red
16    862-877nm      NIR / aerosol distinction
19    916-965nm      H2O absorption
5     1230-1250nm    optical depth
26    1360-1390nm    cirrus band
6     1628-1652nm    snow/ice band
7     2106-2155nm    cloud particle size
20    3660-3840nm    SWIR
21    3929-3989      another SWIR
27    6535-6895nm    Upper H2O absorption
28    7175-7475nm    Lower H2O absorption
29    8400-8700nm    Infrared cloud phase, emissivity diff 11-8.5um
31    10780-11280nm  clean LWIR
32    11770-12270nm  less clean LWIR
33    14085-14385nm  dirty LWIR
"""

from pathlib import Path
from datetime import datetime as dt
from datetime import timedelta as td
from pprint import pprint as ppt
import numpy as np
import pickle as pkl
import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import Affine2D

from aes670hw2 import laads, modis, geo_helpers, guitools, recipe_book
from aes670hw2 import guitools as gt
from aes670hw2 import geo_plot as gp
from aes670hw2 import enhance as enh
from aes670hw2 import PixelCat
from aes670hw2 import classify

def plot_classes(class_array:np.ndarray, fig_path:Path, class_labels:list,
                 color_map:dict):
    """
    Plots an integer array mapping pixels to a list of class labels
    """
    colors = [ color_map[l] for l in class_labels ]
    print(colors)
    cmap, norm = matplotlib.colors.from_levels_and_colors(
            list(range(len(colors)+1)), colors)
    im = plt.imshow(class_array, cmap=cmap, norm=norm, interpolation="none")
    handles = [ Patch(label=class_labels[i], color=colors[i])
               for i in range(len(class_labels)) ]
    plt.legend(handles=handles)
    print(f"saving figure as {fig_path.as_posix()}")
    plt.tick_params(axis="both", which="both", labelbottom=False,
                    labelleft=False, bottom=False, left=False)
    fig = plt.gcf()
    fig.set_size_inches(16,9)
    plt.savefig(fig_path, bbox_inches="tight", dpi=100)

""" Set up environment """
target_time = dt(year=2022, month=9, day=5, hour=8, minute=35)
l1b_pkl = Path("data/midterm_l1b.pkl")
fig_dir = Path("./figures")
data_dir = Path("./data")
debug=True

""" Load the pkl generated by get_region_pkl.py """
data, info, geo, sunsat = pkl.load(l1b_pkl.open("rb"))
# Convert band number to band array index
b2idx = lambda b: [ info[i]["band"] for i in range(len(info)) ].index(b)
lat, lon, _ = geo
sza, _, vza, _ = sunsat
all_bands = [info[i]["band"] for i in range(len(info))]

#km_bands = (1, 16, 5, 26, 6, 21, 29, 31)
km_bands = (1, 16, 5, 6, 7, 21, 29, 31, 26)
#km_bands = (1, 16, 19, 5, 7, 26)
#km_bands = (1, 16, 19, 5, 7, 26)
class_num = 8
#km_inputs = [ data[b2idx(b)] for b in km_bands ]
km_inputs = [ 10*enh.linear_gamma_stretch(data[b2idx(b)]) for b in km_bands ]
#km_inputs = [ data[b2idx(b)]/np.amax(data[b2idx(b)] ) for b in km_bands ]
pkl_path = Path("data/k_means_8c_all_2.pkl")

'''
tolerance = 1e-3
km,sse = classify.k_means(np.dstack(km_inputs), class_num, get_sse=True,
                 tolerance=tolerance, debug=debug)
with pkl_path.open("wb") as pklfp:
    pkl.dump((km,sse), pklfp)
'''

km, sse = pkl.load(pkl_path.open("rb"))
Y = np.zeros_like(km_inputs[0])
for c in range(class_num):
    # (px in category, band)
    tmp_vals = np.zeros((len(km[c]), len(km_bands)))
    for i in range(len(km[c])):
        y,x = km[c][i]
        #tmp_vals[i,:] = km_arrays[y,x]
        Y[y,x] = c

class_cmap = {
        "savanna":np.array([128,128,0]),
        "rainforest":np.array([0,128,0]),
        "cirrus":np.array([128,255,255]),
        "water":np.array([0,0,255]),
        "general cloud":np.array([220,220,220]),
        "deep cloud":255-np.array([255,255,255]),
        "low cloud":np.array([180,0,180]),
        "fire":np.array([201,174,111]),#np.array([255,0,0]),
        }

labels = [f"Class {i+1}" for i in range(class_num) ]
cmap = { labels[k]:list(class_cmap.values())[k]/255 for k in range(len(labels)) }
plot_classes(Y, fig_dir.joinpath("k_means_8c_all_2.png"), labels, cmap)
ssemin = np.amin(np.array(sse))
sse = [ s-ssemin+1 for s in sse ]

gp.basic_plot(range(len(sse)), sse, fig_dir.joinpath("k_means_sse.png"),
              plot_spec={
                  "title":"Sum of squared errors per K-means iteration",
                  "ylabel":"SSE for [0,1] normalized data (log scale)",
                  "xlabel":"K-means iteration count",
                  "grid":False,
                  })

json_path = data_dir.joinpath("scene_classes_squares.json")
json_classes = json.load(json_path.open("r"))
classes = {C["name"]:list(map(np.array, zip(*C["pixels"])))
           for C in json_classes}
class_px = {json_classes[i]["name"]:list(map(list,json_classes[i]["pixels"]))
           for i in range(len(json_classes))}

class_counts = {}
for c in class_px.keys():
    counts = np.zeros_like(np.arange(class_num))
    for y,x in class_px[c]:
        counts[int(Y[y,x])] += 1
    class_counts.update({c:counts})

print(class_counts)
for c in class_counts.keys():
    print(f"{c} &", " & ".join(list(map(str,class_counts[c]))), "\\\\")