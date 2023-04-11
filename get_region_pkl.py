"""
Generates the pkl file used for the midterm, and does histogram analysis.
"""

from pathlib import Path
from datetime import datetime as dt
from datetime import timedelta as td
from pprint import pprint as ppt
import multiprocessing as mp
import numpy as np
import pickle as pkl

from aes670hw2 import laads, modis, geo_helpers, guitools
from aes670hw2 import guitools as gt
from aes670hw2 import geo_plot as gp
from aes670hw2 import enhance as enh

token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUFMgT0F1dGgyIEF1dGhlbnRpY2F0b3IiLCJpYXQiOjE2Nzg0MDEyNzgsIm5iZiI6MTY3ODQwMTI3OCwiZXhwIjoxNjkzOTUzMjc4LCJ1aWQiOiJtZG9kc29uIiwiZW1haWxfYWRkcmVzcyI6Im10ZDAwMTJAdWFoLmVkdSIsInRva2VuQ3JlYXRvciI6Im1kb2Rzb24ifQ.gwlWtdrGZ1CNqeGuNvj841SjnC1TkUkjxb6r-w4SOmk"

l2_url = "https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/archives/MOD09.A2022248.0835.006.2022250030855.hdf"
l1b_url = "https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/archives/MOD021KM.A2022248.0835.061.2022248191738.hdf"

target_time = dt(year=2022, month=9, day=5, hour=8, minute=35)
time_radius = td(hours=5)

target_latlon = (-5, 26)
#region_width, region_height = 640*2, 512*2
region_width, region_height = 640, 512
nbins = 512 # For histogram analysis
#bands = ("R1_1000", "R4_1000", "R3_1000")
l2_bands = ("R1_500", "R4_500", "R3_500")
l1b_pkl = Path("data/midterm_l1b.pkl")
l1b_bands = (
        3,  # 459-479nm blue
        4,  # 545-565nm green
        1,  # 620-670nm near-red
        16, # 862-877nm NIR / aerosol distinction
        19, # 916-965nm H2O absorption
        5,  # 1230-1250nm optical depth
        26, # 1360-1390nm cirrus band
        6,  # 1628-1652nm snow/ice band
        7,  # 2106-2155nm cloud particle size
        20, # 3660-3840nm SWIR
        21, # 3929-3989 another SWIR
        27, # 6535-6895nm Upper H2O absorption
        28, # 7175-7475nm Lower H2O absorption
        29, # 8400-8700nm Infrared cloud phase, emissivity diff 11-8.5um
        31, # 10780-11280nm clean LWIR
        32, # 11770-12270nm less clean LWIR
        33, # 14085-14385nm dirty LWIR
        )
data_dir = Path("./data")
buf_dir = Path("./buffer")
fig_dir = Path("./figures")
sat = "terra"
img_template = "%Y%m%d_%H%M_{sat}_truecolor.png"
debug=True

""" Restore the subsetted data if the pkl is removed. """
if not l1b_pkl.exists():
    l1b_path = data_dir.joinpath(
            f"{target_time.strftime('%Y%m%d_%H%M')}_{sat}_l1b.hdf")
    l2_path = data_dir.joinpath(
            f"{target_time.strftime('%Y%m%d_%H%M')}_{sat}_l2.hdf")

    if not l1b_path.exists():
        tmp_path = laads.download(l1b_url, data_dir, raw_token=token,
                                  replace=True, debug=debug)
        tmp_path.rename(l1b_path)
    if not l2_path.exists():
        tmp_path = laads.download(l2_url, data_dir, raw_token=token,
                                  replace=True, debug=debug)
        tmp_path.rename(l2_path)

    #data, info, geo = modis.get_modis_data(l2_path, bands=l2_bands)
    data, info, geo, sunsat = modis.get_modis_data(
            l1b_path, bands=l1b_bands)

    # Crop to the appropriate size and unpack values
    dy, dx = geo_helpers.get_geo_range(
            latlon=np.dstack(geo[:2]), target_latlon=target_latlon,
            dx_px=region_width, dy_px=region_height, from_center=True,
            boundary_error=False, debug=True)
    subset = lambda X: X[dy[0]:dy[1], dx[0]:dx[1]]
    data = list(map(subset, data))
    geo = list(map(subset, geo))
    sunsat = list(map(subset, sunsat))

    with l1b_pkl.open("wb") as pklfp:
        pkl.dump((data, info, geo, sunsat), pklfp)
else:
    data, info, geo, sunsat = pkl.load(l1b_pkl.open("rb"))

lat, lon, _ = geo
sza, _, vza, _ = sunsat

""" Do histogram analysis on the selected bands and generate figures """
ref_hists = []
tb_hists = []
for i in range(len(data)):
    ctr_wl = modis.band_to_wl(info[i]['band'])
    if info[i]["is_reflective"]:
        ref_hists.append(enh.do_histogram_analysis(
            data[i], nbins, equalize=False, debug=False))
        ref_hists[-1].update({"band":info[i]["band"],
                              "ctr_wl":info[i]["ctr_wl"]})
    else:
        tb_hists.append(enh.do_histogram_analysis(
            data[i], nbins, equalize=False, debug=False))
        tb_hists[-1].update({"band":info[i]["band"],
                             "ctr_wl":info[i]["ctr_wl"]})

gp.plot_lines(
    domain=range(nbins),
    ylines=[h["hist"] for h in ref_hists],
    labels=[
        f"Band {h['band']} ({h['ctr_wl']:.2f})" + \
            "\n$\mu = $"+str(gp.round_to_n(h["mean"], 5)) + \
            "    $\sigma = $"+str(gp.round_to_n(h["stddev"], 5))
        for h in ref_hists],
    plot_spec = {
        "title": "MODIS reflective band frequency histograms",
        "xlabel": "Brightness level bin",
        "ylabel": "Brightness level frequency",
        "line_width":.5,
        },
    image_path=fig_dir.joinpath("hists_ref.png"),
    show=False
    )

gp.plot_lines(
    domain=range(nbins),
    ylines=[h["hist"] for h in tb_hists],
    labels=[
        f"Band {h['band']} ({h['ctr_wl']:.2f})" + \
            "\n$\mu = $"+str(gp.round_to_n(h["mean"], 5)) + \
            "    $\sigma = $"+str(gp.round_to_n(h["stddev"], 5))
        for h in tb_hists],
    plot_spec = {
        "title": "MODIS thermal band frequency histograms",
        "xlabel": "Brightness level bin",
        "ylabel": "Brightness level frequency",
        "line_width":.5,
        },
    image_path=fig_dir.joinpath("hists_tb.png"),
    show=False
    )

""" Generate scalar plots for each wavelength """

for i in range(len(data)):
    is_ref = info[i]['is_reflective']
    gp.geo_scalar_plot(
            data[i],lat,lon,
            fig_dir.joinpath(f"proj/proj_band_{info[i]['band']:02}.png"),
            plot_spec={
                "title":f"MODIS {info[i]['ctr_wl']:.3f} $\mu m$ "+\
                        ("Brightness Temperature", "Reflectance")[is_ref],
                "cb_label_format":(None, "{x:.3f}")[is_ref],
                "cb_orient":"horizontal",
                "title_size":16,
                }
            )

#ref_enh = lambda X: enh.norm_to_uint(np.clip(X, 0, 1), 256, np.uint8)
#gt.quick_render(ref_enh(data[6]))
#gt.quick_render(enh.norm_to_uint(lon, 256, np.uint8))
