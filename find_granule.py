""" """

from pathlib import Path
from datetime import datetime as dt
from pprint import pprint as ppt
import multiprocessing as mp
import numpy as np

from aes670hw2 import laads, viirs, modis, enhance
from aes670hw2 import geo_helpers, geo_plot, guitools, imstat

token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUFMgT0F1dGgyIEF1dGhlbnRpY2F0b3IiLCJpYXQiOjE2Nzg0MDEyNzgsIm5iZiI6MTY3ODQwMTI3OCwiZXhwIjoxNjkzOTUzMjc4LCJ1aWQiOiJtZG9kc29uIiwiZW1haWxfYWRkcmVzcyI6Im10ZDAwMTJAdWFoLmVkdSIsInRva2VuQ3JlYXRvciI6Im1kb2Rzb24ifQ.gwlWtdrGZ1CNqeGuNvj841SjnC1TkUkjxb6r-w4SOmk"
start_time = dt(year=2022, month=9, day=5)
end_time = dt(year=2022, month=9, day=11, hour=5)
target_latlon = (-5, 26)
#region_width, region_height = 640*2, 512*2
region_width, region_height = 640, 512
#bands = ("R1_1000", "R4_1000", "R3_1000")
modis_bands = ("R1_500", "R4_500", "R3_500")
viirs_bands = (5, 4, 3)
buf_dir = Path("./buffer")
fig_dir = Path("./figures")
img_template = "%Y%m%d_%H%M_{sat}_truecolor.png"

def get_modis_product(prod):
    """ Download hdf and generate an image centered on the domain """
    assert len(prod)==2
    global target_latlon
    time, link = prod
    tmp_file = laads.download(link, buf_dir, raw_token=token,
                              replace=True, debug=True)
    #tmp_file = buf_dir.joinpath(Path("20220905_0835.hdf"))
    product = tmp_file.stem.split(".")[0]
    data, info, geo = modis.get_modis_data(tmp_file, bands=modis_bands)
    geo = np.dstack(geo)
    dy, dx = geo_helpers.get_geo_range(
            latlon=geo, target_latlon=target_latlon, dx_px=region_width,
            dy_px=region_height, from_center=True, boundary_error=False,
            debug=True)
    enh = lambda X: enhance.norm_to_uint(enhance.histogram_equalize(
        X, nbins=256)[0], 256, np.uint8)
    data = [ enh(X[dy[0]:dy[1],dx[0]:dx[1]]) for X in data ]
    geo = geo[ dy[0]:dy[1], dx[0]:dx[1] ]
    cy, cx = geo_helpers.get_closest_pixel(geo, target_latlon)
    rgb = guitools.label_at_index(np.dstack(data), (cy,cx), color=(255,0,0))
    geo_plot.generate_raw_image(rgb, fig_dir.joinpath(Path(
        time.strftime(img_template.format(sat=product)))))
    ppt([enhance.array_stat(d) for d in data])
    tmp_file.unlink()

def get_viirs_product(prod):
    time, dlink, glink = prod
    #dfile = laads.download(dlink, buf_dir, raw_token=token,
    #                       replace=True, debug=True)
    #gfile = laads.download(glink, buf_dir, raw_token=token,
    #                       replace=True, debug=True)
    dfile = buf_dir.joinpath("VNP02IMG.A2022250.1036.002.2022251160738.nc")
    gfile = buf_dir.joinpath("VNP03IMG.A2022250.1036.002.2022251152536.nc")
    data, info = viirs.get_viirs_data(dfile, bands=viirs_bands, debug=True)
    lat, lon, _ = viirs.get_viirs_geoloc(gfile, debug=True)
    print(enhance.array_stat(data[0]))
    #sza, _, vza, _ = viirs.get_viirs_sunsat(gfile, debug=True)
    product = dfile.stem.split(".")[0]
    geo = np.dstack((lat, lon))
    dy, dx = geo_helpers.get_geo_range(
            latlon=geo, target_latlon=target_latlon, dx_px=region_width,
            dy_px=region_height, from_center=True, boundary_error=False,
            debug=True)
    data = [ X[dy[0]:dy[1],dx[0]:dx[1]] for X in data ]
    geo = geo[ dy[0]:dy[1], dx[0]:dx[1] ]
    print(data)
    #enh = lambda X: enhance.norm_to_uint(enhance.histogram_equalize(
    #    X, nbins=256, debug=True)[0], 256, np.uint8)
    data = [ enhance.norm_to_uint(X) for X in data ]
    cy, cx = geo_helpers.get_closest_pixel(geo, target_latlon)
    rgb = guitools.label_at_index(np.dstack(data), (cy,cx), color=(255,0,0))
    geo_plot.generate_raw_image(rgb, fig_dir.joinpath(Path(
        time.strftime(img_template.format(sat=product)))))
    ppt([enhance.array_stat(d) for d in data])
    #dfile.unlink()


'''
# Query full time range
modis_products = []
for p in ("MOD09", "MYD09"):
    in_range = modis.query_modis_l2(
            product_key=p,
            start_time=start_time,
            end_time=end_time,
            latlon=target_latlon,
            day_only=True,
            debug=True
            )
    modis_products += [ (f['atime'], f['downloadsLink']) for f in in_range ]
'''

'''
viirs_products = []
# Query full time range of VIIRS products
for p in ("VJ102IMG", "VNP02IMG"):
    in_range = viirs.query_viirs_l1b(
            product_key=p,
            start_time=start_time,
            end_time=end_time,
            latlon=target_latlon,
            debug=True
            )
    viirs_products += [ (f['atime'], f['downloadsLink'], f['geoLink'])
                       for f in in_range if f["illuminations"]=="D"]
'''

#get_modis_product((dt(2022, 9, 5, 8, 35),
#           'https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/archives/MOD09.A2022248.0835.006.2022250030855.hdf'))
get_viirs_product((dt(2022, 9, 7, 10, 36), 'https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/archives/VNP02IMG.A2022250.1036.002.2022251160738.nc', 'https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/archives/VNP03IMG.A2022250.1036.002.2022251152536.nc'))

# Multiprocess downloads and image generation.
#pool = mp.Pool(processes=4)
#pool.map(get_modis_product, modis_products)
#pool.map(get_viirs_product, viirs_products)
