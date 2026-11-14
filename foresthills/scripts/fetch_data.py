"""Download the public source data for the course. Skips files that already exist.

  data/raw/osm.json            OpenStreetMap golf features, water, buildings (Overpass)
  data/raw/extent.json         project extent, EPSG:26915 (UTM 15N), metres
  data/raw/dem_1m.tif          USGS 3DEP bare-earth elevation, 1 m
  data/raw/naip_rgbn_0.5m.npy  USDA NAIP aerial imagery (R,G,B,NIR), 0.5 m
  data/raw/ctx_dem_5m.tif      surrounding terrain for the video horizon, 5 m
  data/raw/ctx_rgb_2m.jpg      surrounding imagery, 2 m

Lidar is separate: scripts/fetch_lidar.py
"""
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pyproj
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
UA = {"User-Agent": "forest-hills-yardage-book/0.1"}
CLUB_RELATION = 13333847  # Forest Hills Country Club, Clarkson Valley MO
MARGIN = 300              # metres around the club boundary
CTX_MARGIN = 2400         # metres of low-res surroundings around the extent
OVERPASS = ["https://overpass.kumi.systems/api/interpreter",
            "https://overpass.private.coffee/api/interpreter",
            "https://overpass-api.de/api/interpreter"]
DEM = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
NAIP = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"


def http(url, params=None, data=None, timeout=300):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA, data=data)
    for attempt in range(5):
        try:
            return urllib.request.urlopen(req, timeout=timeout).read()
        except Exception as e:
            if attempt == 4:
                raise
            print("   retry", e)
            time.sleep(3 * (attempt + 1))


def overpass(q):
    for host in OVERPASS:
        try:
            body = http(host, data=urllib.parse.urlencode({"data": q}).encode())
            if body[:1] == b"{":
                return json.loads(body)
        except Exception as e:
            print("  ", host, e)
        time.sleep(2)
    raise SystemExit("all Overpass servers failed")


def export(url, bbox, size, out, **extra):
    x0, y0, x1, y1 = bbox
    p = dict(bbox=f"{x0},{y0},{x1},{y1}", bboxSR=26915, imageSR=26915, size=f"{size[0]},{size[1]}",
             format="tiff", interpolation="RSP_BilinearInterpolation", f="image", **extra)
    out.write_bytes(http(url, p))
    return tifffile.imread(out)


def mosaic(url, bbox, res, tag, **extra):
    """Fetch bbox at res metres/pixel in <=4000 px tiles; returns the stitched array."""
    x0, y0, x1, y1 = bbox
    nx = math.ceil((x1 - x0) / res / 2000)
    ny = math.ceil((y1 - y0) / res / 2000)
    xs = np.linspace(x0, x1, nx + 1).round().astype(int)
    ys = np.linspace(y1, y0, ny + 1).round().astype(int)
    rows = []
    for j in range(ny):
        row = []
        for i in range(nx):
            b = (xs[i], ys[j + 1], xs[i + 1], ys[j])
            size = (round((b[2] - b[0]) / res), round((b[3] - b[1]) / res))
            row.append(export(url, b, size, RAW / f"_{tag}_{j}{i}.tif", **extra))
            (RAW / f"_{tag}_{j}{i}.tif").unlink()
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    T = pyproj.Transformer.from_crs(4326, 26915, always_xy=True)
    Ti = pyproj.Transformer.from_crs(26915, 4326, always_xy=True)

    ext_f = RAW / "extent.json"
    if not ext_f.exists():
        b = overpass(f"[out:json];rel({CLUB_RELATION});out bb;")["elements"][0]["bounds"]
        x0, y0 = T.transform(b["minlon"], b["minlat"])
        x1, y1 = T.transform(b["maxlon"], b["maxlat"])
        bbox = [math.floor((x0 - MARGIN) / 100) * 100, math.floor((y0 - MARGIN) / 100) * 100,
                math.ceil((x1 + MARGIN) / 100) * 100, math.ceil((y1 + MARGIN) / 100) * 100]
        ext_f.write_text(json.dumps({"crs": "EPSG:26915", "bbox": bbox}))
    bbox = json.loads(ext_f.read_text())["bbox"]
    print("extent", bbox)

    f = RAW / "osm.json"
    if not f.exists():
        lon0, lat0 = Ti.transform(bbox[0] + 150, bbox[1] + 150)
        lon1, lat1 = Ti.transform(bbox[2] - 150, bbox[3] - 150)
        bb = f"({lat0:.5f},{lon0:.5f},{lat1:.5f},{lon1:.5f})"
        q = (f'[out:json][timeout:120];(rel({CLUB_RELATION});nwr["golf"]{bb};'
             f'nwr["natural"~"water|wood|tree|tree_row"]{bb};nwr["landuse"="forest"]{bb};'
             f'way["waterway"]{bb};way["building"]{bb};way["highway"]{bb};);out geom;')
        f.write_text(json.dumps(overpass(q)))
        print("osm ok")

    f = RAW / "dem_1m.tif"
    if not f.exists():
        a = export(DEM, bbox, (bbox[2] - bbox[0], bbox[3] - bbox[1]), f, pixelType="F32")
        print("dem", a.shape, a.min(), a.max())

    f = RAW / "naip_rgbn_0.5m.npy"
    if not f.exists():
        img = mosaic(NAIP, bbox, 0.5, "naip", pixelType="U8")
        np.save(f, img)
        Image.fromarray(img[::4, ::4, :3]).save(RAW / "naip_preview.jpg", quality=85)
        print("naip", img.shape)

    cb = [bbox[0] - CTX_MARGIN, bbox[1] - CTX_MARGIN, bbox[2] + CTX_MARGIN, bbox[3] + CTX_MARGIN]
    f = RAW / "ctx_dem_5m.tif"
    if not f.exists():
        a = export(DEM, cb, ((cb[2] - cb[0]) // 5, (cb[3] - cb[1]) // 5), f, pixelType="F32")
        print("ctx dem", a.shape)
    f = RAW / "ctx_rgb_2m.jpg"
    if not f.exists():
        img = mosaic(NAIP, cb, 2.0, "ctx", pixelType="U8")
        Image.fromarray(img[..., :3]).save(f, quality=90)
        print("ctx rgb", img.shape)
    (RAW / "ctx_extent.json").write_text(json.dumps({"crs": "EPSG:26915", "bbox": cb}))


if __name__ == "__main__":
    main()
