"""Shared helpers: paths, raster grid, units."""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
DER = ROOT / "data/derived"
OUT = ROOT / "out"
YD = 0.9144  # metres per yard
FT = 0.3048
# which course to build: COURSE=valley python scripts/...  (default: championship)
COURSE = os.environ.get("COURSE", "championship")
SUFFIX = "" if COURSE == "championship" else f"_{COURSE}"


def config():
    return json.loads((ROOT / f"courses/{COURSE}.json").read_text())


def course():
    return json.loads((DER / f"course{SUFFIX}.json").read_text())


def extent():
    return json.loads((RAW / "extent.json").read_text())["bbox"]


class Grid:
    """North-up raster covering the project extent at `res` metres per pixel."""

    def __init__(self, arr, bbox=None, res=None):
        self.a = arr
        self.x0, self.y0, self.x1, self.y1 = bbox or extent()
        self.res = res or (self.x1 - self.x0) / arr.shape[1]

    def rc(self, x, y):
        return (self.y1 - np.asarray(y)) / self.res - 0.5, (np.asarray(x) - self.x0) / self.res - 0.5

    def sample(self, x, y, order=1):
        from scipy.ndimage import map_coordinates
        r, c = self.rc(x, y)
        shp = np.shape(r)
        co = np.vstack([np.ravel(r), np.ravel(c)])
        if self.a.ndim == 2:
            return map_coordinates(self.a, co, order=order, mode="nearest").reshape(shp)
        return np.stack([map_coordinates(self.a[..., k], co, order=order, mode="nearest").reshape(shp)
                         for k in range(self.a.shape[2])], axis=-1)


def dem():
    import tifffile
    return Grid(tifffile.imread(RAW / "dem_1m.tif").astype(np.float32))
