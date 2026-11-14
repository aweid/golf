"""Download USGS 3DEP lidar (Entwine/EPT on AWS) for the project extent and grid it.

Outputs (1 m grid, EPSG:26915, same extent as the DEM):
  data/derived/dsm_1m.npy     highest non-noise return per cell (tree tops, roofs)
  data/derived/ground_1m.npy  mean ground-classified return per cell
  data/derived/lidar_meta.json

Usage: python scripts/fetch_lidar.py [EPT_PROJECT]
"""
import io
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import laspy
import numpy as np
import pyproj

ROOT = Path(__file__).resolve().parent.parent
PROJECT = sys.argv[1] if len(sys.argv) > 1 else "MO_StLouis_2012"
BASE = f"https://s3-us-west-2.amazonaws.com/usgs-lidar-public/{PROJECT}"
UA = {"User-Agent": "forest-hills-yardage-book/0.1"}


def fetch(url, tries=4):
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()
        except Exception:
            if i == tries - 1:
                raise


def main():
    ext = json.loads((ROOT / "data/raw/extent.json").read_text())
    bx0, by0, bx1, by1 = ext["bbox"]
    W, H = bx1 - bx0, by1 - by0
    to_merc = pyproj.Transformer.from_crs(26915, 3857, always_xy=True)
    to_utm = pyproj.Transformer.from_crs(3857, 26915, always_xy=True)
    xs, ys = to_merc.transform([bx0, bx1, bx0, bx1], [by0, by0, by1, by1])
    qx0, qx1, qy0, qy1 = min(xs), max(xs), min(ys), max(ys)

    ept = json.loads(fetch(f"{BASE}/ept.json"))
    b = ept["bounds"]
    size0 = b[3] - b[0]

    def node_bounds(d, x, y):
        s = size0 / (1 << d)
        return b[0] + x * s, b[1] + y * s, b[0] + (x + 1) * s, b[1] + (y + 1) * s

    def overlaps(key):
        d, x, y, _ = key
        x0, y0, x1, y1 = node_bounds(d, x, y)
        return x0 < qx1 and x1 > qx0 and y0 < qy1 and y1 > qy0

    # walk the hierarchy, following sub-hierarchy files (-1) only where they overlap
    nodes = []
    pending = [(0, 0, 0, 0)]
    while pending:
        with ThreadPoolExecutor(16) as ex:
            hs = list(ex.map(lambda k: json.loads(fetch(f"{BASE}/ept-hierarchy/{'-'.join(map(str, k))}.json")), pending))
        nxt = []
        for h in hs:
            for name, n in h.items():
                key = tuple(int(v) for v in name.split("-"))
                if not overlaps(key):
                    continue
                if n == -1:
                    nxt.append(key)
                elif n > 0:
                    nodes.append(key)
        pending = nxt
    nodes = sorted(set(nodes))
    print(f"{PROJECT}: {len(nodes)} nodes, max depth {max(k[0] for k in nodes)}", flush=True)

    dsm = np.full((H, W), -np.inf, np.float32)
    gsum = np.zeros((H, W), np.float64)
    gcnt = np.zeros((H, W), np.int32)
    total = 0

    def load(key):
        raw = fetch(f"{BASE}/ept-data/{'-'.join(map(str, key))}.laz")
        las = laspy.read(io.BytesIO(raw))
        x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
        cls = np.asarray(las.classification)
        m = (x >= qx0) & (x <= qx1) & (y >= qy0) & (y <= qy1)
        return x[m], y[m], z[m], cls[m]

    with ThreadPoolExecutor(24) as ex:
        for i, (x, y, z, cls) in enumerate(ex.map(load, nodes)):
            if i % 200 == 0:
                print(f"  {i}/{len(nodes)} nodes, {total:,} points", flush=True)
            if not len(x):
                continue
            ux, uy = to_utm.transform(x, y)
            c = np.floor(ux - bx0).astype(np.int64)
            r = np.floor(by1 - uy).astype(np.int64)
            ok = (c >= 0) & (c < W) & (r >= 0) & (r < H) & (cls != 7) & (cls != 18)
            c, r, z, cls = c[ok], r[ok], z[ok], cls[ok]
            total += len(z)
            np.maximum.at(dsm, (r, c), z.astype(np.float32))
            g = cls == 2
            np.add.at(gsum, (r[g], c[g]), z[g])
            np.add.at(gcnt, (r[g], c[g]), 1)

    ground = np.where(gcnt > 0, gsum / np.maximum(gcnt, 1), np.nan).astype(np.float32)
    dsm[~np.isfinite(dsm)] = np.nan
    out = ROOT / "data/derived"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "dsm_1m.npy", dsm)
    np.save(out / "ground_1m.npy", ground)
    meta = {"project": PROJECT, "points": int(total), "nodes": len(nodes),
            "density_per_m2": total / (W * H), "dsm_coverage": float(np.isfinite(dsm).mean()),
            "srs": ept.get("srs", {}).get("horizontal"), "vertical": ept.get("srs", {}).get("vertical")}
    (out / "lidar_meta.json").write_text(json.dumps(meta, indent=1))
    print(meta)


if __name__ == "__main__":
    main()
