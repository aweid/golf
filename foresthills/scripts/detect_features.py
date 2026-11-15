"""Find greens and bunkers in the aerial imagery for a course whose holes are
mapped in OpenStreetMap only as lines of play (the Valley course).

  COURSE=valley python scripts/detect_features.py  ->  data/derived/detected_valley.json

Greens: the smooth, darker turf patch grown outward from the end of each line of play.
Bunkers: bright, bare, low patches close to a hole and surrounded by turf.
"""
import json

import numpy as np
import pyproj
from scipy import ndimage as ndi
from shapely.geometry import LineString, Point, Polygon

from common import DER, RAW, SUFFIX, config, extent

RES = 0.5
T = pyproj.Transformer.from_crs(4326, 26915, always_xy=True)


def disk(r_m):
    n = int(round(r_m / RES))
    y, x = np.ogrid[-n:n + 1, -n:n + 1]
    return x * x + y * y <= n * n


def outline(mask, r0, c0, x0, y1, simp=0.6):
    """Outer boundary of a boolean window as a simplified polygon in map coordinates."""
    from shapely.geometry import box
    from shapely.ops import unary_union
    rr, cc = np.nonzero(mask)
    runs = []
    for r in np.unique(rr):                      # merge each row's pixels into runs
        c = cc[rr == r]
        brk = np.nonzero(np.diff(c) > 1)[0]
        for a, b in zip(np.r_[0, brk + 1], np.r_[brk, len(c) - 1]):
            runs.append(box(x0 + (c0 + c[a]) * RES, y1 - (r0 + r + 1) * RES, x0 + (c0 + c[b] + 1) * RES, y1 - (r0 + r) * RES))
    g = unary_union(runs).buffer(1.0).buffer(-1.0).simplify(simp)
    if g.geom_type == "MultiPolygon":
        g = max(g.geoms, key=lambda p: p.area)
    return Polygon(g.exterior)


def main():
    cfg = config()
    x0, y0, x1, y1 = extent()
    img = np.load(RAW / "naip_rgbn_0.5m.npy").astype(np.float32)
    chm = np.kron(np.load(DER / "chm_1m.npy"), np.ones((2, 2), np.float32))
    els = json.loads((RAW / "osm.json").read_text())["elements"]
    valley = cfg.get("osm_holes") == "handicap"
    lines = {int(e["tags"]["ref"]): LineString([T.transform(p["lon"], p["lat"]) for p in e["geometry"]])
             for e in els if e.get("tags", {}).get("golf") == "hole" and ("handicap" in e["tags"]) == valley}

    def win(geom, pad):
        a, b, c, d = geom.bounds
        c0, c1 = int((a - pad - x0) / RES), int((c + pad - x0) / RES)
        r0, r1 = int((y1 - d - pad) / RES), int((y1 - b + pad) / RES)
        return r0, r1, c0, c1

    def dist_to(geom, r0, r1, c0, c1):
        """Distance in metres from each window pixel to geom (rasterised)."""
        m = np.ones((r1 - r0, c1 - c0), bool)
        L = geom.length if geom.length else 0
        for s in np.arange(0, L + 0.01, 0.25) if L else [0]:
            p = geom.interpolate(s) if L else geom
            m[int((y1 - p.y) / RES) - r0, int((p.x - x0) / RES) - c0] = False
        return ndi.distance_transform_edt(m) * RES

    out = {"greens": {}, "bunkers": []}
    for n, line in sorted(lines.items()):
        end = Point(line.coords[-1])
        # ---- green: region grow in colour from the end of the line
        r0, r1, c0, c1 = win(end, 34)
        w = ndi.gaussian_filter(img[r0:r1, c0:c1], (1.2, 1.2, 0))
        d = dist_to(end, r0, r1, c0, c1)
        ref = np.median(w[d < 3.0], axis=0)
        # putting surface: darker red, strong near-infrared and greener than the collar around it
        gr, gr0 = w[..., 1] - w[..., 0], ref[1] - ref[0]
        best = None
        for tol in (1.0, 0.8, 0.65, 0.5):
            m = ((np.abs(w[..., 0] - ref[0]) < 9 * tol) & (w[..., 3] > ref[3] - 10 * tol) & (gr > gr0 - 3.5 * tol)
                 & (d < 30) & (chm[r0:r1, c0:c1] < 1.5))
            m = ndi.binary_opening(m, disk(1.5))
            lab, _ = ndi.label(m)
            k = lab[d.argmin() // d.shape[1], d.argmin() % d.shape[1]]
            if not k:
                continue
            m = ndi.binary_closing(ndi.binary_fill_holes(lab == k), disk(1.5))
            area, reach = m.sum() * RES * RES, d[m].max()
            best = (tol, m, area)
            if area < 950 and reach < 28:
                break
        thr, m, area = best
        g = outline(m, r0, c0, x0, y1)
        g = Polygon(g.buffer(-2.5).buffer(4.5).buffer(-2.0).simplify(0.3).exterior)      # round off the ragged edge
        out["greens"][n] = [[round(x, 2), round(y, 2)] for x, y in g.exterior.coords]

        # ---- bunkers: bright bare sand within 55 m of the hole
        r0, r1, c0, c1 = win(line, 60)
        w = img[r0:r1, c0:c1]
        r_, g_, b_, nir = (w[..., k] for k in range(4))
        ndvi = (nir - r_) / (nir + r_ + 1e-6)
        lum = (r_ + g_ + b_) / 3
        d = dist_to(line, r0, r1, c0, c1)
        # sand is brighter than concrete cart paths and reflects more near-infrared
        sand = (lum > 192) & (nir > 138) & (ndvi < 0.0) & (chm[r0:r1, c0:c1] < 1.0) & (d < 55)
        sand = ndi.binary_opening(sand, disk(1.5))
        lab, nl = ndi.label(sand)
        for k in range(1, nl + 1):
            mk = lab == k
            a = mk.sum() * RES * RES
            if not 12 <= a <= 500:
                continue
            ring = ndi.binary_dilation(mk, disk(6.0)) & ~ndi.binary_dilation(mk, disk(2.5))
            turf = ((ndvi > 0.02) & (chm[r0:r1, c0:c1] < 1.0))[ring].mean()
            if turf < 0.62:      # beside a road, drive or house
                continue
            p = outline(ndi.binary_fill_holes(ndi.binary_dilation(mk, disk(1.0))), r0, c0, x0, y1, 0.4)
            if 4 * np.pi * p.area / p.length ** 2 < 0.45:     # long thin: a path, not a bunker
                continue
            if any(Polygon(q).intersects(p) for q in out["bunkers"]):
                continue
            out["bunkers"].append([[round(x, 2), round(y, 2)] for x, y in p.exterior.coords])

        print(f"{n}: green {g.area:.0f} m2 (thr {thr:.2f})  bunkers so far {len(out['bunkers'])}")

    (DER / f"detected{SUFFIX}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
