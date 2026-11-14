"""Derive canopy, tree and building layers from lidar + NAIP.

  data/derived/chm_1m.npy        canopy height (m above ground), gap-filled
  data/derived/treemask_1m.npy   bool, tree canopy
  data/derived/bldg_h_1m.npy     building height above ground (0 elsewhere)
  data/derived/trees.npy         one row per tree: x, y, z_ground, height, crown_radius, r, g, b
"""
import json

import numpy as np
from scipy import ndimage as ndi

from common import DER, RAW, Grid, dem, extent


def main():
    D = dem()
    z = D.a
    dsm = np.load(DER / "dsm_1m.npy")
    chm = np.nan_to_num(dsm - z, nan=0.0).clip(0, 45)
    # the 2012 lidar is leaf-off and speckled: fill crowns
    chm = ndi.grey_closing(chm, size=(5, 5))
    chm = ndi.median_filter(chm, size=3)

    img = np.load(RAW / "naip_rgbn_0.5m.npy").astype(np.float32)
    img1 = img.reshape(img.shape[0] // 2, 2, img.shape[1] // 2, 2, 4).mean(axis=(1, 3))
    r, g, b, nir = (img1[..., k] for k in range(4))
    ndvi = ndi.gaussian_filter((nir - r) / (nir + r + 1e-6), 1.0)
    lum = ndi.gaussian_filter(0.3 * r + 0.59 * g + 0.11 * b, 1.5)

    veg = ndvi > 0.10
    tall = chm > 3.0
    # mown turf is brighter than canopy in summer imagery; drop "trees" that now look like turf
    turf_lum = np.percentile(lum[veg & (chm < 0.5)], 50)
    tree_lum = np.percentile(lum[veg & (chm > 8)], 50)
    thr = 0.35 * tree_lum + 0.65 * turf_lum
    print(f"lum turf {turf_lum:.0f} tree {tree_lum:.0f} thr {thr:.0f}")
    tree = tall & veg & (lum < thr)
    tree = ndi.binary_opening(tree, iterations=1)
    tree = ndi.binary_closing(tree, iterations=2)

    bld = (chm > 2.5) & (ndvi < 0.05) & ~tree
    bld = ndi.binary_opening(bld, iterations=2)
    lab, nlab = ndi.label(bld)
    area = ndi.sum(bld, lab, np.arange(1, nlab + 1))
    # keep solid, building-sized blobs (drops power-line towers and wires); flat roof per building
    idx = np.arange(1, nlab + 1)
    med = ndi.median(chm, lab, idx)
    sl = ndi.find_objects(lab)
    box = np.array([(s[0].stop - s[0].start) * (s[1].stop - s[1].start) for s in sl])
    ok = (area >= 70) & (area / box > 0.38) & (med < 13)
    lut = np.zeros(nlab + 1, np.float32)
    lut[1:] = np.where(ok, np.clip(med, 3, 11), 0)
    bh = lut[lab]
    bld = bh > 0

    # individual trees: canopy peaks (isolated crowns) plus a jittered grid to fill closed canopy
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(7)
    sm = ndi.gaussian_filter(np.where(tree, chm, 0), 1.2)
    peak = (sm == ndi.maximum_filter(sm, size=5)) & tree & (sm > 2.5)
    pr, pc = np.nonzero(peak)
    step = 4.5
    gy, gx = np.meshgrid(np.arange(2, tree.shape[0] - 2, step), np.arange(2, tree.shape[1] - 2, step), indexing="ij")
    gx = gx + (np.arange(gy.shape[0])[:, None] % 2) * step / 2
    gr = (gy + rng.uniform(-1.5, 1.5, gy.shape)).round().astype(int).clip(0, tree.shape[0] - 1).ravel()
    gc_ = (gx + rng.uniform(-1.5, 1.5, gx.shape)).round().astype(int).clip(0, tree.shape[1] - 1).ravel()
    keep = tree[gr, gc_]
    cand = np.column_stack([np.r_[pr, gr[keep]], np.r_[pc, gc_[keep]]]).astype(float)
    # greedy thinning: keep earlier (peak) points, drop anything within 3.6 m of a kept one
    kd = cKDTree(cand)
    alive = np.ones(len(cand), bool)
    for i, nb in enumerate(kd.query_ball_point(cand, 3.6)):
        if alive[i]:
            for j in nb:
                if j > i:
                    alive[j] = False
    rr, cc = cand[alive].astype(int).T
    hmax = ndi.maximum_filter(chm, size=5)
    h = hmax[rr, cc].clip(4, 34) * rng.uniform(0.9, 1.05, len(rr))
    x0, y0, x1, y1 = extent()
    x = x0 + cc + 0.5 + rng.uniform(-0.5, 0.5, len(rr))
    y = y1 - rr - 0.5 + rng.uniform(-0.5, 0.5, len(rr))
    rad = (0.16 * h + 2.0) * rng.uniform(0.85, 1.15, len(h))
    rgbs = ndi.gaussian_filter(img1[..., :3], (2, 2, 0))[rr, cc] / 255.0
    trees = np.column_stack([x, y, z[rr, cc], h, rad, rgbs]).astype(np.float32)

    np.save(DER / "chm_1m.npy", chm.astype(np.float32))
    np.save(DER / "treemask_1m.npy", tree)
    np.save(DER / "bldg_h_1m.npy", bh)
    np.save(DER / "trees.npy", trees)
    print(json.dumps({"trees": len(trees), "tree_cover": float(tree.mean()), "building_cover": float(bld.mean()),
                      "median_tree_h_m": float(np.median(h))}))


if __name__ == "__main__":
    main()
