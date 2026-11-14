"""Turn the OSM features + scorecard + elevation into data/derived/course.json.

All coordinates are EPSG:26915 metres. Each hole gets: line of play, blue tee point,
green polygon with front/centre/back, the tee boxes, fairways, bunkers and water that
belong to it, and elevations.
"""
import json

import numpy as np
import pyproj
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from common import DER, FT, RAW, YD, config, dem

T = pyproj.Transformer.from_crs(4326, 26915, always_xy=True)


def xy(e):
    return [T.transform(p["lon"], p["lat"]) for p in e["geometry"]]


def polys(e):
    """Polygons of a way or multipolygon relation."""
    if e["type"] == "way":
        c = xy(e)
        return [Polygon(c)] if len(c) >= 4 else []
    out = []
    for m in e.get("members", []):
        if m.get("role") == "outer" and "geometry" in m and len(m["geometry"]) >= 4:
            out.append(Polygon([T.transform(p["lon"], p["lat"]) for p in m["geometry"]]))
    return out


def coords(g):
    return [[round(x, 2), round(y, 2)] for x, y in g.exterior.coords]


def main():
    cfg = config()
    els = json.loads((RAW / "osm.json").read_text())["elements"]
    D = dem()

    def tagged(k, *vals):
        return [e for e in els if e.get("tags", {}).get(k) in vals]

    hole_ways = tagged("golf", "hole")
    champ = {int(e["tags"]["ref"]): LineString(xy(e)) for e in hole_ways if "handicap" not in e["tags"]}
    other = [LineString(xy(e)) for e in hole_ways if "handicap" in e["tags"]]
    assert sorted(champ) == list(range(1, 19)), sorted(champ)
    all_lines = [(n, l) for n, l in champ.items()] + [(None, l) for l in other]

    def owner(geom, maxd, head=None):
        """Hole whose line of play is nearest to geom (None if it belongs to the 9-hole course)."""
        best = (maxd, None, False)
        for n, l in all_lines:
            ll = l if head is None else LineString([l.interpolate(t * head, normalized=True) for t in np.linspace(0, 1, 12)])
            d = geom.distance(ll)
            if d < best[0]:
                best = (d, n, True)
        return best[1] if best[2] else -1

    greens = [p.buffer(0) for e in tagged("golf", "green") for p in polys(e)]
    tees = [p.buffer(0) for e in tagged("golf", "tee") for p in polys(e)]
    bunkers = [p.buffer(0) for e in tagged("golf", "bunker") for p in polys(e)]
    fairways = [p.buffer(0) for e in tagged("golf", "fairway") for p in polys(e)]
    water = [p.buffer(0) for e in els if e.get("tags", {}).get("natural") == "water"
             or e.get("tags", {}).get("golf") in ("water_hazard", "lateral_water_hazard") for p in polys(e)]
    streams = [LineString(xy(e)) for e in tagged("waterway", "stream", "drain", "ditch")]
    paths = [LineString(xy(e)) for e in tagged("golf", "path", "cartpath") if e["type"] == "way" and len(e["geometry"]) >= 2]
    buildings = [p.buffer(0) for e in els if "building" in e.get("tags", {}) for p in polys(e)]

    holes = []
    for n in range(1, 19):
        line = champ[n]
        pts = list(line.coords)
        end = Point(pts[-1])
        green = min(greens, key=lambda g: g.distance(end))
        gc = green.centroid
        par, yards = cfg["par"][n - 1], cfg["yards"][n - 1]

        def play_len(p):
            path = [(p.x, p.y)] + pts[1:-1] + [(gc.x, gc.y)]
            return LineString(path).length / YD

        # tee boxes on this hole: nearest to the first 45 % of this hole's line
        mine = [t for t in tees if owner(t.centroid, 45, head=0.45) == n]
        if not mine:
            mine = [min(tees, key=lambda t: t.distance(Point(pts[0])))]
        box = min(mine, key=lambda t: abs(play_len(t.centroid) - yards))
        c = box.centroid
        nxt = np.array(pts[1] if len(pts) > 2 else (gc.x, gc.y))
        u = (nxt - (c.x, c.y)) / np.linalg.norm(nxt - (c.x, c.y))
        inner = box.buffer(-1.5) if not box.buffer(-1.5).is_empty else box
        cands = [Point(c.x + t * u[0], c.y + t * u[1]) for t in np.arange(-80, 80.01, 0.5)]
        cands = [p for p in cands if inner.contains(p)] or [c]
        tee = min(cands, key=lambda p: abs(play_len(p) - yards))

        path = [(tee.x, tee.y)] + pts[1:-1] + [(gc.x, gc.y)]
        play = LineString(path)

        # green front / back along the approach direction
        a0 = np.array(path[-2])
        ap = np.array([gc.x, gc.y]) - a0
        ap /= np.linalg.norm(ap)
        g_xy = np.array(green.exterior.coords)
        proj = (g_xy - (gc.x, gc.y)) @ ap
        side = (g_xy - (gc.x, gc.y)) @ np.array([-ap[1], ap[0]])
        front = np.array([gc.x, gc.y]) + proj.min() * ap
        back = np.array([gc.x, gc.y]) + proj.max() * ap

        def feats(items, maxd):
            return [g for g in items if owner(g, maxd) == n]

        h_bunkers = feats(bunkers, 70)
        h_fair = [f for f in fairways if f.intersection(play.buffer(20)).area > 0.25 * min(f.area, play.buffer(20).area)
                  or owner(f.centroid, 40) == n]
        h_fair = [f for f in h_fair if owner(f.centroid, 80) in (n, -1) or f.intersects(play)]
        near = play.buffer(90)
        h_water = [w for w in water if w.intersects(near)]
        h_streams = [s for s in streams if s.intersects(near)]

        def hazard(g, kind):
            v = np.array(g.exterior.coords)
            dt = np.hypot(*(v - (tee.x, tee.y)).T) / YD
            dg = np.hypot(*(v - (gc.x, gc.y)).T) / YD
            return {"kind": kind, "poly": coords(g), "reach": round(float(dt.min())), "carry": round(float(dt.max())),
                    "to_green_near": round(float(dg.min())), "to_green_far": round(float(dg.max())),
                    "greenside": bool(g.distance(green) < 25),
                    "centroid": [round(g.centroid.x, 2), round(g.centroid.y, 2)]}

        z_tee = float(D.sample(tee.x, tee.y))
        z_green = float(D.sample(gc.x, gc.y))
        # elevation profile along the line of play
        s = np.arange(0, play.length + 1, 2.0)
        pp = [play.interpolate(v) for v in s]
        prof = D.sample([p.x for p in pp], [p.y for p in pp])

        measured = play.length / YD
        holes.append({
            "n": n, "par": par, "yards": yards, "handicap": cfg["handicap"][n - 1],
            "measured_yards": round(measured, 1),
            "tee": [round(tee.x, 2), round(tee.y, 2)],
            "line": [[round(x, 2), round(y, 2)] for x, y in path],
            "green": coords(green),
            "green_center": [round(gc.x, 2), round(gc.y, 2)],
            "green_front": [round(float(front[0]), 2), round(float(front[1]), 2)],
            "green_back": [round(float(back[0]), 2), round(float(back[1]), 2)],
            "green_depth_yd": round(float(proj.max() - proj.min()) / YD),
            "green_width_yd": round(float(side.max() - side.min()) / YD),
            "approach_dir": [round(float(ap[0]), 4), round(float(ap[1]), 4)],
            "tee_boxes": [coords(t) for t in mine],
            "blue_box": coords(box),
            "fairways": [coords(f) for f in h_fair],
            "bunkers": [hazard(b, "bunker") for b in h_bunkers],
            "water": [hazard(w, "water") for w in h_water],
            "streams": [[[round(x, 2), round(y, 2)] for x, y in s_.coords] for s_ in h_streams],
            "elev_tee_ft": round(z_tee / FT, 1), "elev_green_ft": round(z_green / FT, 1),
            "elev_change_ft": round((z_green - z_tee) / FT, 1),
            "plays_like": round(yards + (z_green - z_tee) / YD),
            "profile": {"step_m": 2.0, "z_m": [round(float(v), 2) for v in prof]},
        })
        print(f"{n:2d} par {par} blue {yards:3d}  measured {measured:5.0f} ({measured - yards:+4.0f})  "
              f"tees {len(mine)}  fw {len(h_fair)}  bunkers {len(h_bunkers)}  water {len(h_water)}  "
              f"green {holes[-1]['green_depth_yd']}x{holes[-1]['green_width_yd']}  elev {holes[-1]['elev_change_ft']:+.0f} ft")

    out = {
        **{k: cfg[k] for k in ("club", "course", "location", "tee", "rating", "slope")},
        "crs": "EPSG:26915",
        "holes": holes,
        "all": {
            "greens": [coords(g) for g in greens], "tees": [coords(t) for t in tees],
            "bunkers": [coords(b) for b in bunkers], "fairways": [coords(f) for f in fairways],
            "water": [coords(w) for w in water],
            "streams": [[[round(x, 2), round(y, 2)] for x, y in s.coords] for s in streams],
            "paths": [[[round(x, 2), round(y, 2)] for x, y in p.coords] for p in paths],
            "buildings": [coords(b) for b in buildings],
            "nine_lines": [[[round(x, 2), round(y, 2)] for x, y in l.coords] for l in other],
        },
    }
    DER.mkdir(parents=True, exist_ok=True)
    (DER / "course.json").write_text(json.dumps(out))
    tot = sum(h["measured_yards"] for h in holes)
    print(f"total measured {tot:.0f} vs card {sum(cfg['yards'])}")


if __name__ == "__main__":
    main()
