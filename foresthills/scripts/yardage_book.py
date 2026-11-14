"""Build the pocket yardage book: out/yardage_book.pdf (4.25 x 7 in pages).

Per hole: a hole map (tee at the bottom, distances from the blue tee in black, distances
to the centre of the green in red, 5 ft contours, elevation profile) and a green page
(1 ft contours, fall arrows, depth lines every 5 yards from the front edge).

  python scripts/yardage_book.py            # whole book
  python scripts/yardage_book.py --holes 7  # just these holes (for checking)
  python scripts/yardage_book.py --png      # also write out/pages/*.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon as MPoly
from scipy import ndimage as ndi
from shapely.geometry import LineString, Point, Polygon

from common import DER, FT, OUT, YD, Grid, course, dem

PAGE = (4.25, 7.0)
INK = "#1d2a22"
RED = "#c0271f"
COL = dict(rough="#dbe8c6", trees="#9dbf8a", tree_edge="#6f9a63", fairway="#b9dc9b", green="#8fd18a",
           green_edge="#3f8f4a", tee="#a9d6a0", blue="#1f4fd6", sand="#f3e6b8", sand_edge="#b9a66a",
           water="#9cc9ea", water_edge="#4f8fc0", path="#c9b99a", bldg="#b9b3aa", contour="#8a7a5c")
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7,
                     "pdf.fonttype": 42, "axes.linewidth": 0.6})


class Frame:
    """Local frame: origin o, +Y along `up`, +X to the right of it. Units metres."""

    def __init__(self, o, up):
        self.o = np.asarray(o, float)
        self.u = np.asarray(up, float) / np.linalg.norm(up)
        self.r = np.array([self.u[1], -self.u[0]])

    def to(self, p):
        d = np.atleast_2d(np.asarray(p, float)) - self.o
        return np.column_stack([d @ self.r, d @ self.u])

    def world(self, X, Y):
        return (self.o[0] + X * self.r[0] + Y * self.u[0], self.o[1] + X * self.r[1] + Y * self.u[1])

    @property
    def north(self):
        return np.array([self.r[1], self.u[1]])


def raster(grid, fr, xlim, ylim, step, order=1):
    xs = np.arange(xlim[0], xlim[1] + step, step)
    ys = np.arange(ylim[0], ylim[1] + step, step)
    X, Y = np.meshgrid(xs, ys)
    wx, wy = fr.world(X, Y)
    return xs, ys, grid.sample(wx, wy, order=order)


def draw_base(ax, C, fr, xlim, ylim, L, detail=False):
    """Trees, contours and every course feature that falls in the window."""
    ax.set_facecolor(COL["rough"])
    step = 0.5 if detail else 1.0
    xs, ys, tm = raster(L["trees"], fr, xlim, ylim, step)
    ax.contourf(xs, ys, tm, levels=[0.5, 2], colors=[COL["trees"]], zorder=1)
    ax.contour(xs, ys, tm, levels=[0.5], colors=[COL["tree_edge"]], linewidths=0.35, zorder=1)
    win = Polygon([fr.world(xlim[0], ylim[0]), fr.world(xlim[1], ylim[0]), fr.world(xlim[1], ylim[1]),
                   fr.world(xlim[0], ylim[1])]).buffer(20)

    def add(polys, fc, ec, z, lw=0.5):
        for p in polys:
            if Polygon(p).intersects(win):
                ax.add_patch(MPoly(fr.to(p), closed=True, fc=fc, ec=ec, lw=lw, zorder=z))

    A = C["all"]
    add(A["buildings"], COL["bldg"], "#8d877e", 2, 0.3)
    add(A["water"], COL["water"], COL["water_edge"], 2)
    for s in A["streams"]:
        if LineString(s).intersects(win):
            q = fr.to(s)
            ax.plot(q[:, 0], q[:, 1], color=COL["water_edge"], lw=0.9, zorder=2)
    add(A["fairways"], COL["fairway"], "#8fbf74", 3, 0.4)
    for s in A["paths"]:
        if LineString(s).intersects(win):
            q = fr.to(s)
            ax.plot(q[:, 0], q[:, 1], color=COL["path"], lw=1.1 if detail else 0.8, zorder=4, solid_capstyle="round")
    add(A["tees"], COL["tee"], "#6fae68", 5, 0.4)
    add(A["greens"], COL["green"], COL["green_edge"], 5, 0.7)
    add(A["bunkers"], COL["sand"], COL["sand_edge"], 6, 0.5)


def setup(ax, xlim, ylim):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#5d6b60")


def north_arrow(ax, fr, xlim, ylim, corner="ur"):
    n = fr.north
    size = (ylim[1] - ylim[0]) * 0.035
    cx = xlim[1] - size * 1.6 if "r" in corner else xlim[0] + size * 1.6
    cy = ylim[1] - size * 1.8
    ax.annotate("", xy=(cx + n[0] * size, cy + n[1] * size), xytext=(cx - n[0] * size, cy - n[1] * size),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=7), zorder=20)
    ax.text(cx + n[0] * size * 1.55, cy + n[1] * size * 1.55, "N", ha="center", va="center", fontsize=5.5,
            color=INK, zorder=20, weight="bold")


def label(ax, x, y, s, color=INK, size=6.2, ha="center", va="center", weight="bold", z=30):
    ax.text(x, y, s, color=color, fontsize=size, ha=ha, va=va, weight=weight, zorder=z,
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec=color, lw=0.4, alpha=0.93))


def header(fig, h, C, sub):
    fig.text(0.06, 0.957, f"{h['n']}", fontsize=30, weight="bold", color=INK, va="center")
    x = 0.06 + (0.105 if h["n"] < 10 else 0.185)
    fig.text(x, 0.968, f"PAR {h['par']}   ·   {h['yards']} YDS", fontsize=10.5, weight="bold", color=INK, va="center")
    fig.text(x, 0.944, sub, fontsize=6.5, color="#3d4a42", va="center")
    fig.text(0.94, 0.968, f"HCP {h['handicap']}", fontsize=8, weight="bold", color=INK, ha="right", va="center")
    fig.text(0.80, 0.968, f"{C['tee'].upper()}", fontsize=8, color=COL["blue"], ha="right", va="center", weight="bold")
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.926, 0.926], color=INK, lw=0.8))


def footer(fig, C, page):
    fig.text(0.06, 0.022, f"{C['club']} · {C['course']}", fontsize=5.2, color="#66736a")
    fig.text(0.94, 0.022, str(page), fontsize=5.2, color="#66736a", ha="right")


# ---------------------------------------------------------------- hole page

def point_at_distance_to_green(play, gc, d_m):
    """Point on the line of play, nearest the green, whose straight distance to the green centre is d."""
    s = np.arange(play.length, 0, -0.5)
    pts = np.array([play.interpolate(v).coords[0] for v in s])
    dist = np.hypot(*(pts - gc).T)
    i = np.argmax(dist >= d_m)
    if dist[i] < d_m:
        return None
    return pts[i], s[i]


def tick(ax, fr, play, s, half=11.0):
    p = np.array(play.interpolate(s).coords[0])
    q = np.array(play.interpolate(min(s + 1, play.length)).coords[0])
    if np.allclose(p, q):
        q = p + (p - np.array(play.interpolate(s - 1).coords[0]))
    d = (q - p) / np.linalg.norm(q - p)
    nrm = np.array([-d[1], d[0]])
    a, b = fr.to([p - nrm * half, p + nrm * half])
    return a, b


def hole_page(pdf, C, h, L, page):
    fig = plt.figure(figsize=PAGE)
    el = h["elev_change_ft"]
    sub = f"{'Uphill' if el > 0 else 'Downhill'} {abs(el):.0f} ft   ·   plays {h['plays_like']}   ·   green {h['green_depth_yd']} deep × {h['green_width_yd']} wide"
    header(fig, h, C, sub)
    footer(fig, C, page)

    tee, gc = np.array(h["tee"]), np.array(h["green_center"])
    fr = Frame(tee, gc - tee)
    play = LineString(h["line"])
    box = [0.06, 0.205, 0.88, 0.705]                               # axes rect in figure fractions
    aw, ah = box[2] * PAGE[0], box[3] * PAGE[1]
    pts = np.vstack([fr.to(h["line"]), fr.to(h["green"])] + [fr.to(f) for f in h["fairways"]]
                    + [fr.to(b["poly"]) for b in h["bunkers"]])
    ylo, yhi = pts[:, 1].min() - 22, pts[:, 1].max() + 22
    xlo, xhi = pts[:, 0].min() - 28, pts[:, 0].max() + 28
    scale = max((yhi - ylo) / ah, (xhi - xlo) / aw)                # metres per inch
    cx, cy = (xlo + xhi) / 2, (ylo + yhi) / 2
    xlim = (cx - scale * aw / 2, cx + scale * aw / 2)
    ylim = (cy - scale * ah / 2, cy + scale * ah / 2)
    ax = fig.add_axes(box)
    draw_base(ax, C, fr, xlim, ylim, L)

    # 5 ft contours
    xs, ys, z = raster(L["dem_s"], fr, xlim, ylim, 2.0)
    zf = z / FT
    lev = np.arange(np.floor(zf.min() / 5) * 5, zf.max() + 5, 5)
    ax.contour(xs, ys, zf, levels=lev, colors=[COL["contour"]], linewidths=0.25, alpha=0.75, zorder=7)
    ax.contour(xs, ys, zf, levels=lev[lev % 25 == 0], colors=[COL["contour"]], linewidths=0.55, alpha=0.8, zorder=7)

    ln = fr.to(h["line"])
    ax.plot(ln[:, 0], ln[:, 1], color="white", lw=1.6, zorder=8, alpha=0.7)
    ax.plot(ln[:, 0], ln[:, 1], color=INK, lw=0.6, ls=(0, (4, 3)), zorder=9)
    ax.add_patch(MPoly(fr.to(h["blue_box"]), closed=True, fc=COL["blue"], ec="white", lw=0.4, zorder=10, alpha=0.85))
    t0 = fr.to(tee)[0]
    ax.plot(*t0, "o", ms=2.6, color="white", mec=COL["blue"], mew=0.8, zorder=11)
    g0 = fr.to(gc)[0]
    ax.plot(*g0, "o", ms=2.2, color=RED, zorder=11)

    side_of = lambda p: 1 if fr.to(p)[0][0] >= np.interp(fr.to(p)[0][1], ln[:, 1], ln[:, 0]) else -1
    if h["par"] > 3:
        # red: yards to the centre of the green
        red_s = []
        for d in (100, 150, 200, 250):
            r = point_at_distance_to_green(play, gc, d * YD)
            if r is None or r[1] < 110 * YD:
                continue
            red_s.append(r[1])
            a, b = tick(ax, fr, play, r[1])
            ax.plot([a[0], b[0]], [a[1], b[1]], color=RED, lw=1.1, zorder=12)
            label(ax, b[0] + scale * 0.09, b[1], str(d), RED, ha="left")
        # black: carry from the blue tee
        for d in (200, 225, 250, 275):
            s = d * YD
            if s > play.length - 70 * YD:
                continue
            p = np.array(play.interpolate(s).coords[0])
            # distance along the line of play ~ straight carry until the dogleg; use straight distance
            ss = np.arange(0, play.length, 0.5)
            pp = np.array([play.interpolate(v).coords[0] for v in ss])
            i = np.argmax(np.hypot(*(pp - tee).T) >= s)
            if i == 0 or any(abs(ss[i] - r_) < 13 for r_ in red_s):
                continue
            a, b = tick(ax, fr, play, ss[i], half=7)
            ax.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=0.9, zorder=12)
            ax.text(a[0] - scale * 0.05, a[1], str(d), color=INK, fontsize=5.4, ha="right", va="center", zorder=30,
                    weight="bold", bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))

    # hazards
    corridor = play.buffer(32)
    for hz in h["bunkers"] + h["water"]:
        poly = Polygon(hz["poly"]).buffer(0)
        part = poly if hz["kind"] == "bunker" else poly.intersection(corridor)
        if part.is_empty or hz["greenside"]:
            continue
        geoms = [part] if part.geom_type == "Polygon" else [g for g in getattr(part, "geoms", []) if g.geom_type == "Polygon"]
        if not geoms:
            continue
        v = np.vstack([np.array(g.exterior.coords) for g in geoms])
        dt = np.hypot(*(v - tee).T) / YD
        dg = np.hypot(*(v - gc).T) / YD
        c = np.array(part.centroid.coords[0])
        sgn = side_of(c)
        lv = fr.to(v)
        x_out = lv[:, 0].max() + scale * 0.07 if sgn > 0 else lv[:, 0].min() - scale * 0.07
        ha = "left" if sgn > 0 else "right"
        ymid = fr.to(c)[0][1]
        if h["par"] > 3 and 140 <= dt.min() <= 330:
            label(ax, x_out, ymid + scale * 0.07, f"{dt.min():.0f}–{dt.max():.0f}", INK, ha=ha, size=5.6)
            if dg.min() > 35:
                label(ax, x_out, ymid - scale * 0.07, f"{dg.min():.0f}", RED, ha=ha, size=5.6)
        elif dg.min() > 35 and h["par"] > 3:
            label(ax, x_out, ymid, f"{dg.min():.0f}", RED, ha=ha, size=5.6)
        elif h["par"] == 3:
            label(ax, x_out, ymid, f"{dt.min():.0f}–{dt.max():.0f}", INK, ha=ha, size=5.6)

    # streams crossing the line of play
    for st_ in h["streams"]:
        x = LineString(st_).intersection(play)
        if x.is_empty:
            continue
        for p in ([x] if x.geom_type == "Point" else [g_ for g_ in getattr(x, "geoms", []) if g_.geom_type == "Point"]):
            p = np.array(p.coords[0])
            dt_, dg_ = np.hypot(*(p - tee)) / YD, np.hypot(*(p - gc)) / YD
            q = fr.to(p)[0]
            if dt_ > 60 and dg_ > 35:
                label(ax, q[0] - scale * 0.1, q[1] + scale * 0.05, f"creek {dt_:.0f}", COL["water_edge"], ha="right", size=5.4)
                if h["par"] > 3:
                    label(ax, q[0] + scale * 0.1, q[1] + scale * 0.05, f"{dg_:.0f}", RED, ha="left", size=5.4)

    # green front / centre / back from the tee on par 3s; otherwise from the last corner
    if h["par"] == 3:
        for key, name in (("green_back", "B"), ("green_center", "C"), ("green_front", "F")):
            p = np.array(h[key])
            q = fr.to(p)[0]
            d = np.hypot(*(p - tee)) / YD
            label(ax, xlim[1] - scale * 0.12, q[1] + {"B": 1, "C": 0, "F": -1}[name] * scale * 0.14, f"{name} {d:.0f}", INK, ha="right", size=6.5)
    setup(ax, xlim, ylim)
    north_arrow(ax, fr, xlim, ylim, "ul" if g0[0] > cx else "ur")
    # scale bar, 50 yards
    bx, by = xlim[0] + scale * 0.12, ylim[0] + scale * 0.14
    ax.plot([bx, bx + 50 * YD], [by, by], color=INK, lw=1.4, zorder=30, solid_capstyle="butt")
    ax.text(bx + 25 * YD, by + scale * 0.05, "50 yds", fontsize=5, ha="center", va="bottom", color=INK, zorder=30)

    # elevation profile
    pa = fig.add_axes([0.11, 0.065, 0.83, 0.10])
    zp = np.array(h["profile"]["z_m"]) / FT
    sp = np.arange(len(zp)) * h["profile"]["step_m"] / YD
    base = zp.min() - 6
    pa.fill_between(sp, zp, base, color="#c7dcae", lw=0)
    pa.plot(sp, zp, color="#4d7a45", lw=1.0)
    pa.plot([sp[0], sp[-1]], [zp[0], zp[-1]], "o", ms=3, color=INK)
    pa.text(sp[0], zp[0] + 2, f" tee {zp[0]:.0f}'", fontsize=5.4, va="bottom", ha="left", color=INK)
    pa.text(sp[-1], zp[-1] + 2, f"green {zp[-1]:.0f}' ", fontsize=5.4, va="bottom", ha="right", color=INK)
    pa.set_xlim(0, sp[-1])
    pa.set_ylim(base, zp.max() + 14)
    pa.set_yticks([])
    pa.tick_params(axis="x", labelsize=5, length=2, pad=1.5)
    for s in ("top", "right", "left"):
        pa.spines[s].set_visible(False)
    fig.text(0.06, 0.178, "ELEVATION ALONG THE LINE OF PLAY (yards from tee, feet above sea level)", fontsize=4.8, color="#66736a")
    fig.text(0.94, 0.19, "black: from blue tee    red: to centre of green", fontsize=4.8, color="#66736a", ha="right")
    pdf.savefig(fig)
    return fig


# ---------------------------------------------------------------- green page

def clock(vec, fr):
    """Describe a downhill direction in the approach frame."""
    x, y = vec @ fr.r, vec @ fr.u
    ang = np.degrees(np.arctan2(x, y)) % 360                       # 0 = toward the back, 90 = right
    names = ["back", "back-right", "right", "front-right", "front", "front-left", "left", "back-left"]
    return names[int(((ang + 22.5) % 360) // 45)]


def green_page(pdf, C, h, L, page):
    fig = plt.figure(figsize=PAGE)
    gc = np.array(h["green_center"])
    fr = Frame(gc, h["approach_dir"])
    gpoly = Polygon(h["green"]).buffer(0)
    g = fr.to(h["green"])
    half = max(np.abs(g).max() + 13, 26)
    box = [0.06, 0.34, 0.88, 0.88 * PAGE[0] / PAGE[1]]
    xlim, ylim = (-half, half), (-half, half)

    # surface: smoothed 1 m lidar ground
    xs, ys, z = raster(L["dem_g"], fr, xlim, ylim, 0.5)
    zf = z / FT
    X, Y = np.meshgrid(xs, ys)
    wx, wy = fr.world(X, Y)
    from shapely import contains_xy
    inside = contains_xy(gpoly, wx, wy)
    zin = zf[inside]
    A = np.column_stack([X[inside], Y[inside], np.ones(inside.sum())])
    coef, *_ = np.linalg.lstsq(A, zin * FT, rcond=None)            # metres per metre
    slope = np.hypot(coef[0], coef[1]) * 100
    down_local = -np.array([coef[0], coef[1]]) / max(np.hypot(coef[0], coef[1]), 1e-9)
    down_world = down_local[0] * fr.r + down_local[1] * fr.u
    rng_ft = zin.max() - zin.min()

    sub = (f"{h['green_depth_yd']} deep × {h['green_width_yd']} wide   ·   falls {slope:.1f}% to the {clock(down_world, fr)}"
           f"   ·   {rng_ft:.1f} ft high to low")
    header(fig, h, C, sub)
    footer(fig, C, page)
    ax = fig.add_axes(box)
    draw_base(ax, C, fr, xlim, ylim, L, detail=True)

    # relief tint inside the green, 1 ft contours everywhere, 0.5 ft inside
    zrel = zf - zin.min()
    tint = ax.imshow(zrel, extent=(xs[0], xs[-1], ys[0], ys[-1]), origin="lower", cmap="YlGn_r", alpha=0.85, zorder=5.5,
              vmin=-0.3 * rng_ft, vmax=rng_ft * 1.05, interpolation="bilinear")
    edge = ax.add_patch(MPoly(g, closed=True, fc="none", ec=COL["green_edge"], lw=1.0, zorder=9))
    tint.set_clip_path(edge)
    lev = np.arange(np.floor(zf.min()), zf.max() + 1, 1.0)
    cs = ax.contour(xs, ys, zf, levels=lev, colors=[COL["contour"]], linewidths=0.45, zorder=7)
    ax.clabel(cs, levels=lev[::2], fmt="%.0f", fontsize=4.2, inline_spacing=1)
    zi = np.where(ndi.binary_dilation(inside, iterations=2), zf, np.nan)
    ax.contour(xs, ys, zi, levels=np.arange(np.floor(zf.min()), zf.max() + 1, 0.5), colors=["#5a4e36"], linewidths=0.3,
               linestyles="dotted", zorder=7)

    # fall arrows every 5 m inside the green
    gy_, gx_ = np.gradient(z, 0.5)                                 # dz/dY, dz/dX in m per m
    st = 10
    for i in range(st // 2, len(ys), st):
        for j in range(st // 2, len(xs), st):
            if not inside[i, j]:
                continue
            sx, sy = -gx_[i, j], -gy_[i, j]
            m = np.hypot(sx, sy)
            if m < 0.004:
                continue
            ln = 1.6 + min(m, 0.08) * 38
            col = "#2b6cb0" if m < 0.02 else ("#b7791f" if m < 0.04 else RED)
            ax.annotate("", xy=(xs[j] + sx / m * ln, ys[i] + sy / m * ln), xytext=(xs[j], ys[i]),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=0.55, mutation_scale=4.5), zorder=10)

    # depth lines every 5 yards from the front edge
    yf, yb = g[:, 1].min(), g[:, 1].max()
    for k in range(5, int((yb - yf) / YD) + 1, 5):
        yy = yf + k * YD
        seg = gpoly.intersection(LineString([fr.world(-half, yy), fr.world(half, yy)]))
        if seg.is_empty:
            continue
        q = fr.to(np.array(seg.coords if seg.geom_type == "LineString" else
                           np.vstack([np.array(s_.coords) for s_ in seg.geoms])))
        ax.plot([q[:, 0].min(), q[:, 0].max()], [yy, yy], color="white", lw=0.5, ls=(0, (2, 2)), zorder=8)
        ax.text(q[:, 0].max() + 0.8, yy, f"{k}", fontsize=4.6, color=INK, va="center", ha="left", zorder=12)
    ax.plot(0, 0, "+", ms=5, color=RED, mew=0.9, zorder=12)

    # greenside bunkers: yards from the centre of the green
    for b in h["bunkers"]:
        if b["greenside"]:
            q = fr.to(b["centroid"])[0]
            if abs(q[0]) < half - 3 and abs(q[1]) < half - 3:
                d = np.hypot(*q) / YD
                ax.text(q[0], q[1], f"{d:.0f}", fontsize=5, color="#6b5a22", ha="center", va="center", zorder=12, weight="bold")

    setup(ax, xlim, ylim)
    north_arrow(ax, fr, xlim, ylim)
    ax.annotate("", xy=(0, yf - 2), xytext=(0, -half + 2.5),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, ls="--", mutation_scale=7), zorder=12)
    ax.text(1.2, -half + 3, "line of play", fontsize=4.8, color=INK, ha="left", va="bottom", zorder=12)
    bx, by = -half + 2.5, -half + 2.5
    ax.plot([bx, bx + 10 * YD], [by, by], color=INK, lw=1.4, zorder=30, solid_capstyle="butt")
    ax.text(bx + 5 * YD, by + 0.8, "10 yds", fontsize=5, ha="center", va="bottom", color=INK, zorder=30)

    # numbers
    tee = np.array(h["tee"])
    rows = []
    if h["par"] == 3:
        f, c, b = (np.hypot(*(np.array(h[k]) - tee)) / YD for k in ("green_front", "green_center", "green_back"))
        rows.append(("From the blue tee", f"front {f:.0f}   ·   centre {c:.0f}   ·   back {b:.0f}"))
        rows.append(("Plays like", f"{h['plays_like']} to the centre ({h['elev_change_ft']:+.0f} ft)"))
    else:
        rows.append(("Front / back of centre", f"front −{np.hypot(*(np.array(h['green_front']) - gc)) / YD:.0f}   ·   back +{np.hypot(*(np.array(h['green_back']) - gc)) / YD:.0f} yards"))
        rows.append(("Approach", f"green is {h['elev_change_ft']:+.0f} ft from the tee; add or take 1 yard per 3 ft"))
    gs = [b for b in h["bunkers"] if b["greenside"]]
    if gs:
        desc = []
        for b in gs:
            q = fr.to(b["centroid"])[0]
            where = ("front" if q[1] < -6 else "back" if q[1] > 6 else "") + ("-" if abs(q[1]) > 6 and abs(q[0]) > 5 else "") + ("left" if q[0] < -5 else "right" if q[0] > 5 else "")
            desc.append(where or "centre")
        rows.append(("Greenside bunkers", ", ".join(sorted(set(desc), key=desc.index))))
    y = 0.305
    for k, v in rows:
        fig.text(0.06, y, k.upper(), fontsize=5, color="#66736a", va="center")
        fig.text(0.36, y, v, fontsize=6.4, color=INK, va="center")
        y -= 0.027
    fig.text(0.06, y - 0.004, "Arrows show the fall: blue under 2%, amber 2–4%, red over 4%. Contours 1 ft (dotted 6 in).",
             fontsize=4.8, color="#66736a", va="center")
    y -= 0.04
    fig.text(0.06, y, "NOTES", fontsize=5, color="#66736a")
    while y > 0.07:
        y -= 0.032
        fig.add_artist(plt.Line2D([0.06, 0.94], [y, y], color="#b9c2bb", lw=0.4))
    pdf.savefig(fig)
    return fig, dict(slope=slope, toward=clock(down_world, fr))


# ---------------------------------------------------------------- front matter

def cover(pdf, C):
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("#173a2a")
    tot = sum(h["yards"] for h in C["holes"])
    fig.text(0.5, 0.66, "FOREST HILLS", fontsize=27, weight="bold", color="white", ha="center")
    fig.text(0.5, 0.615, "COUNTRY CLUB", fontsize=15, color="white", ha="center")
    fig.add_artist(plt.Line2D([0.3, 0.7], [0.585, 0.585], color="#c9b458", lw=1.2))
    fig.text(0.5, 0.545, C["course"], fontsize=12, color="#e7efe9", ha="center")
    fig.text(0.5, 0.515, C["location"], fontsize=8, color="#b9cdbf", ha="center")
    fig.text(0.5, 0.40, "YARDAGE BOOK", fontsize=13, weight="bold", color="#c9b458", ha="center")
    fig.text(0.5, 0.365, f"{C['tee']} tees  ·  {tot:,} yards  ·  Par {sum(h['par'] for h in C['holes'])}", fontsize=8.5, color="white", ha="center")
    fig.text(0.5, 0.34, f"Rating {C['rating']}  ·  Slope {C['slope']}", fontsize=8.5, color="white", ha="center")
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    return fig


def scorecard(pdf, C, L):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.06, 0.957, "SCORECARD", fontsize=13, weight="bold", color=INK, va="center")
    fig.text(0.94, 0.957, f"{C['tee']} · {C['rating']} / {C['slope']}", fontsize=7.5, color=COL["blue"], ha="right", va="center", weight="bold")
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.936, 0.936], color=INK, lw=0.8))
    H = C["holes"]
    cols = [0.09, 0.2, 0.31, 0.45, 0.6, 0.76, 0.9]
    heads = ["Hole", "Par", "Hcp", "Yards", "Plays", "Elev ft", "Score"]
    y = 0.915
    for x, t in zip(cols, heads):
        fig.text(x, y, t.upper(), fontsize=5.2, color="#66736a", ha="center")
    y -= 0.008

    def row(vals, bold=False, shade=False):
        nonlocal y
        y -= 0.0185
        if shade:
            fig.add_artist(plt.Rectangle((0.06, y - 0.006), 0.88, 0.0185, fc="#e9f0e6", ec="none"))
        for x, t in zip(cols, vals):
            fig.text(x, y, str(t), fontsize=6.6, ha="center", color=INK, weight="bold" if bold else "normal")

    def block(hs, name):
        for h in hs:
            row([h["n"], h["par"], h["handicap"], h["yards"], h["plays_like"], f"{h['elev_change_ft']:+.0f}", ""])
        row([name, sum(h["par"] for h in hs), "", sum(h["yards"] for h in hs), sum(h["plays_like"] for h in hs), "", ""], True, True)

    block(H[:9], "OUT")
    block(H[9:], "IN")
    row(["TOT", sum(h["par"] for h in H), "", sum(h["yards"] for h in H), sum(h["plays_like"] for h in H), "", ""], True, True)

    # routing map, north up
    ax = fig.add_axes([0.06, 0.05, 0.88, 0.40])
    pts = np.vstack([np.array(h["line"]) for h in H])
    c = (pts.min(0) + pts.max(0)) / 2
    fr = Frame(c, (0, 1))
    aw, ah = 0.88 * PAGE[0], 0.40 * PAGE[1]
    span = pts.max(0) - pts.min(0) + 90
    sc = max(span[0] / aw, span[1] / ah)
    xlim, ylim = (-sc * aw / 2, sc * aw / 2), (-sc * ah / 2, sc * ah / 2)
    draw_base(ax, C, fr, xlim, ylim, L)
    for h in H:
        q = fr.to(h["line"])
        ax.plot(q[:, 0], q[:, 1], color=INK, lw=0.5, ls=(0, (3, 2)), zorder=9)
        ax.text(q[0, 0], q[0, 1], str(h["n"]), fontsize=5.4, weight="bold", color="white", ha="center", va="center", zorder=12,
                bbox=dict(boxstyle="circle,pad=0.18", fc=COL["blue"], ec="white", lw=0.4))
    setup(ax, xlim, ylim)
    north_arrow(ax, fr, xlim, ylim)
    fig.text(0.06, 0.022, "“Plays” adds 1 yard per 3 ft of climb from blue tee to green.", fontsize=5, color="#66736a")
    pdf.savefig(fig)
    return fig


def notes_page(pdf, C):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.06, 0.957, "HOW TO READ THIS BOOK", fontsize=11, weight="bold", color=INK, va="center")
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.936, 0.936], color=INK, lw=0.8))
    txt = [
        ("Black numbers", "Yards from the blue tee. Ticks on the line of play are carry distances; “235–251” beside a bunker or water is the distance to reach it and to carry it."),
        ("Red numbers", "Yards to the centre of the green. Red ticks mark 100, 150, 200 and 250 out; a red number beside a hazard is what is left once you are past it."),
        ("Par 3s", "F, C and B are yards from the blue tee to the front, centre and back of the green along the line of play."),
        ("Contours", "Hole maps: every 5 ft, heavier every 25 ft. Green pages: every 1 ft, dotted every 6 in, with arrows showing which way the ground falls."),
        ("Plays like", "Scorecard yards plus 1 yard for every 3 ft the green sits above the tee (minus when it is below). Wind and temperature are yours to add."),
        ("Where the numbers come from", "Hole layout, greens, tees and bunkers: OpenStreetMap contributors. Elevation: USGS 3DEP 1 m lidar. Trees: USGS lidar (2012) checked against USDA NAIP imagery. Scorecard: the club’s blue tees, 6,356 yards, par 72."),
        ("Caveats", "Blue tee positions are the mapped tee box that best matches the card; markers move daily. Green contours are from lidar, so a green rebuilt since the survey will differ. Pin positions are not shown."),
    ]
    y = 0.90
    for k, v in txt:
        fig.text(0.06, y, k.upper(), fontsize=5.6, color=COL["blue"] if "Black" not in k and "Red" not in k else (INK if "Black" in k else RED), weight="bold")
        t = fig.text(0.06, y - 0.018, v, fontsize=6.3, color=INK, va="top", wrap=True)
        t._get_wrap_line_width = lambda: PAGE[0] * 72 * 0.88
        y -= 0.118
    pdf.savefig(fig)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holes", default="")
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args()
    C = course()
    D = dem()
    L = {"trees": Grid(np.load(DER / "treemask_1m.npy").astype(np.float32)),
         "dem_s": Grid(ndi.gaussian_filter(D.a, 2.0)),
         "dem_g": Grid(ndi.gaussian_filter(D.a, 1.2))}
    OUT.mkdir(exist_ok=True)
    want = [int(v) for v in a.holes.split(",")] if a.holes else list(range(1, 19))
    name = OUT / ("yardage_book.pdf" if not a.holes else "yardage_book_sample.pdf")
    figs = []
    greens = {}
    with PdfPages(name) as pdf:
        if not a.holes:
            figs += [cover(pdf, C), scorecard(pdf, C, L)]
        page = 3
        for h in C["holes"]:
            if h["n"] not in want:
                continue
            figs.append(hole_page(pdf, C, h, L, page))
            f, info = green_page(pdf, C, h, L, page + 1)
            figs.append(f)
            greens[h["n"]] = info
            page += 2
        if not a.holes:
            figs.append(notes_page(pdf, C))
    if a.png:
        (OUT / "pages").mkdir(exist_ok=True)
        for i, f in enumerate(figs):
            f.savefig(OUT / f"pages/p{i + 1:02d}.png", dpi=170, facecolor=f.get_facecolor())
    print("wrote", name, len(figs), "pages")
    for n, g in greens.items():
        print(f"  green {n}: {g['slope']:.1f}% toward {g['toward']}")


if __name__ == "__main__":
    main()
