"""Render the hole-by-hole flyover video.

  python scripts/render_flyover.py                    # every hole -> out/flyover.mp4
  COURSE=valley python scripts/render_flyover.py      # the Valley course -> out/flyover_valley.mp4
  python scripts/render_flyover.py --holes 1,7        # just these
  python scripts/render_flyover.py --stills           # a few test frames per hole -> out/stills/
  python scripts/render_flyover.py --size 1280x720 --fps 24

Scene: 1 m lidar terrain (with buildings) draped with NAIP imagery, one low-poly tree per
lidar-detected crown, flagsticks, tee markers, low-res surroundings out to the horizon.
"""
import argparse
import subprocess
import time

import numpy as np
import pyvista as pv
import tifffile
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

from common import DER, OUT, RAW, SUFFIX, YD, Grid, course, dem, extent

SKY_TOP = (0.36, 0.56, 0.86)
SKY_BOT = (0.80, 0.88, 0.96)
HAZE = np.array([0.80, 0.87, 0.95], np.float32)


# ---------------------------------------------------------------- scene

def grade(rgb, sat=1.22, gamma=0.88, gain=1.06):
    a = (rgb.astype(np.float32) / 255.0) ** gamma * gain
    l = a.mean(axis=2, keepdims=True)
    return (np.clip(l + (a - l) * sat, 0, 1) * 255).astype(np.uint8)


def terrain_mesh(z, bbox, origin, tex_rgb, zoff=0.0):
    x0, y0, x1, y1 = bbox
    h, w = z.shape
    xs = np.linspace(x0, x1, w, dtype=np.float32) - origin[0]
    ys = np.linspace(y1, y0, h, dtype=np.float32) - origin[1]
    X, Y = np.meshgrid(xs, ys)
    g = pv.StructuredGrid(X, Y, z.astype(np.float32) + zoff)
    # StructuredGrid stores points in Fortran order
    u = ((X + origin[0] - x0) / (x1 - x0)).ravel(order="F")
    v = ((Y + origin[1] - y0) / (y1 - y0)).ravel(order="F")
    g.active_texture_coordinates = np.column_stack([u, v]).astype(np.float32)
    tex = pv.Texture(np.ascontiguousarray(tex_rgb))
    tex.interpolate = True
    tex.mipmap = True
    try:
        tex.SetMaximumAnisotropicFiltering(16.0)
    except Exception:
        pass
    return g, tex


def icosphere():
    s = pv.Icosphere(nsub=1, radius=1.0)
    return np.asarray(s.points, np.float32), s.faces.reshape(-1, 4)[:, 1:]


def tree_mesh(trees, origin):
    """Merged mesh: lumpy ellipsoid crown + trunk prism per tree, per-vertex colour."""
    rng = np.random.default_rng(3)
    sv, sf = icosphere()
    n, k = len(trees), len(sv)
    x, y, zg, h, rad = (trees[:, i] for i in range(5))
    col = trees[:, 5:8]
    crown_h = h * rng.uniform(0.62, 0.8, n)
    cz = zg + h - crown_h / 2
    lump = rng.uniform(0.82, 1.12, (n, k, 1)).astype(np.float32)
    P = sv[None] * lump * np.stack([rad, rad, crown_h / 2], 1)[:, None, :]
    P += np.stack([x - origin[0], y - origin[1], cz], 1)[:, None, :]
    shade = 0.50 + 0.62 * (sv[:, 2] + 1) / 2                        # dark underside, bright top
    tint = rng.uniform(0.82, 1.14, (n, 1, 1)) * np.array([0.92, 1.10, 0.80])
    col = np.clip(col, 0.12, 1) ** 0.86
    C = np.clip(col[:, None, :] * tint * shade[None, :, None] * 1.2, 0, 1)
    F = sf[None] + (np.arange(n) * k)[:, None, None]
    crowns = pv.PolyData(P.reshape(-1, 3), np.column_stack([np.full(n * len(sf), 3), F.reshape(-1, 3)]).ravel())
    crowns["rgb"] = (C.reshape(-1, 3) * 255).astype(np.uint8)

    # trunks: 4-sided prisms from the ground into the crown
    ang = np.arange(4) * np.pi / 2
    ring = np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
    tr = np.clip(0.02 * h + 0.12, 0.2, 0.6)
    base = np.stack([x - origin[0], y - origin[1]], 1)[:, None, :] + ring[None] * tr[:, None, None]
    bot = np.concatenate([base, np.broadcast_to((zg - 0.5)[:, None, None], (n, 4, 1))], 2)
    top = np.concatenate([base, np.broadcast_to(cz[:, None, None], (n, 4, 1))], 2)
    TP = np.concatenate([bot, top], 1).reshape(-1, 3)
    q = np.array([[i, (i + 1) % 4, (i + 1) % 4 + 4, i + 4] for i in range(4)])
    TF = q[None] + (np.arange(n) * 8)[:, None, None]
    trunks = pv.PolyData(TP, np.column_stack([np.full(n * 4, 4), TF.reshape(-1, 4)]).ravel())
    return crowns, trunks


def build_scene(size):
    C = course()
    bbox = extent()
    origin = (bbox[0], bbox[1])
    D = dem()
    bh = np.load(DER / "bldg_h_1m.npy")
    z = D.a + bh
    Image.MAX_IMAGE_PIXELS = None
    img = np.asarray(Image.open(DER / "texture_0.25m.jpg"))        # already graded by build_texture.py

    pl = pv.Plotter(off_screen=True, window_size=list(size), lighting="none")
    pl.set_background(SKY_BOT, top=SKY_TOP)
    sun = pv.Light(position=(-0.5, -0.6, 0.62), light_type="scene light", intensity=0.9)
    sun.positional = False
    pl.add_light(sun)
    pl.add_light(pv.Light(position=(0.4, 0.5, 0.75), light_type="scene light", intensity=0.3))

    g, tex = terrain_mesh(z, bbox, origin, img)
    pl.add_mesh(g, texture=tex, ambient=0.72, diffuse=0.38, specular=0, smooth_shading=True)

    cb = [bbox[0] - 2400, bbox[1] - 2400, bbox[2] + 2400, bbox[3] + 2400]
    cz = tifffile.imread(RAW / "ctx_dem_5m.tif").astype(np.float32)
    cimg = grade(np.asarray(Image.open(RAW / "ctx_rgb_2m.jpg")))
    hh, ww = cz.shape
    xs = np.linspace(cb[0], cb[2], ww)
    ys = np.linspace(cb[3], cb[1], hh)
    inside = (xs[None, :] > bbox[0] + 12) & (xs[None, :] < bbox[2] - 12) & (ys[:, None] > bbox[1] + 12) & (ys[:, None] < bbox[3] - 12)
    cz = np.where(inside, cz - 60, cz - 1.5)
    cg, ctex = terrain_mesh(cz, cb, origin, cimg)
    pl.add_mesh(cg, texture=ctex, ambient=0.72, diffuse=0.38, specular=0)

    trees = np.load(DER / "trees.npy")
    crowns, trunks = tree_mesh(trees, origin)
    pl.add_mesh(crowns, scalars="rgb", rgb=True, ambient=0.55, diffuse=0.6, specular=0, smooth_shading=True)
    pl.add_mesh(trunks, color=(0.28, 0.22, 0.17), ambient=0.6, diffuse=0.4)

    for h in C["holes"]:
        gx, gy = h["green_center"]
        gz = float(D.sample(gx, gy))
        p = np.array([gx - origin[0], gy - origin[1], gz])
        pl.add_mesh(pv.Cylinder(center=p + (0, 0, 1.5), direction=(0, 0, 1), radius=0.07, height=3.0),
                    color="white", ambient=0.9)
        a = np.array(h["approach_dir"])
        side = np.array([-a[1], a[0], 0])
        flag = pv.Triangle([p + (0, 0, 3.0), p + (0, 0, 2.2), p + side * 1.3 + (0, 0, 2.6)])
        pl.add_mesh(flag, color=(0.95, 0.1, 0.1), ambient=0.9, lighting=False)
        tx, ty = h["tee"]
        tz = float(D.sample(tx, ty))
        d0 = np.array(h["line"][1]) - np.array(h["line"][0])
        d0 /= np.linalg.norm(d0)
        s0 = np.array([-d0[1], d0[0]])
        for sgn in (-1, 1):
            q = np.array([tx, ty]) + sgn * 2.5 * s0
            pl.add_mesh(pv.Sphere(radius=0.28, center=(q[0] - origin[0], q[1] - origin[1], tz + 0.25)),
                        color=(0.1, 0.25, 0.95), ambient=0.8)
    try:
        pl.enable_anti_aliasing("msaa", multi_samples=8)
    except Exception:
        pass
    pl.camera.view_angle = 42
    return pl, C, D, origin


# ---------------------------------------------------------------- camera paths

def smooth(a, sigma):
    return ndi.gaussian_filter1d(a, sigma, axis=0, mode="nearest")


def resample(path, step=1.0):
    path = np.asarray(path, float)
    seg = np.hypot(*np.diff(path, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    t = np.arange(0, s[-1], step)
    return np.column_stack([np.interp(t, s, path[:, 0]), np.interp(t, s, path[:, 1])])


def ease(n):
    t = np.linspace(0, 1, n)
    return t * t * (3 - 2 * t)


def hole_path(h, D, clear, fps, speed=30.0):
    """Returns (positions Nx3, targets Nx3) in projected coordinates."""
    line = np.array(h["line"], float)
    d0 = (line[1] - line[0]) / np.linalg.norm(line[1] - line[0])
    gc = np.array(h["green_center"], float)
    track = resample(np.vstack([line[0] - d0 * 70, line]), 1.0)
    track = smooth(track, 18)                                       # round the doglegs
    L = len(track)
    lead = 95
    # stop the travelling shot ~60 m short of the green centre
    stop = max(L - 60, lead + 30)
    n_fly = int((stop - 0) / speed * fps)
    # ease in/out along the track
    u = np.linspace(0, 1, n_fly)
    prog = (u * u * (3 - 2 * u)) * 0.25 + u * 0.75
    idx = prog * (stop - 1)
    cam = np.column_stack([np.interp(idx, np.arange(L), track[:, k]) for k in range(2)])
    tgt_idx = np.minimum(idx + 150, L - 1)
    tgt = np.column_stack([np.interp(tgt_idx, np.arange(L), track[:, k]) for k in range(2)])
    w = np.clip((idx + 150 - (L - 1)) / 60, 0, 1)[:, None]         # settle on the green centre
    tgt = tgt * (1 - w) + gc * w

    # orbit ~140 degrees round the green, closing in
    n_orb = int(5.5 * fps)
    v = cam[-1] - gc
    r0, a0 = np.linalg.norm(v), np.arctan2(v[1], v[0])
    sgn = 1 if h["n"] % 2 else -1
    e = ease(n_orb)
    ang = a0 + sgn * np.radians(140) * e
    rad = r0 + (48 - r0) * e
    orb = gc + np.column_stack([np.cos(ang), np.sin(ang)]) * rad[:, None]
    cam = np.vstack([cam, orb])
    tgt = np.vstack([tgt, np.repeat(gc[None], n_orb, 0)])

    hold = int(1.6 * fps)                                           # opening hold behind the tee
    cam = np.vstack([np.repeat(cam[:1], hold, 0), cam])
    tgt = np.vstack([np.repeat(tgt[:1], hold, 0), tgt])
    cam, tgt = smooth(cam, fps * 0.35), smooth(tgt, fps * 0.45)

    ground = D.sample(cam[:, 0], cam[:, 1])
    # stay above the canopy under the camera and see over what is just ahead
    look = tgt - cam
    look /= np.maximum(np.linalg.norm(look, axis=1, keepdims=True), 1e-6)
    top = np.max([clear.sample(cam[:, 0] + look[:, 0] * d, cam[:, 1] + look[:, 1] * d) + 9 - 0.16 * d
                  for d in (0, 15, 30, 45, 60, 80)], axis=0)
    height = np.r_[np.full(hold + n_fly, 26.0), 26 + (15 - 26) * e]
    zc = np.maximum(ground + height, top)
    zc = smooth(ndi.maximum_filter1d(zc, int(fps * 1.2)), fps * 0.8)
    zt = smooth(D.sample(tgt[:, 0], tgt[:, 1]) + 2.0, fps * 0.6)
    return np.column_stack([cam, zc]), np.column_stack([tgt, zt])


def overview_path(C, D, fps, secs=7.0):
    pts = np.array([h["green_center"] for h in C["holes"]] + [h["tee"] for h in C["holes"]])
    c = pts.mean(0)
    n = int(secs * fps)
    e = ease(n)
    ang = np.radians(215) + np.radians(70) * e
    rad = 1500 - 250 * e
    cam = c + np.column_stack([np.cos(ang), np.sin(ang)]) * rad[:, None]
    z0 = float(D.sample(*c))
    zc = z0 + 700 - 150 * e
    return np.column_stack([cam, zc]), np.repeat(np.array([[c[0], c[1], z0]]), n, 0)


# ---------------------------------------------------------------- overlay + encode

def font(sz, bold=False):
    for f in (["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/HelveticaNeue.ttc"] if bold
              else ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/HelveticaNeue.ttc"]):
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            pass
    return ImageFont.load_default(sz)


def hole_card(C, h, size):
    W, H = size
    s = H / 1080
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x, y, w, hh = int(48 * s), int(44 * s), int(430 * s), int(150 * s)
    d.rounded_rectangle([x, y, x + w, y + hh], int(14 * s), fill=(12, 28, 20, 175))
    d.rectangle([x, y + int(14 * s), x + int(8 * s), y + hh - int(14 * s)], fill=(40, 90, 230, 255))
    d.text((x + 28 * s, y + 14 * s), f"HOLE {h['n']}", font=font(int(58 * s), True), fill="white")
    d.text((x + 30 * s, y + 86 * s), f"Par {h['par']}   {h['yards']} yds   Hcp {h['handicap']}",
           font=font(int(30 * s)), fill=(235, 240, 235, 255))
    el = h["elev_change_ft"]
    sub = f"{C['tee']} tees  ·  {'up' if el > 0 else 'down'} {abs(el):.0f} ft  ·  green {h['green_depth_yd']} deep"
    d.text((x + 30 * s, y + hh + 8 * s), sub, font=font(int(22 * s)), fill=(255, 255, 255, 230),
           stroke_width=max(1, int(2 * s)), stroke_fill=(0, 0, 0, 160))
    t = f"{C['club']}  ·  {C['course']}"
    d.text((W - 48 * s, H - 54 * s), t, font=font(int(22 * s)), fill=(255, 255, 255, 215), anchor="ra",
           stroke_width=max(1, int(2 * s)), stroke_fill=(0, 0, 0, 150))
    return im


def title_card(C, size):
    W, H = size
    s = H / 1080
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([0, int(H * 0.36), W, int(H * 0.64)], fill=(12, 28, 20, 150))
    d.text((W / 2, H * 0.44), C["club"].upper(), font=font(int(78 * s), True), fill="white", anchor="mm")
    d.text((W / 2, H * 0.53), C["course"], font=font(int(44 * s)), fill=(235, 240, 235, 255), anchor="mm")
    tot = sum(h["yards"] for h in C["holes"])
    d.text((W / 2, H * 0.595), f"{C['location']}   ·   {C['tee']} tees   ·   {tot:,} yards   ·   Par {sum(h['par'] for h in C['holes'])}"
           f"   ·   {C['rating']} / {C['slope']}", font=font(int(26 * s)), fill=(225, 232, 225, 255), anchor="mm")
    return im


def shoot(pl, pos, tgt, origin):
    o = np.array([origin[0], origin[1], 0.0])
    pl.camera.position = tuple(pos - o)
    pl.camera.focal_point = tuple(tgt - o)
    pl.camera.up = (0, 0, 1)
    pl.camera.clipping_range = (3.0, 9000.0)
    pl.render()
    img = pl.screenshot(None, return_img=True).astype(np.float32) / 255.0
    depth = -pl.get_image_depth()                                   # metres from camera, nan = sky
    f = 1 - np.exp(-(np.nan_to_num(depth, nan=0.0) / 2600.0) ** 1.4)
    f = np.where(np.isfinite(depth), f, 0)[..., None] * 0.9
    img = img * (1 - f) + HAZE * f
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def composite(frame, card, alpha=1.0):
    if card is None or alpha <= 0:
        return frame
    im = Image.fromarray(frame).convert("RGBA")
    c = card if alpha >= 1 else Image.blend(Image.new("RGBA", card.size, (0, 0, 0, 0)), card, alpha)
    return np.asarray(Image.alpha_composite(im, c).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holes", default="")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stills", action="store_true")
    ap.add_argument("--out", default=str(OUT / f"flyover{SUFFIX}.mp4"))
    a = ap.parse_args()
    size = tuple(int(v) for v in a.size.split("x"))
    OUT.mkdir(exist_ok=True)

    pl, C, D, origin = build_scene(size)
    chm = np.load(DER / "chm_1m.npy") + np.load(DER / "bldg_h_1m.npy")
    clear = Grid(ndi.maximum_filter(D.a + chm, size=31))
    holes = [h for h in C["holes"] if not a.holes or str(h["n"]) in a.holes.split(",")]

    if a.stills:
        S = OUT / f"stills{SUFFIX}"
        S.mkdir(exist_ok=True)
        p, t = overview_path(C, D, a.fps)
        Image.fromarray(composite(shoot(pl, p[len(p) // 2], t[0], origin), title_card(C, size))).save(S / "overview.jpg", quality=90)
        for h in holes:
            p, t = hole_path(h, D, clear, a.fps)
            card = hole_card(C, h, size)
            for k, fr in enumerate([0.02, 0.4, 0.75, 0.97]):
                i = int(fr * (len(p) - 1))
                t0 = time.time()
                f = composite(shoot(pl, p[i], t[i], origin), card)
                Image.fromarray(f).save(S / f"hole{h['n']:02d}_{k}.jpg", quality=90)
                print(h["n"], k, f"{time.time() - t0:.2f}s")
        return

    import imageio_ffmpeg
    ff = subprocess.Popen(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{size[0]}x{size[1]}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-preset", "medium",
         "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)

    def emit(frames_pos, frames_tgt, card, fade=None):
        n = len(frames_pos)
        fi = int(0.4 * a.fps)
        for i in range(n):
            f = shoot(pl, frames_pos[i], frames_tgt[i], origin)
            f = composite(f, card, min(1.0, (i + 1) / fi))
            edge = min(i, n - 1 - i)                                # dip to black between segments
            if edge < fi:
                f = (f.astype(np.float32) * ((edge + 1) / (fi + 1))).astype(np.uint8)
            ff.stdin.write(np.ascontiguousarray(f).tobytes())

    t0 = time.time()
    total = 0
    if not a.holes:
        p, t = overview_path(C, D, a.fps)
        emit(p, t, title_card(C, size))
        total += len(p)
    for h in holes:
        p, t = hole_path(h, D, clear, a.fps)
        emit(p, t, hole_card(C, h, size))
        total += len(p)
        print(f"hole {h['n']}: {len(p)} frames, total {total}, {total / (time.time() - t0):.1f} fps", flush=True)
    ff.stdin.close()
    ff.wait()
    print("wrote", a.out, f"{total / a.fps:.0f}s of video in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
