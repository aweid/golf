"""Build the ground texture for the video: data/derived/texture_0.25m.jpg

NAIP is ~0.6 m native, so close to the ground it is soft. This upsamples it to 0.25 m,
sharpens and colour-grades it, redraws greens, tees, fairways, bunkers and water from the
mapped outlines so their edges are crisp, and adds fine grass grain.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi

from common import DER, RAW, course, extent

Image.MAX_IMAGE_PIXELS = None
RES = 0.25
SAND = np.array([238, 226, 190], np.float32)
WATER = np.array([40, 66, 84], np.float32)


def grade(a, sat=1.25, gamma=0.86, gain=1.06):
    a = (a.astype(np.float32) / 255.0) ** gamma * gain
    l = a.mean(axis=2, keepdims=True)
    return np.clip(l + (a - l) * sat, 0, 1) * 255


def main():
    C = course()
    x0, y0, x1, y1 = extent()
    src = np.load(RAW / "naip_rgbn_0.5m.npy")[..., :3]
    W, H = int((x1 - x0) / RES), int((y1 - y0) / RES)
    im = Image.fromarray(src).resize((W, H), Image.BICUBIC).filter(ImageFilter.UnsharpMask(radius=3, percent=70, threshold=2))
    tex = np.asarray(im).copy()
    rng = np.random.default_rng(11)

    # colour grade + grass grain, in strips to keep memory down
    grain = rng.normal(0, 1, (1024, 1024)).astype(np.float32)
    grain = ndi.gaussian_filter(grain, 0.8, mode="wrap")
    coarse = ndi.zoom(rng.normal(0, 1, (128, 128)).astype(np.float32), 8, order=3, mode="wrap")
    grain = grain / grain.std() * 0.035 + coarse / coarse.std() * 0.02
    for r in range(0, H, 1024):
        blk = grade(tex[r:r + 1024])
        g = np.tile(grain, (1, W // 1024 + 1))[: blk.shape[0], :W]
        veg = (blk[..., 1] > blk[..., 0]) & (blk[..., 1] > blk[..., 2])
        blk *= (1 + g * veg)[..., None]
        tex[r:r + 1024] = np.clip(blk, 0, 255).astype(np.uint8)

    def paint(poly, mode, strength):
        p = np.asarray(poly)
        px = (p[:, 0] - x0) / RES
        py = (y1 - p[:, 1]) / RES
        c0, c1 = int(max(px.min() - 6, 0)), int(min(px.max() + 6, W))
        r0, r1 = int(max(py.min() - 6, 0)), int(min(py.max() + 6, H))
        if c1 <= c0 or r1 <= r0:
            return
        m = Image.new("L", (c1 - c0, r1 - r0), 0)
        ImageDraw.Draw(m).polygon(list(zip(px - c0, py - r0)), fill=255)
        a = np.asarray(m.filter(ImageFilter.GaussianBlur(1.2)), np.float32)[..., None] / 255.0 * strength
        win = tex[r0:r1, c0:c1].astype(np.float32)
        core = np.asarray(m) > 128
        if not core.any():
            return
        lum = win.mean(axis=2, keepdims=True)
        rel = np.clip(lum / max(lum[core].mean(), 1), 0.86, 1.1)     # keep the real light and shade
        if mode == "sand":
            tgt = SAND * rel
        elif mode == "water":
            tgt = WATER * np.clip(rel, 0.9, 1.15)
        else:
            mean = win[core].mean(axis=0)
            tgt = mean * {"green": 1.07, "fairway": 1.04, "tee": 1.05}[mode] * rel
        tex[r0:r1, c0:c1] = np.clip(win * (1 - a) + tgt * a, 0, 255).astype(np.uint8)

    A = C["all"]
    for p in A["water"]:
        paint(p, "water", 0.55)
    for p in A["fairways"]:
        paint(p, "fairway", 0.45)
    for p in A["tees"]:
        paint(p, "tee", 0.5)
    for p in A["greens"]:
        paint(p, "green", 0.7)
    for p in A["bunkers"]:
        paint(p, "sand", 0.8)

    # building walls: the terrain mesh stretches edge pixels down the walls, so make the edge a wall colour
    bld = np.load(DER / "bldg_h_1m.npy") > 0
    ring = bld & ~ndi.binary_erosion(bld, iterations=1)
    ring = np.kron(ring, np.ones((4, 4), bool))
    wall = np.array([196, 186, 170], np.float32)
    tex[ring] = (0.25 * tex[ring] + 0.75 * wall).astype(np.uint8)

    Image.fromarray(tex).save(DER / "texture_0.25m.jpg", quality=92)
    Image.fromarray(tex[::8, ::8]).save(DER / "texture_preview.jpg", quality=85)
    print("texture", tex.shape)


if __name__ == "__main__":
    main()
