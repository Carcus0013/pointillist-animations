"""
dandelion.py - a pointillist dandelion, from the first leaf to the last seed on the wind.

A rosette of toothed leaves unfurls, a stalk rises with a green bud, the bud opens
into a golden flower, folds shut and bows, then straightens taller and opens again
as a silver seed clock. The wind picks up and carries the seeds away, one by one.

The same plant, at three times of day:
    night     a meadow under the moon, stars and fireflies                     (default)
    sunset    a burning sky, the sun sinking behind the hills, swallows overhead,
              the plant rim-lit in orange, the first fireflies at dusk
    sunrise   a pastel dawn, light beams through mist that slowly lifts, dew
              sparkling on every leaf, blade and seed

    python dandelion.py [--theme night|sunset|sunrise]    interactive window
    python dandelion.py --theme sunset --png               save five stages
    python dandelion.py --theme sunrise --gif              save the whole life as a GIF
    python dandelion_sunset.py / dandelion_sunrise.py      shortcuts for the other two

Window keys: Space pause, R restart, S save frame, Esc quit.
Needs numpy and Pillow (tkinter ships with Python).
"""
import math
import sys
import time

import numpy as np
from PIL import Image

SEED = 11
T_END = 54.0
LOOP = False          # start again after the last seed has gone
WIN_FRAC = 0.92
TICK_MS = 10

SOIL = [(16, 22, 18), (24, 32, 24), (34, 42, 30), (46, 52, 36), (30, 28, 24)]
GRASS = [(14, 38, 30), (20, 54, 36), (30, 74, 44), (50, 98, 52)]
LEAF = [(24, 80, 46), (36, 108, 54), (58, 138, 60), (96, 168, 72)]
LEAF_RIB = [(156, 208, 128), (186, 224, 150)]
STEM = [(150, 84, 96), (118, 138, 90), (126, 168, 98), (160, 196, 118)]
BRACT = [(44, 102, 54), (64, 132, 64), (94, 158, 78)]
BRACT_TIP = [(142, 72, 98), (118, 62, 88)]
BRACT_DRY = [(118, 114, 72), (96, 94, 60)]
PETAL = [(255, 242, 136), (255, 224, 72), (255, 200, 40), (250, 168, 22)]
PETAL_DEAD = [(170, 128, 64), (142, 104, 56), (116, 86, 52)]
SILK = [(238, 242, 254), (216, 224, 244), (196, 206, 232)]
ACHENE = [(142, 98, 58), (112, 78, 48)]
MOON = [(252, 248, 228), (238, 234, 212), (222, 218, 198)]
DEW = np.array([214, 232, 246], np.float32)

THEMES = {
    "night": dict(
        title="Dandelion - night", sky=None, clouds=None, hills=(), sun=None, moon=True,
        stars=(380, 0.0, 0.0, 0.68), venus=False, sky_fade=(1.0, 1.0),
        light=None, flies=(0.0, 0.0, 1.0), birds=0, dew=False, mist=0.0, rays=0, motes=0,
        soil=(1.0, 1.0, 1.0), grass=GRASS, streak=(110, 130, 170), seed_glow=0.0),
    "sunset": dict(
        title="Dandelion - sunset",
        sky=[(0.00, (30, 22, 62)), (0.20, (76, 40, 100)), (0.40, (166, 70, 108)),
             (0.56, (234, 112, 70)), (0.68, (255, 168, 88)), (0.80, (255, 204, 126))],
        clouds=dict(items=[(0.22, 0.17, 0.20, 0.012, 2600), (0.58, 0.11, 0.24, 0.010, 2600),
                           (0.84, 0.29, 0.17, 0.012, 2200), (0.36, 0.35, 0.22, 0.010, 2400),
                           (0.64, 0.45, 0.20, 0.008, 2000), (0.12, 0.50, 0.14, 0.007, 1400)],
                    lit=[(255, 150, 100), (255, 122, 108), (255, 180, 120)],
                    shade=[(104, 56, 108), (128, 64, 112), (86, 46, 96)]),
        hills=[(0.735, 0.024, (118, 62, 96)), (0.768, 0.018, (70, 36, 64))],
        sun=dict(x=0.70, y0=0.60, y1=0.79, r=0.060, reach=5.0, glow=(150, 64, 22), glow_t=(1.0, 0.55),
                 core=[(255, 206, 128), (255, 164, 82), (242, 112, 60)]),
        moon=False, stars=(70, 34.0, 50.0, 0.32), venus=True, sky_fade=(1.0, 0.70),
        light=dict(dir=(1.0, -0.12), amb=(0.64, 0.50, 0.58), key=(255, 150, 70), power=1.0),
        flies=(40.0, 52.0, 0.75), birds=7, dew=False, mist=0.0, rays=0, motes=0,
        soil=(1.05, 0.78, 0.86), grass=[(30, 24, 34), (40, 32, 40), (54, 42, 48), (78, 58, 56)],
        streak=(255, 170, 110), seed_glow=0.5),
    "sunrise": dict(
        title="Dandelion - sunrise",
        sky=[(0.00, (108, 146, 202)), (0.26, (160, 178, 222)), (0.46, (218, 186, 210)),
             (0.62, (252, 202, 182)), (0.73, (255, 224, 182)), (0.80, (250, 236, 206))],
        clouds=dict(items=[(0.30, 0.19, 0.25, 0.010, 2200), (0.72, 0.14, 0.22, 0.008, 2000),
                           (0.56, 0.31, 0.18, 0.008, 1800), (0.14, 0.40, 0.12, 0.006, 1200)],
                    lit=[(255, 206, 190), (255, 220, 200), (252, 196, 196)],
                    shade=[(232, 218, 236), (214, 206, 232), (244, 230, 240)]),
        hills=[(0.742, 0.020, (150, 156, 190)), (0.772, 0.016, (104, 118, 146))],
        sun=dict(x=0.22, y0=0.765, y1=0.56, r=0.045, reach=6.0, glow=(110, 92, 52), glow_t=(0.7, 1.1),
                 core=[(255, 253, 236), (255, 246, 212), (255, 234, 184)]),
        moon=False, stars=(0, 0.0, 0.0, 0.0), venus=False, sky_fade=(0.86, 1.04),
        light=dict(dir=(-1.0, -0.22), amb=(0.94, 0.97, 1.0), key=(255, 214, 150), power=0.55),
        flies=(0.0, 0.0, 0.0), birds=0, dew=True, mist=1.0, rays=14, motes=70,
        soil=(1.9, 2.5, 2.2), grass=[(30, 72, 58), (42, 98, 66), (64, 130, 76), (112, 166, 94)],
        streak=(255, 238, 205), seed_glow=1.0),
}

# floret rings, outside in: count, length, open angle, closed angle, tip curl, colour position
RINGS = [
    (46, 1.00, 1.52, 0.16, 1.0, 0.00),
    (40, 0.90, 1.30, 0.14, 0.8, 0.25),
    (33, 0.78, 1.05, 0.12, 0.5, 0.50),
    (26, 0.62, 0.78, 0.10, 0.2, 0.75),
    (17, 0.44, 0.45, 0.08, 0.0, 1.00),
]
# rosette leaves: azimuth on the ground (0 = right, +pi/2 = towards the viewer), length, delay
LEAVES = [
    (0.10, 1.00, 0.00), (3.05, 0.96, 0.05), (0.85, 0.80, 0.12), (2.25, 0.86, 0.10),
    (-0.55, 0.78, 0.18), (3.75, 0.74, 0.22), (1.62, 0.62, 0.28),
]
DEPTH = 0.35          # how much the ground plane is foreshortened


# ---- small helpers -----------------------------------------------------------------
def ss(x, a, b):
    u = np.clip((np.asarray(x, np.float64) - a) / (b - a), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def disk_offsets(r=3):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    d2 = (xx ** 2 + yy ** 2).ravel()
    keep = np.nonzero(d2 <= r * r)[0]
    order = keep[np.argsort(d2[keep], kind="stable")]
    return np.stack([xx.ravel()[order], yy.ravel()[order]], 1).astype(np.int32), d2[order]


OFFS, OFF_D2 = disk_offsets(3)


def _footprint(shape, x, y, rad):
    H, W = shape[:2]
    k = int(np.searchsorted(OFF_D2, float(rad.max()) ** 2 + 0.01, side="right"))
    off, d2 = OFFS[:k], OFF_D2[:k]
    xs = np.rint(x).astype(np.int32)[:, None] + off[None, :, 0]
    ys = np.rint(y).astype(np.int32)[:, None] + off[None, :, 1]
    inside = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    return xs, ys, d2, inside


def stamp(img, x, y, col, rad):
    """Paint round dots (later dots cover earlier ones)."""
    if not len(x):
        return
    xs, ys, d2, inside = _footprint(img.shape, x, y, rad)
    m = inside & (d2[None, :] <= (rad * rad)[:, None] + 0.01)
    img[ys[m], xs[m]] = col[np.nonzero(m)[0]]


def glow(img, x, y, col, rad):
    """Add soft light (dots brighten whatever is under them)."""
    x = np.atleast_1d(np.asarray(x, np.float32))
    if not len(x):
        return
    y = np.atleast_1d(np.asarray(y, np.float32))
    rad = np.broadcast_to(np.asarray(rad, np.float32), x.shape)
    col = np.broadcast_to(np.asarray(col, np.float32), (len(x), 3))
    xs, ys, d2, inside = _footprint(img.shape, x, y, rad)
    fall = np.clip(1 - np.sqrt(d2)[None, :] / (rad[:, None] + 0.6), 0, 1)
    m = inside & (fall > 0)
    if not m.any():
        return
    xs, ys = xs[m], ys[m]
    vals = col[np.nonzero(m)[0]] * fall[m][:, None]
    x0, y0 = int(xs.min()), int(ys.min())
    bw, bh = int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1
    if bw * bh > 10 * len(xs):                      # sparse: add dot by dot
        np.add.at(img, (ys, xs), vals)
        return
    idx = (ys - y0) * bw + (xs - x0)               # dense: sum per pixel, then add the patch
    patch = img[y0:y0 + bh, x0:x0 + bw]
    for c in range(3):
        patch[..., c] += np.bincount(idx, vals[:, c], bw * bh).reshape(bh, bw)


def shades(rng, pal, g, spread=0.5, jitter=0.06):
    """Pick a colour per point from a palette, guided by g in [0, 1] with some randomness."""
    pal = np.asarray(pal, np.float32)
    g = np.atleast_1d(np.asarray(g, np.float64))
    i = np.clip(np.rint(np.clip(g, 0, 1) * (len(pal) - 1) + rng.normal(0, spread, g.shape)), 0, len(pal) - 1)
    return (pal[i.astype(int)] * (1 + rng.normal(0, jitter, (len(g), 1)))).astype(np.float32)


def gradient(stops, g):
    """Colour at heights g (fractions of the frame) from (height, colour) stops."""
    ys = np.array([s[0] for s in stops], np.float64)
    cs = np.array([s[1] for s in stops], np.float64)
    g = np.asarray(g, np.float64)
    return np.stack([np.interp(g, ys, cs[:, c]) for c in range(3)], -1).astype(np.float32)


def sample_u(rng, n, width):
    """Positions along a shape, denser where it is wider, so the fill looks even."""
    out, got = [], 0
    while got < n:
        u = rng.random(2 * n)
        keep = u[rng.random(2 * n) < width(u)]
        out.append(keep)
        got += keep.size
    return np.concatenate(out)[:n]


def sample_v(rng, n, rim=0.12):
    v = rng.uniform(-1, 1, n)
    r = rng.random(n) < rim
    v[r] = np.sign(v[r]) * rng.uniform(0.85, 1.0, int(r.sum()))
    return v


def shuffle(d, rng):
    p = rng.permutation(len(next(iter(d.values()))))
    return {k: v[p] for k, v in d.items()}


def join(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def mix(a, b, w):
    return a + (b - a) * np.asarray(w, np.float32)[:, None]


def bezier(P, s):
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = P
    q = 1 - s
    x = q ** 3 * x0 + 3 * q * q * s * x1 + 3 * q * s * s * x2 + s ** 3 * x3
    y = q ** 3 * y0 + 3 * q * q * s * y1 + 3 * q * s * s * y2 + s ** 3 * y3
    dx = 3 * q * q * (x1 - x0) + 6 * q * s * (x2 - x1) + 3 * s * s * (x3 - x2)
    dy = 3 * q * q * (y1 - y0) + 6 * q * s * (y2 - y1) + 3 * s * s * (y3 - y2)
    return x, y, dx, dy


def leaf_width(u):
    body = np.sin(np.pi * np.clip(u, 0, 1) ** 1.5) ** 0.75                 # widest two-thirds out
    f = (u * 5.5) % 1.0
    return body * np.where(u < 0.82, 0.45 + 0.55 * (1 - f) ** 0.7, 1.0)    # lobes that point back


def phases(t):
    """Every moving part of the story, as 0..1 dials over time (works on arrays too)."""
    return dict(
        leaf=ss(t, 0.5, 7.5),
        grow=0.42 * ss(t, 3.0, 10.0) + 0.23 * ss(t, 10.0, 17.0) + 0.35 * ss(t, 27.0, 33.5),
        bud=ss(t, 3.5, 10.0),
        open=ss(t, 10.0, 16.5) * (1 - ss(t, 21.5, 25.5)),
        wither=ss(t, 22.0, 26.5),
        bow=ss(t, 22.5, 26.5) * (1 - ss(t, 27.5, 32.0)),
        puff=ss(t, 28.5, 34.5),
        wind=ss(t, 35.0, 37.5),
        bloom_fx=ss(t, 12.0, 16.0) * (1 - ss(t, 22.0, 25.0)),
    )


class Dandelion:
    def __init__(self, W, H, theme="night"):
        self.W, self.H = W, H
        self.theme = THEMES[theme]
        self.rng = np.random.default_rng(SEED)          # the plant: the same in every theme
        self.srng = np.random.default_rng(SEED + 101)   # sky and weather
        self.cx = W * 0.5
        self.y0 = H * 0.80
        self.ground = float(self.surface(np.array([self.cx]))[0])
        self.Ls = H * 0.40          # stalk length when the seed clock stands tallest
        self.Lh = H * 0.085         # flower radius
        L = self.theme["light"]
        if L:
            d = np.array(L["dir"], np.float64)
            self.ldir = d / np.linalg.norm(d)
            self.amb = np.array(L["amb"], np.float32)
            self.key = np.array(L["key"], np.float32) * L["power"]
        self.dew = []               # sparkling dew positions gathered while drawing each frame
        self.build_sky()
        self.build_ground()
        self.build_leaves()
        self.build_stem()
        self.build_florets()
        self.build_bracts()
        self.build_seeds()
        self.build_flecks()
        self.build_fx()
        self.build_weather()

    # ---- pose of stalk and head --------------------------------------------------
    def surface(self, x):
        xn = x / self.W
        return (self.y0 - 0.016 * self.H * np.exp(-((xn - 0.5) / 0.25) ** 2)
                - 0.004 * self.H * np.sin(xn * 31) - 0.003 * self.H * np.sin(xn * 83 + 1.3))

    def pose(self, t):
        ph = phases(t)
        H, cx, y0 = self.H, self.cx, self.ground
        L = self.Ls * ph["grow"]
        sway = H * (0.006 * np.sin(0.9 * t) + 0.003 * np.sin(2.3 * t + 1.0))
        gust = ph["wind"] * H * (0.016 + 0.012 * np.sin(1.3 * t) * np.sin(0.47 * t + 0.8))
        bend = ph["bow"]                       # after flowering the stalk lies down for a while
        P = ((cx + 0 * L, y0 + 4 + 0 * L),
             (cx - 0.04 * L, y0 - 0.35 * L),
             (cx + 0.03 * L + 0.5 * (sway + gust) - 0.25 * L * bend, y0 - 0.72 * L + 0.12 * L * bend),
             (cx + sway + gust - 0.42 * L * bend, y0 - L * (1 - 0.45 * bend)))
        tx, ty = P[3][0] - P[2][0], P[3][1] - P[2][1]
        return dict(ph=ph, P=P, hx=P[3][0], hy=P[3][1], lean=np.arctan2(tx, -ty + 1e-6),
                    tilt=0.15 + 0.60 * ph["open"] + 0.12 * ph["puff"],
                    hs=0.25 + 0.75 * ph["bud"], gust=gust)

    @staticmethod
    def project(xl, yl, zl, hx, hy, lean, tilt):
        """Head-local coords (y along the stalk, z towards the viewer) -> screen x, y, depth."""
        ct, st = np.cos(tilt), np.sin(tilt)
        yp, zp = yl * ct - zl * st, yl * st + zl * ct
        cl, sl = np.cos(lean), np.sin(lean)
        return hx + xl * cl + yp * sl, hy - (yp * cl - xl * sl), zp

    def lit(self, col, x, y, ox, oy, size, resp, wash=0.0):
        """Low sunlight: shadows take the sky's tint, the side facing the sun catches a warm rim.
        `wash` lets light through thin things (seed silk) so they glow all over."""
        if self.theme["light"] is None:
            return col
        f = ((x - ox) * self.ldir[0] + (y - oy) * self.ldir[1]) / size
        k = (np.clip(0.5 + 0.6 * f, 0, 1) ** 2 + wash) * resp
        return col * self.amb + self.key * np.broadcast_to(k, np.shape(x))[:, None]

    # ---- sky ------------------------------------------------------------------------
    def build_sky(self):
        th, W, H, rng = self.theme, self.W, self.H, self.srng
        yy = np.arange(H, dtype=np.float32)[:, None]
        if th["sky"] is None:
            self.base = np.zeros((H, W, 3), np.float32)
            self.build_night_sky()
        else:
            grad = gradient(th["sky"], np.arange(H) / H)
            self.base = np.repeat((grad * 0.8)[:, None, :], W, axis=1)       # a darker ground under the dots
            n = int(W * self.y0 * 0.26)
            x, y = rng.uniform(0, W, n), rng.uniform(0, self.y0, n)
            col = gradient(th["sky"], y / H + rng.normal(0, 0.025, n)) * (1 + rng.normal(0, 0.045, (n, 1)))
            odd = rng.random(n) < 0.03                                       # a few dots from a little higher or lower
            col[odd] = gradient(th["sky"], y[odd] / H + rng.choice([-0.09, 0.09], odd.sum()))
            stamp(self.base, x, y, col.astype(np.float32),
                  rng.choice([1.0, 1.5, 2.0], n, p=[0.3, 0.45, 0.25]).astype(np.float32))
            self.build_clouds()
        self.build_hills()
        self.sky_mask = (yy < self.hill_top[None, :]).astype(np.float32)
        self.above = (yy < self.surface(np.arange(W, dtype=np.float32))[None, :]).astype(np.float32)
        self.build_sun()
        self.build_stars()

    def build_night_sky(self):
        W, H, rng = self.W, self.H, self.srng
        img = self.base
        n = 7000                                          # haze over the horizon
        y = self.y0 - (rng.random(n) ** 2.2) * 0.38 * H
        x = rng.uniform(0, W, n)
        glow(img, x, y, shades(rng, [(10, 14, 28), (16, 22, 44), (24, 30, 58)], rng.random(n), 0.8) * 0.8, 1.0)
        # the moon, with a soft halo and darker seas
        mx, my, mr = 0.76 * W, 0.20 * H, 0.052 * H
        self.moon = (mx, my, mr)
        n = 2600
        a, r = rng.uniform(0, 2 * np.pi, n), mr * (1.05 + rng.random(n) ** 1.8 * 1.6)
        glow(img, mx + np.cos(a) * r, my + np.sin(a) * r, np.array([34, 34, 44]) * (1 - (r / mr - 1) / 1.7)[:, None], 1.5)
        n = 6500
        a, r = rng.uniform(0, 2 * np.pi, n), mr * np.sqrt(rng.random(n))
        px, py = mx + np.cos(a) * r, my + np.sin(a) * r
        seas = sum(w * np.exp(-((px - mx - ox * mr) ** 2 + (py - my - oy * mr) ** 2) / (sr * mr) ** 2)
                   for ox, oy, sr, w in ((-0.3, -0.2, 0.35, 0.22), (0.25, 0.1, 0.3, 0.18),
                                         (0.05, 0.45, 0.22, 0.15), (-0.2, 0.35, 0.18, 0.12)))
        limb = 1 - 0.18 * (r / mr) ** 3
        col = shades(rng, MOON, rng.random(n), 0.8) * ((1 - seas) * limb)[:, None]
        stamp(img, px, py, col, rng.choice([1.0, 1.5], n, p=[0.6, 0.4]).astype(np.float32))

    def build_clouds(self):
        C = self.theme["clouds"]
        rng, W, H = self.srng, self.W, self.H
        for cx, cy, a, b, n in C["items"]:
            n, b = int(1.8 * n * W * H / 1e6), 2.4 * b
            dx = np.clip(rng.normal(0, 0.42, n), -1, 1)
            thick = (1 - dx ** 2) ** 0.8
            dy = rng.normal(0, 0.5, n) * thick
            x = (cx + a * dx) * W
            y = (cy + b * dy) * H + 0.006 * H * np.sin(dx * 5 + cx * 10)
            under = np.clip(dy / (thick + 1e-3) + 0.5, 0, 1) ** 0.8          # lit from below
            col = mix(shades(rng, C["shade"], rng.random(n)), shades(rng, C["lit"], rng.random(n)), under)
            stamp(self.base, x, y, col, rng.choice([1.0, 1.5, 2.0], n, p=[0.3, 0.45, 0.25]).astype(np.float32))

    def build_hills(self):
        rng, W, H = self.srng, self.W, self.H
        xs = np.arange(W, dtype=np.float32)
        self.hill_top = self.surface(xs).astype(np.float32)
        yy = np.arange(H, dtype=np.float32)[:, None]
        bottom = self.y0 + 0.03 * H
        for k, (base, amp, col) in enumerate(self.theme["hills"]):
            top = H * (base - amp * (0.55 + 0.45 * np.sin(xs / W * 2 * np.pi * 1.3 + k * 2.1))
                       - amp * 0.35 * np.sin(xs / W * 2 * np.pi * 3.7 + k))
            if k == 0:
                self.hill_top = top.astype(np.float32)
            col = np.asarray(col, np.float32)
            fill = (yy >= top[None, :]) & (yy < bottom)
            self.base[fill] = col * 0.82
            n = int(W * (bottom - top.min()) * 0.45)
            x = rng.uniform(0, W, n)
            ytop = np.interp(x, xs, top)
            depth = rng.random(n)
            y = ytop + depth * (bottom - ytop)
            c = col * (1.08 - 0.2 * depth)[:, None] * (1 + rng.normal(0, 0.07, (n, 1)))   # hazier at the crest
            stamp(self.base, x, y, c.astype(np.float32),
                  rng.choice([1.0, 1.5, 2.0], n, p=[0.3, 0.45, 0.25]).astype(np.float32))

    def build_sun(self):
        S = self.theme["sun"]
        self.sun = None
        if S is None:
            return
        rng, H = self.srng, self.H
        R = S["r"] * H
        reach = S["reach"] * R
        size = int(2 * reach) + 3
        c = size / 2
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        d = np.hypot(xx - c, yy - c)
        out = np.maximum(d - R, 0) / R
        halo = (0.75 * np.exp(-out / 0.8) + 0.35 * np.exp(-out / 2.6)) * np.clip(1 - d / reach, 0, 1) ** 0.5
        gcol = np.asarray(S["glow"], np.float32)
        halo_img = gcol * halo[..., None]
        n = int(8000 * (reach / (0.3 * H)) ** 2)                          # stipple the halo too
        a, r = rng.uniform(0, 2 * np.pi, n), R + (reach - R) * rng.random(n) ** 2.0
        w = np.clip(1 - (r - R) / (reach - R), 0, 1) ** 2 * 0.7 * ss(r, 1.05 * R, 1.8 * R)
        glow(halo_img, c + np.cos(a) * r, c + np.sin(a) * r, gcol * 0.45 * w[:, None], 1.5)
        core = np.empty((size, size, 3), np.float32)
        core[:] = np.asarray(S["core"][1], np.float32)
        n = int(np.pi * R * R * 0.9)
        a, rr = rng.uniform(0, 2 * np.pi, n), R * np.sqrt(rng.random(n))
        px, py = c + np.cos(a) * rr, c + np.sin(a) * rr
        g = np.clip((py - (c - R)) / (2 * R), 0, 1)                      # redder towards the bottom
        stamp(core, px, py, shades(rng, S["core"], g * 0.8 + 0.2 * rng.random(n), 0.4),
              rng.choice([1.0, 1.5], n).astype(np.float32))
        self.sun = dict(core=core, alpha=np.clip(R + 0.5 - d, 0, 1), glow=halo_img, c=c, size=size,
                        r=int(math.ceil(R)))

    def build_stars(self):
        rng, W, H = self.srng, self.W, self.H
        n, _, _, ymax = self.theme["stars"]
        sx, sy = rng.uniform(0, W, n), rng.random(n) ** 1.3 * ymax * H
        if self.theme["moon"]:
            mx, my, mr = self.moon
            keep = np.hypot(sx - mx, sy - my) > 2.4 * mr
            sx, sy = sx[keep], sy[keep]
        n = len(sx)
        self.stars = dict(x=sx, y=sy, b=rng.uniform(0.25, 1.0, n) ** 2, w=rng.uniform(0.6, 2.4, n),
                          ph=rng.uniform(0, 6.28, n), rad=rng.choice([1.0, 1.6], n, p=[0.85, 0.15]))

    # ---- the meadow and the plant ---------------------------------------------------
    def build_ground(self):
        W, H, rng, th = self.W, self.H, self.rng, self.theme
        tint = np.asarray(th["soil"], np.float32)
        if th["sky"] is not None:
            below = np.arange(H, dtype=np.float32)[:, None] >= self.surface(np.arange(W, dtype=np.float32))[None, :]
            self.base[below] = np.asarray(SOIL[0], np.float32) * tint * 0.7
        n = 24000
        x = rng.uniform(0, W, n)
        d = rng.random(n) ** 1.4
        y = self.surface(x) + d * (H - self.y0 + 0.02 * H)
        edge = np.clip(np.minimum(x, W - x) / (0.18 * W), 0, 1) ** 0.7
        clump = 0.5 + 0.5 * np.sin(x * 0.06 + y * 0.04) * np.sin(y * 0.1 - x * 0.025)
        b = (1 - d) ** 1.1 * edge * (0.5 + 0.35 * clump + 0.35 * rng.random(n))
        col = shades(rng, SOIL, rng.random(n), 1.0) * tint * b[:, None]
        stamp(self.base, x, y, col, rng.choice([1.0, 1.5, 2.0], n, p=[0.4, 0.4, 0.2]).astype(np.float32))
        # grass blades, swaying every frame
        nb, ns = 170, 14
        bx = rng.uniform(0, W, nb)
        bd = rng.random(nb)
        thin = (np.abs(bx - self.cx) < 0.13 * W) & (bd > 0.3) & (rng.random(nb) < 0.8)
        bx, bd = bx[~thin], bd[~thin]
        nb = len(bx)
        blade = np.repeat(np.arange(nb), ns * 2)
        s = np.tile(np.repeat(np.linspace(0, 1, ns), 2), nb)
        side = np.tile([-1.0, 1.0], nb * ns)
        self.grass = dict(
            blade=blade, s=s, side=side,
            bx=bx, by=self.surface(bx) + bd * 0.12 * H, d=bd,
            h=(0.018 + 0.075 * rng.random(nb) ** 1.6) * H * (0.6 + 0.8 * bd),
            lean=rng.normal(0, 0.28, nb), ph=rng.uniform(0, 6.28, nb),
            col=shades(rng, th["grass"], s * 0.85 + 0.15 * rng.random(len(s)), 0.4)
            * (0.4 + 0.6 * bd[blade])[:, None],
            rad=rng.choice([1.0, 1.5], len(s), p=[0.6, 0.4]).astype(np.float32))

    def build_leaves(self):
        rng, parts, n = self.rng, [], 1100
        for i, (az, ln, delay) in enumerate(LEAVES):
            u = sample_u(rng, n, leaf_width)
            v = sample_v(rng, n, 0.15)
            rib = rng.random(n) < 0.08
            v[rib] = rng.normal(0, 0.03, int(rib.sum()))
            col = shades(rng, LEAF, u * 0.9 + rng.normal(0, 0.12, n), 0.5)
            col[rib] = shades(rng, LEAF_RIB, rng.random(int(rib.sum())))
            parts.append(dict(id=np.full(n, i), u=u, v=v, col=col,
                              rad=rng.choice([1.0, 1.5, 2.0], n, p=[0.35, 0.45, 0.2]).astype(np.float32)))
        self.leaves = shuffle(join(parts), rng)
        self.leaf_az = np.array([l[0] for l in LEAVES])
        self.leaf_len = np.array([l[1] for l in LEAVES])
        self.leaf_delay = np.array([l[2] for l in LEAVES])

    def build_stem(self):
        rng, n = self.rng, 3200
        s = rng.random(n)
        lat = sample_v(rng, n, 0.2)
        col = shades(rng, STEM, s ** 0.6 + rng.normal(0, 0.08, n), 0.35)
        hi = lat > 0.55
        col[hi] = col[hi] * 1.25 + 12                       # a light edge on the hollow stalk
        self.stem = dict(s=s, lat=lat, col=col,
                         rad=rng.choice([1.0, 1.5, 2.0], n, p=[0.35, 0.45, 0.2]).astype(np.float32))

    def build_florets(self):
        rng, parts = self.rng, []
        width = lambda u: 0.3 + 0.7 * ss(u, 0.0, 0.3)
        for k, (cnt, ln, ao, ac, curl, cpos) in enumerate(RINGS):
            d = int(64 * ln)
            for i in range(cnt):
                th = 2 * np.pi * (i + 0.5 * (k % 2)) / cnt + rng.normal(0, 0.05)
                u = sample_u(rng, d, width)
                v = sample_v(rng, d, 0.15)
                u = np.minimum(u, 1 - 0.06 * (0.5 + 0.5 * np.cos(5 * np.pi * v)))   # toothed tips
                f = lambda val: np.full(d, val, np.float32)
                parts.append(dict(
                    th=f(th), ln=f(ln * rng.uniform(0.93, 1.07)), ao=f(ao), ac=f(ac), curl=f(curl),
                    aj=f(rng.normal(0, 0.05)), u=u, v=v,
                    live=shades(rng, PETAL, cpos * 0.7 + u * 0.35 + rng.normal(0, 0.1, d), 0.45),
                    dead=shades(rng, PETAL_DEAD, rng.random(d), 0.7), rn=rng.random(d),
                    rad=rng.choice([1.0, 1.5, 2.0], d, p=[0.3, 0.45, 0.25]).astype(np.float32)))
        self.flor = shuffle(join(parts), rng)

    def build_bracts(self):
        rng, parts = self.rng, []
        width = lambda u: np.sin(np.pi * np.clip(u, 0, 1) ** 0.7) ** 0.8
        for inner, cnt, ln, wd, d in ((1, 15, 0.62, 0.17, 70), (0, 11, 0.36, 0.20, 45)):
            for i in range(cnt):
                u, v = sample_u(rng, d, width), sample_v(rng, d)
                col = shades(rng, BRACT, u, 0.5)
                tip = (u > 0.84) & (rng.random(d) < 0.65)
                col[tip] = shades(rng, BRACT_TIP, rng.random(int(tip.sum())))
                f = lambda val: np.full(d, val, np.float32)
                parts.append(dict(
                    inner=f(inner), th=f(2 * np.pi * (i + 0.5 * inner) / cnt + rng.normal(0, 0.06)),
                    ln=f(ln * rng.uniform(0.9, 1.1)), wd=f(wd), u=u, v=v, col=col,
                    dry=shades(rng, BRACT_DRY, rng.random(d)),
                    rad=rng.choice([1.0, 1.5], d, p=[0.5, 0.5]).astype(np.float32)))
        self.bracts = join(parts)

    def seed_lengths(self, g):
        Lh = self.Lh
        return 0.11 * Lh, 0.10 * Lh * (0.6 + 0.4 * g), 0.80 * Lh * g, 0.34 * Lh * (0.25 + 0.75 * g), 0.12 + 1.18 * g

    def build_seeds(self):
        rng = self.rng
        # directions spread evenly over a sphere, minus the cap where the stalk joins
        m = 240
        k = np.arange(m) + 0.5
        yy = 1 - 2 * k / m
        rr = np.sqrt(1 - yy ** 2)
        ga = np.pi * (3 - np.sqrt(5)) * k
        dirs = np.stack([rr * np.cos(ga), yy, rr * np.sin(ga)], 1)
        dirs = dirs[dirs[:, 1] > -0.8]
        dirs += rng.normal(0, 0.03, dirs.shape)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        n = len(dirs)
        helper = np.where(np.abs(dirs[:, 1:2]) < 0.9, [[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]])
        e1 = np.cross(dirs, helper)
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(dirs, e1)

        # the seeds facing downwind (right) and up go first; a few hang on to the end
        order = 0.55 * rng.random(n) + 0.30 * (1 - (dirs[:, 0] + 1) / 2) + 0.15 * (1 - (dirs[:, 1] + 1) / 2)
        order[rng.random(n) < 0.03] = 1.6
        t_rel = 36.5 + 10.5 * order

        # where each seed is when it lets go
        pz = self.pose(t_rel)
        R0, La, Lb, Lf, beta = self.seed_lengths(1.0)
        c = dirs * (R0 + La + Lb)
        sx, sy, _ = self.project(c[:, 0], c[:, 1], c[:, 2], pz["hx"], pz["hy"], pz["lean"], pz["tilt"])
        ox, oy, _ = self.project(0 * sx, 0 * sx, 0 * sx, pz["hx"], pz["hy"], pz["lean"], pz["tilt"])
        far = rng.random(n) < 0.28
        self.seed = dict(
            d=dirs, e1=e1, e2=e2, t_rel=t_rel, sx=sx, sy=sy, a0=np.arctan2(sx - ox, -(sy - oy)),
            u=self.W * 0.075 * rng.uniform(0.5, 1.5, n),
            vy=self.H * 0.04 * rng.uniform(-0.4, 1.6, n),
            ax=self.W * 0.04 * rng.random(n), wx=rng.uniform(0.4, 1.4, n), px=rng.uniform(0, 6.28, n),
            ay=self.H * 0.05 * rng.random(n), wy=rng.uniform(0.5, 1.6, n), py=rng.uniform(0, 6.28, n),
            near=np.where(far, rng.uniform(0.08, 0.25, n), 0.0),
            spin=rng.uniform(0.5, 1.5, n) * rng.choice([-1, 1], n),
            up=0.25 + rng.uniform(-0.2, 0.2, n), wob=rng.uniform(0, 6.28, n))

        # the dots that make up each seed: 3 on the seed body, 5 on the beak, 10 x 2 on the parachute
        nf = 10
        kind = np.r_[np.zeros(3), np.ones(5), np.full(nf * 2, 2)]
        frac = np.r_[np.linspace(0, 1, 3), np.linspace(0.12, 1, 5), np.tile([0.55, 1.0], nf)]
        fil = np.r_[np.zeros(8), np.repeat(np.arange(nf), 2)]
        per = len(kind)
        sid = np.repeat(np.arange(n), per)
        kind, frac = np.tile(kind, n), np.tile(frac, n)
        psi = 2 * np.pi * np.tile(fil, n) / nf + rng.normal(0, 0.12, n * per) + np.repeat(rng.uniform(0, 6.28, n), per)
        col = np.where((kind == 2)[:, None], shades(rng, SILK, frac * 0.3 + rng.random(n * per) * 0.7, 0.6),
                       np.where((kind == 1)[:, None], shades(rng, SILK[1:], rng.random(n * per)) * 0.9,
                                shades(rng, ACHENE, rng.random(n * per))))
        rad = np.where(kind == 0, 1.5, 1.0)
        self.seedpts = dict(sid=sid, kind=kind, frac=frac, psi=psi, col=col.astype(np.float32),
                            rad=rad.astype(np.float32),
                            resp=np.where(kind == 2, 0.85, np.where(kind == 1, 0.6, 0.25)))

    def build_flecks(self):
        """Dried florets that fall away as the seed clock pushes up."""
        rng, n = self.rng, 180
        ts = 28.6 + 2.4 * rng.random(n)
        pz = self.pose(ts)
        Lh = self.Lh
        lx, ly, lz = rng.uniform(-0.15, 0.15, n) * Lh, (0.55 + 0.2 * rng.random(n)) * Lh, rng.uniform(-0.15, 0.15, n) * Lh
        sx, sy, _ = self.project(lx, ly, lz, pz["hx"], pz["hy"], pz["lean"], pz["tilt"])
        g = 0.03 * self.H
        land = self.ground + rng.uniform(-0.01, 0.03, n) * self.H
        self.flecks = dict(ts=ts, sx=sx, sy=sy, g=g, land=land, tl=np.sqrt(2 * np.maximum(land - sy, 1) / g),
                           vx=self.W * rng.uniform(-0.02, 0.07, n),
                           ph=rng.uniform(0, 6.28, n), col=shades(rng, PETAL_DEAD, rng.random(n), 0.7),
                           rad=np.full(n, 1.5, np.float32))

    def build_fx(self):
        rng = self.rng
        n = 14
        self.flies = dict(a=rng.uniform(0.04, 0.10, n), b=rng.uniform(0.05, 0.12, n),
                          p1=rng.uniform(0, 6.28, n), p2=rng.uniform(0, 6.28, n),
                          blink=rng.uniform(0.5, 1.2, n), p3=rng.uniform(0, 6.28, n))
        n = 90
        self.pollen = dict(ph=rng.random(n), sp=rng.uniform(0.6, 1.4, n), ang=rng.uniform(0, 6.28, n),
                           r=rng.random(n), drift=rng.normal(0, 1, n))
        n = 500
        self.halo = dict(a=rng.uniform(0, 6.28, n), r=0.9 + rng.random(n) ** 1.5 * 1.3, fl=rng.random(n))
        n = 40
        self.streaks = dict(y=self.H * (0.12 + 0.68 * rng.random(n)), sp=rng.uniform(0.7, 1.3, n),
                            ph=rng.random(n), w=rng.uniform(0, 6.28, n))

    def build_weather(self):
        th, rng, W, H = self.theme, self.srng, self.W, self.H
        nb = th["birds"]
        self.birds = None if not nb else dict(
            p=rng.uniform(24, 40, nb), off=rng.random(nb), y=rng.uniform(0.14, 0.42, nb),
            ph=rng.uniform(0, 6.28, nb), w=rng.uniform(5.0, 8.0, nb), s=rng.uniform(0.008, 0.014, nb),
            dir=rng.choice([-1.0, 1.0], nb, p=[0.3, 0.7]))
        if th["dew"]:
            n = 110
            self.dew_leaf = dict(i=rng.integers(0, len(LEAVES), n), u=rng.uniform(0.3, 0.95, n),
                                 v=rng.uniform(-0.75, 0.75, n), ph=rng.uniform(0, 6.28, n), w=rng.uniform(0.8, 2.2, n))
            nb = len(self.grass["bx"])
            self.grass_dew = dict(on=rng.random(nb) < 0.45, ph=rng.uniform(0, 6.28, nb), w=rng.uniform(0.8, 2.2, nb))
            sp = self.seedpts
            m = len(sp["kind"])
            self.seed_dew = dict(on=(sp["kind"] == 2) & (sp["frac"] > 0.9) & (rng.random(m) < 0.07),
                                 ph=rng.uniform(0, 6.28, m), w=rng.uniform(0.8, 2.2, m))
        if th["mist"]:
            n = 1800
            y = np.clip(H * (0.775 + rng.normal(0, 0.035, n)), 0.66 * H, 0.93 * H)
            self.mist = dict(x=rng.uniform(0, W, n), y=y, sp=rng.uniform(0.002, 0.008, n) * W,
                             b=np.exp(-((y / H - 0.775) / 0.045) ** 2) * rng.uniform(0.5, 1.0, n),
                             front=rng.random(n) < 0.3, ph=rng.uniform(0, 6.28, n))
        if th["rays"]:
            n = th["rays"]
            self.rays = dict(ang=np.sort(rng.uniform(-0.55, 0.45, n)), wd=rng.uniform(0.006, 0.02, n) * H,
                             b=rng.uniform(0.4, 1.0, n), ph=rng.uniform(0, 6.28, n),
                             d=rng.random((n, 110)), jit=rng.normal(0, 1, (n, 110)))
        if th["motes"]:
            n = th["motes"]
            self.motes = dict(x0=rng.uniform(0.1, 0.9, n), y0=rng.uniform(0.35, 0.8, n),
                              sp=rng.uniform(0.6, 1.4, n), ph=rng.random(n), w=rng.uniform(0, 6.28, n))

    # ---- drawing ----------------------------------------------------------------------
    def render(self, t):
        pz = self.pose(t)
        ph = pz["ph"]
        f0, f1 = self.theme["sky_fade"]
        fade = f0 + (f1 - f0) * min(t / T_END, 1.0)
        img = self.base * np.float32(fade) if fade != 1.0 else self.base.copy()
        self.dew = []

        self.draw_back(img, t, pz)

        dots = []                       # (x, y, colour, radius, depth); nearer dots drawn last
        self.draw_birds(dots, t)
        self.draw_grass(dots, t, pz)
        self.draw_leaves(dots, t, ph)
        if ph["grow"] > 0.01:
            self.draw_stem(dots, pz)
        if ph["bud"] > 0.02 and ph["grow"] > 0.02:
            self.draw_head(dots, pz)
        if ph["puff"] > 0.01:
            self.draw_seeds(dots, t, pz)
        self.draw_flecks(dots, t)

        x, y, col, rad, z = (np.concatenate([np.broadcast_to(np.asarray(p[i], np.float32),
                                                             (len(p[0]),) + ((3,) if i == 2 else ()))
                                             for p in dots]) for i in range(5))
        o = np.argsort(z, kind="stable")
        stamp(img, x[o], y[o], np.clip(col[o], 0, 255), rad[o])

        self.draw_lights(img, t, pz)
        return np.clip(img, 0, 255).astype(np.uint8)

    def draw_back(self, img, t, pz):
        """Everything behind the plant that changes: stars, sun, light beams, far mist."""
        th, W, H = self.theme, self.W, self.H
        n, a, b, _ = th["stars"]
        amt = 1.0 if a == b else float(ss(t, a, b))
        if n and amt > 0.01:
            S = self.stars
            tw = S["b"] * (0.55 + 0.45 * np.sin(t * S["w"] + S["ph"])) * amt
            glow(img, S["x"], S["y"], np.array([210, 215, 255]) * tw[:, None], S["rad"])
        if th["venus"]:                                   # the evening star comes out first
            v = float(ss(t, 26.0, 40.0))
            if v > 0.01:
                glow(img, 0.17 * W, 0.13 * H, np.array([255, 250, 232]) * v, 2.2)
                glow(img, 0.17 * W, 0.13 * H, np.array([90, 90, 110]) * v, 3.0)
        if self.sun is not None:
            sx, sy = self.draw_sun(img, t)
            if th["rays"]:
                R = self.rays
                k = 0.6 + 0.4 * np.sin(0.35 * t + R["ph"])
                L = 1.3 * W * R["d"]
                ca, sa = np.cos(R["ang"])[:, None], np.sin(R["ang"])[:, None]
                x = sx + ca * L - sa * R["jit"] * R["wd"][:, None]
                y = sy + sa * L + ca * R["jit"] * R["wd"][:, None]
                fade = 1 - 0.5 * ss(t, 10.0, 50.0)
                b = (R["b"] * k)[:, None] * (1 - R["d"]) ** 1.5 * 0.22 * fade
                glow(img, x.ravel(), y.ravel(), np.array([255, 236, 200]) * b.ravel()[:, None], 2.0)
        if th["mist"]:
            self.draw_mist(img, t, front=False)

    def draw_sun(self, img, t):
        S, sp = self.theme["sun"], self.sun
        k = min(t / T_END, 1.0)
        sx = S["x"] * self.W
        sy = self.H * (S["y0"] + (S["y1"] - S["y0"]) * k)
        gk = S["glow_t"][0] + (S["glow_t"][1] - S["glow_t"][0]) * k
        size = sp["size"]
        x0, y0 = int(round(sx - sp["c"])), int(round(sy - sp["c"]))
        xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + size, self.W), min(y0 + size, self.H)
        # the disc first, only inside its own small box; it hides behind the hills
        r = sp["r"] + 2
        dx0, dy0 = int(round(sx - r)), int(round(sy - r))
        dxa, dya = max(dx0, 0), max(dy0, 0)
        dxb, dyb = min(dx0 + 2 * r + 1, self.W), min(dy0 + 2 * r + 1, self.H)
        if dxa < dxb and dya < dyb:
            dst = (slice(dya, dyb), slice(dxa, dxb))
            src = (slice(dya - y0, dyb - y0), slice(dxa - x0, dxb - x0))
            a = (sp["alpha"][src] * self.sky_mask[dst])[..., None]
            img[dst] = img[dst] * (1 - a) + sp["core"][src] * a
        # then the glow over everything, disc included, so the disc stays the brightest thing
        if xa < xb and ya < yb:
            dst = (slice(ya, yb), slice(xa, xb))
            src = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
            img[dst] += sp["glow"][src] * (gk * self.above[dst])[..., None]
        return sx, sy

    def draw_mist(self, img, t, front):
        M = self.mist
        amt = self.theme["mist"] * (1 - 0.6 * ss(t, 8.0, 50.0))           # the mist lifts as the sun climbs
        m = M["front"] if front else ~M["front"]
        x = (M["x"][m] + M["sp"][m] * t) % self.W
        y = M["y"][m] + 0.004 * self.H * np.sin(0.3 * t + M["ph"][m])
        glow(img, x, y, np.array([255, 246, 236]) * (M["b"][m] * amt * 0.26)[:, None], 3.0)

    def draw_birds(self, dots, t):
        B = self.birds
        if B is None:
            return
        W, H = self.W, self.H
        prog = (t / B["p"] + B["off"]) % 1.0
        bx = W * (0.5 + B["dir"] * (-0.58 + 1.16 * prog))
        by = B["y"] * H + 0.012 * H * np.sin(t * 0.6 + B["ph"])
        flap = np.sin(t * B["w"] + B["ph"])
        span = (B["s"] * H)[:, None]
        j = np.arange(1, 6) / 5
        wx = np.r_[-j, j][None, :]
        wy = np.r_[j, j][None, :] ** 1.4
        X = np.c_[bx, bx[:, None] + span * wx]
        Y = np.c_[by, by[:, None] - span * wy * (0.15 + 0.55 * flap[:, None])]
        n = X.size
        rad = np.tile(np.r_[1.5, np.ones(10)], len(bx)).astype(np.float32)
        dots.append((X.ravel(), Y.ravel(), np.tile(np.array([[40, 22, 38]], np.float32), (n, 1)), rad,
                     np.full(n, -1e9)))

    def draw_grass(self, dots, t, pz):
        G = self.grass
        b = G["blade"]
        gust = pz["gust"] / self.H
        bend = (G["lean"] + 0.10 * np.sin(1.2 * t + G["bx"] / self.W * 9 + G["ph"])
                + gust * 16 * (0.65 + 0.35 * np.sin(3.1 * t + G["bx"] * 0.02)))
        s = G["s"]
        a = bend[b] * s ** 1.3 * 0.7
        h = G["h"][b]
        x = G["bx"][b] + h * s * np.sin(a) + G["side"] * 0.0025 * self.H * (1 - s) * (0.6 + 0.8 * G["d"][b])
        y = G["by"][b] - h * s * np.cos(a)
        z = (G["by"] - self.ground) / DEPTH
        col = self.lit(G["col"], x, y, G["bx"][b], (G["by"] - 0.5 * G["h"])[b], (0.5 * G["h"])[b], 0.9 * s ** 1.5)
        dots.append((x, y, col, G["rad"], z[b]))
        if self.theme["dew"]:                                   # a drop hanging from many blade tips
            D = self.grass_dew
            on = D["on"]
            at = bend[on] * 0.7
            tx = G["bx"][on] + G["h"][on] * np.sin(at)
            ty = G["by"][on] - G["h"][on] * np.cos(at) + 1.0
            dots.append((tx, ty, np.tile(DEW, (len(tx), 1)), np.full(len(tx), 1.5, np.float32), z[on] + 0.5))
            self.dew.append((tx, ty, D["ph"][on], D["w"][on]))

    def leaf_pose(self, i, u, v, t, ph):
        lf = ss(ph["leaf"] * 1.35 - self.leaf_delay, 0.0, 1.0)[i]
        L = 0.21 * self.H * self.leaf_len[i] * lf
        az = self.leaf_az[i] + 0.02 * np.sin(1.5 * t + i) * (1 + 3 * ph["wind"])
        rise = 0.35 + 0.9 * (1 - lf)                       # young leaves stand up, then lie down
        am = rise - 0.55 * lf * u
        along, height = L * u * np.cos(am), L * u * np.sin(am)
        lat = v * 0.20 * L * (0.4 + 0.6 * lf) * leaf_width(u)
        height = height + 0.45 * np.abs(lat)               # the halves fold up along the midrib
        gx = np.cos(az) * along - np.sin(az) * lat
        gz = np.sin(az) * along + np.cos(az) * lat
        return self.cx + gx, self.ground - height + DEPTH * gz, gz, lf

    def draw_leaves(self, dots, t, ph):
        if ph["leaf"] <= 0.01:
            return
        Lv = self.leaves
        X, Y, gz, lf = self.leaf_pose(Lv["id"], Lv["u"], Lv["v"], t, ph)
        show = lf > 0.02
        X, Y, gz, u = X[show], Y[show], gz[show], Lv["u"][show]
        shade = np.clip(0.62 + 0.3 * gz / (0.21 * self.H) + 0.22 * u, 0.4, 1.15)
        col = self.lit(Lv["col"][show] * shade[:, None], X, Y, self.cx, self.ground - 0.03 * self.H,
                       0.21 * self.H, 0.35)
        dots.append((X, Y, col, Lv["rad"][show], gz))
        if self.theme["dew"]:
            D = self.dew_leaf
            X, Y, gz, lf = self.leaf_pose(D["i"], D["u"], D["v"], t, ph)
            on = lf > 0.7
            if on.any():
                dots.append((X[on], Y[on], np.tile(DEW, (int(on.sum()), 1)),
                             np.full(int(on.sum()), 1.5, np.float32), gz[on] + 0.5))
                self.dew.append((X[on], Y[on], D["ph"][on], D["w"][on]))

    def draw_stem(self, dots, pz):
        S = self.stem
        x, y, dx, dy = bezier(pz["P"], S["s"])
        n = np.hypot(dx, dy) + 1e-6
        w = 0.0055 * self.H * (1 - 0.25 * S["s"]) * np.clip(0.4 + 1.5 * pz["ph"]["grow"], 0.4, 1.0)
        lat = S["lat"] * w
        X, Y = x - dy / n * lat, y + dx / n * lat
        dots.append((X, Y, self.lit(S["col"], X, Y, x, y, w, 0.6), S["rad"], np.zeros(len(x))))

    def draw_head(self, dots, pz):
        ph, Lh = pz["ph"], self.Lh * pz["hs"]
        o, w, p = float(ph["open"]), float(ph["wither"]), float(ph["puff"])
        args = (pz["hx"], pz["hy"], pz["lean"], pz["tilt"])
        hcx, hcy, _ = self.project(0.0, 0.2 * Lh, 0.0, *args)

        keep = float(1 - ss(p, 0.0, 0.45))
        if keep > 0.01:
            F = self.flor
            nv = int(len(F["u"]) * keep)
            g = {k: val[:nv] for k, val in F.items()}
            u, v = g["u"], g["v"]
            ext = max(o, 0.8 * w)
            ell = Lh * g["ln"] * (0.5 + 0.5 * ext)
            alpha = g["ac"] + (g["ao"] - g["ac"]) * o + g["aj"]
            am = alpha + (0.30 * o - 0.10) * g["curl"] * u * 0.5
            r, h = ell * u * np.sin(am), ell * u * np.cos(am) + 0.10 * Lh
            lat = v * 0.10 * ell * (0.4 + 0.6 * o)
            ct, st = np.cos(g["th"]), np.sin(g["th"])
            X, Y, zp = self.project(ct * r - st * lat, h, st * r + ct * lat, *args)
            wi = np.clip(w * (1.3 + 0.4 * u) - 0.35 * g["rn"], 0, 1)
            shade = 0.62 + 0.38 * np.clip(zp / Lh * 0.5 + 0.5, 0, 1) + 0.08 * h / Lh
            col = self.lit(mix(g["live"], g["dead"], wi) * shade[:, None], X, Y, hcx, hcy, Lh, 0.45)
            dots.append((X, Y, col, g["rad"], zp))

        B = self.bracts
        inner = B["inner"] > 0.5
        a_in = (0.10 + 1.65 * o) * (1 - p) + 2.5 * p       # fold down under the open flower
        a_out = (1.25 + 0.45 * max(o, w)) * (1 - p) + 2.75 * p
        alpha = np.where(inner, a_in, a_out)
        curl = np.where(inner, -0.15 * (1 - p) + 0.6 * p, 0.5 + 0.3 * p)
        u, v = B["u"], B["v"]
        ell = Lh * B["ln"]
        am = alpha + curl * u * 0.5
        bulge = np.where(inner, 0.12 * Lh * np.sin(np.pi * u) * (1 - o) * (1 - p), 0.0)   # plump bud
        r = 0.13 * Lh + ell * u * np.sin(am) + bulge
        h = 0.04 * Lh + ell * u * np.cos(am)
        lat = v * B["wd"] * ell * np.sin(np.pi * np.clip(u, 0, 1) ** 0.7) ** 0.8
        ct, st = np.cos(B["th"]), np.sin(B["th"])
        X, Y, zp = self.project(ct * r - st * lat, h, st * r + ct * lat, *args)
        shade = 0.6 + 0.4 * np.clip(zp / Lh * 0.5 + 0.5, 0, 1)
        col = mix(B["col"], B["dry"], np.full(len(u), 0.6 * p)) * shade[:, None]
        dots.append((X, Y, self.lit(col, X, Y, hcx, hcy, Lh, 0.35), B["rad"], zp))

        # the receptacle: a pale dome, bare once the seeds have flown
        if p > 0.3:
            n = 160
            a = np.linspace(0, 2 * np.pi, n, endpoint=False) * 7.0
            rr = 0.11 * Lh * np.sqrt(np.linspace(0, 1, n))
            X, Y, zp = self.project(np.cos(a) * rr, 0.06 * Lh + 0 * a, np.sin(a) * rr, *args)
            col = np.where((np.arange(n) % 5 == 0)[:, None], [[120, 104, 80]], [[214, 204, 168]]).astype(np.float32)
            dots.append((X, Y, self.lit(col, X, Y, hcx, hcy, Lh, 0.3), np.full(n, 1.5, np.float32), zp - 1))

    def draw_seeds(self, dots, t, pz):
        Sd, Sp = self.seed, self.seedpts
        g = float(pz["ph"]["puff"])
        R0, La, Lb, Lf, beta = self.seed_lengths(g)
        sid, kind, frac, psi = Sp["sid"], Sp["kind"], Sp["frac"], Sp["psi"]
        attached = t < Sd["t_rel"][sid]

        # still on the clock
        m = attached
        if m.any():
            d, e1, e2 = Sd["d"][sid[m]], Sd["e1"][sid[m]], Sd["e2"][sid[m]]
            k, f, ps = kind[m], frac[m], psi[m]
            along = np.where(k == 0, R0 + f * La, R0 + La + np.where(k == 1, f * Lb, Lb))
            fdir = (np.cos(beta) * d + np.sin(beta) * (np.cos(ps)[:, None] * e1 + np.sin(ps)[:, None] * e2))
            pos = along[:, None] * d + np.where((k == 2)[:, None], (f * Lf)[:, None] * fdir, 0.0)
            args = (pz["hx"], pz["hy"], pz["lean"], pz["tilt"])
            X, Y, zp = self.project(pos[:, 0], pos[:, 1] + 0.06 * self.Lh, pos[:, 2], *args)
            cx, cy, _ = self.project(0.0, 0.06 * self.Lh, 0.0, *args)
            R = R0 + La + Lb + Lf
            front = np.clip(zp / R * 0.5 + 0.5, 0, 1)
            # silk is faint and see-through: the beaks read as spokes, the parachutes as a fuzzy shell
            bright = np.where(k == 2, 0.26 + 0.42 * front + 0.18 * f, np.where(k == 1, 0.22 + 0.33 * front, 0.5 + 0.5 * front))
            col = self.lit(Sp["col"][m] * bright[:, None], X, Y, cx, cy, R, Sp["resp"][m] * bright, wash=0.35)
            dots.append((X, Y, col, Sp["rad"][m], zp))
            if self.theme["dew"]:
                dm = self.seed_dew["on"][m]
                if dm.any():
                    self.dew.append((X[dm], Y[dm], self.seed_dew["ph"][m][dm], self.seed_dew["w"][m][dm]))

        # on the wind
        m = ~attached
        if m.any():
            s = sid[m]
            fl = t - Sd["t_rel"][s]
            ease = 1 - np.exp(-0.8 * fl)
            x = Sd["sx"][s] + Sd["u"][s] * (fl + 0.12 * fl * fl) + Sd["ax"][s] * np.sin(Sd["wx"][s] * fl + Sd["px"][s]) * ease
            y = Sd["sy"][s] - Sd["vy"][s] * fl + Sd["ay"][s] * np.sin(Sd["wy"][s] * fl + Sd["py"][s]) * ease
            sc = np.minimum(1 + Sd["near"][s] * fl, 2.4)            # some drift towards the viewer
            a = (Sd["up"][s] + (Sd["a0"][s] - Sd["up"][s]) * np.exp(-1.8 * fl)
                 + 0.22 * np.sin(2.6 * fl + Sd["wob"][s]) * np.exp(-0.15 * fl))
            Ax, Ay, Sx, Sy = np.sin(a), -np.cos(a), np.cos(a), np.sin(a)
            R0, La, Lb, Lf, beta = self.seed_lengths(1.0)
            k, f = kind[m], frac[m]
            ps = psi[m] + Sd["spin"][s] * fl
            al = np.where(k == 0, -(Lb + La * (1 - f)), np.where(k == 1, -Lb * (1 - f),
                          f * Lf * (np.cos(beta) + 0.35 * np.sin(beta) * np.sin(ps))))
            sd = np.where(k == 2, f * Lf * np.sin(beta) * np.cos(ps), 0.0)
            X = x + sc * (al * Ax + sd * Sx)
            Y = y + sc * (al * Ay + sd * Sy)
            sg = self.theme["seed_glow"]                            # against a bright sky the silk must shine
            fade = np.clip(1 - (fl - 9.0) / 3.0, 0, 1) * np.where(k == 0, 0.45, np.where(k == 1, 0.6 + 0.4 * sg,
                                                                                         0.85 + 0.15 * sg))
            col = self.lit(Sp["col"][m] * fade[:, None], X, Y, x, y, Lf * sc + 1, Sp["resp"][m] * fade, wash=0.5)
            dots.append((X, Y, col, np.minimum(Sp["rad"][m], 1.0), 1e4 + sc * 10))   # small seed bodies in flight

    def draw_flecks(self, dots, t):
        F = self.flecks
        fl = t - F["ts"]
        m = fl > 0
        if not m.any():
            return
        fl, tl = fl[m], F["tl"][m]
        fe = np.minimum(fl, tl)
        air = fl < tl
        x = F["sx"][m] + F["vx"][m] * fe + 0.02 * self.W * np.sin(2.2 * fe + F["ph"][m]) * air
        y = np.minimum(F["sy"][m] + 0.5 * F["g"] * fe ** 2 + 0.01 * self.H * np.sin(3 * fe + F["ph"][m]) * air,
                       F["land"][m])
        z = np.where(air, 9e3, (F["land"][m] - self.ground) / DEPTH)
        dots.append((x, y, self.lit(F["col"][m], x, y, x, y, 1.0, 0.6), F["rad"][m], z))

    def draw_lights(self, img, t, pz):
        th, ph, H, W = self.theme, pz["ph"], self.H, self.W
        # fireflies over the meadow
        s0, s1, amax = th["flies"]
        amt = amax if s0 == s1 else amax * float(ss(t, s0, s1))
        if amt > 0.01:
            Fz = self.flies
            fx = W * (0.08 + 0.84 * (0.5 + 0.5 * np.sin(t * Fz["a"] + Fz["p1"])))
            fy = H * (0.48 + 0.26 * (0.5 + 0.5 * np.sin(t * Fz["b"] + Fz["p2"])))
            bl = np.clip(np.sin(t * Fz["blink"] + Fz["p3"]), 0, 1) ** 3 * amt
            glow(img, fx, fy, np.array([150, 200, 70]) * 0.45 * bl[:, None], 3.0)
            glow(img, fx, fy, np.array([220, 255, 140]) * bl[:, None], 1.2)

        fx_amt = float(ph["bloom_fx"])
        if fx_amt > 0.01:
            Lh = self.Lh * pz["hs"]
            cx, cy, _ = self.project(0.0, 0.25 * Lh, 0.0, pz["hx"], pz["hy"], pz["lean"], pz["tilt"])
            Hl = self.halo
            r = Hl["r"] * Lh
            glow(img, cx + np.cos(Hl["a"]) * r, cy + np.sin(Hl["a"]) * r * 0.8,
                 np.array([46, 34, 6]) * fx_amt * (0.6 + 0.4 * np.sin(2 * t + Hl["fl"] * 9))[:, None], 1.0)
            P = self.pollen
            life = (t * 0.18 * P["sp"] + P["ph"]) % 1.0
            px = cx + np.cos(P["ang"]) * Lh * (0.3 + 0.9 * P["r"]) + P["drift"] * life * 0.04 * W
            py = cy + np.sin(P["ang"]) * Lh * 0.4 - life * 0.16 * H
            glow(img, px, py, np.array([255, 205, 90]) * (np.sin(np.pi * life) * fx_amt * 0.8)[:, None], 1.2)

        wnd = float(ph["wind"])
        if wnd > 0.01:
            St = self.streaks
            head = ((t * 0.25 * St["sp"] + St["ph"]) % 1.4 - 0.2) * W
            j = np.arange(16)
            xs = (head[:, None] - j[None, :] * 4.0 * W / 1000).ravel()
            ys = (St["y"][:, None] + 0.004 * H * np.sin(xs.reshape(-1, 16) / W * 12 + St["w"][:, None])).ravel()
            b = (wnd * 0.16 * (1 - j / 16))[None, :].repeat(len(head), 0).ravel()
            glow(img, xs, ys, np.array(th["streak"], np.float32) * b[:, None], 1.0)

        if th["motes"]:                                    # dust turning gold in the low light
            M = self.motes
            life = (t * 0.05 * M["sp"] + M["ph"]) % 1.0
            x = (M["x0"] + 0.06 * np.sin(t * 0.2 * M["sp"] + M["w"]) + 0.08 * life) * W
            y = (M["y0"] - 0.10 * life) * H
            b = np.sin(np.pi * life) * (0.5 + 0.5 * np.sin(t * 2 * M["sp"] + M["w"] * 3)) * 0.55
            glow(img, x, y, np.array([255, 236, 200]) * b[:, None], 1.3)
        if th["mist"]:
            self.draw_mist(img, t, front=True)
        for x, y, ph_, w in self.dew:                      # dew drops catch the sun and wink
            tw = np.clip(np.sin(t * w + ph_), 0, 1) ** 10
            glow(img, x, y, np.array([255, 255, 255]) * 0.95 * tw[:, None], 2.5)
            glow(img, x, y, np.array([110, 125, 140]) * 0.35, 1.0)


# ---- exports and window -------------------------------------------------------------
def prefix(theme):
    return "dandelion" if theme == "night" else f"dandelion_{theme}"


def save_png(theme="night"):
    size = 1000
    d = Dandelion(size, size, theme)
    times = [5.5, 14.5, 25.0, 34.5, 41.0]
    strip = Image.new("RGB", (500 * len(times), 500))
    for i, t in enumerate(times):
        frame = Image.fromarray(d.render(t))
        frame.save(f"{prefix(theme)}_{i + 1}.png")
        strip.paste(frame.resize((500, 500), Image.LANCZOS), (500 * i, 0))
    strip.save(f"{prefix(theme)}_stages.png")
    print(f"Saved: {prefix(theme)}_stages.png and {prefix(theme)}_1..5.png")


def save_gif(theme="night", fps=12, size=640):
    d = Dandelion(size, size, theme)
    frames = [Image.fromarray(d.render(i / fps)).convert("P", palette=Image.ADAPTIVE)
              for i in range(int(T_END * fps) + 1)]
    frames[0].save(f"{prefix(theme)}.gif", save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=False)
    print(f"Saved: {prefix(theme)}.gif")


class App:
    def __init__(self, theme="night"):
        import tkinter as tk
        from PIL import ImageTk

        self.theme = theme
        self.root = tk.Tk()
        self.root.title(THEMES[theme]["title"])
        self.root.configure(bg="black")
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        size = int(min(sh * WIN_FRAC, sw * 0.96))
        self.root.geometry(f"{size}x{size}+{(sw - size) // 2}+{max(0, (sh - size) // 2 - 20)}")
        self.canvas = tk.Canvas(self.root, width=size, height=size, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_text(size // 2, size // 2, text="planting...", fill="#777777")
        self.root.update()

        self.plant = Dandelion(size, size, theme)
        self.photo = ImageTk.PhotoImage(Image.fromarray(self.plant.render(0.0)))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("<space>", lambda e: self.toggle())
        self.root.bind("<r>", lambda e: self.restart())
        self.root.bind("<s>", lambda e: self.snapshot())
        self.clock, self.last, self.paused = 0.0, time.perf_counter(), False
        self.tick()

    def toggle(self):
        self.paused = not self.paused

    def restart(self):
        self.clock = 0.0

    def snapshot(self):
        Image.fromarray(self.plant.render(min(self.clock, T_END))).save(f"{prefix(self.theme)}_frame.png")

    def tick(self):
        now = time.perf_counter()
        if not self.paused:
            self.clock += now - self.last
        self.last = now
        if LOOP and self.clock > T_END + 3:
            self.clock = 0.0
        self.photo.paste(Image.fromarray(self.plant.render(min(self.clock, T_END))))
        self.root.after(TICK_MS, self.tick)

    def run(self):
        self.root.mainloop()


def main(theme=None, argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--theme" in argv and argv.index("--theme") + 1 < len(argv):
        theme = argv[argv.index("--theme") + 1]
    theme = theme or "night"
    if theme not in THEMES:
        sys.exit(f"unknown theme {theme!r}; pick one of: {', '.join(THEMES)}")
    if "--png" in argv:
        save_png(theme)
    elif "--gif" in argv:
        save_gif(theme)
    else:
        App(theme).run()


if __name__ == "__main__":
    main()
