"""
night_bloom.py - a Queen of the Night cactus blooming for its one night, in dots.

The sun has gone down behind the red rock and the desert turns blue. On a ribbed cactus a
long-tubed bud lifts and swells. Stars come out, the moon climbs, and the bud slowly opens:
the bronze outer petals peel back like rays, the white inner petals spread into a wide cup,
and a crown of cream stamens and a starry stigma rise from it, glowing in the moonlight. A
hawk moth comes whirring in, hovers, and drinks from the flower with its long tongue, dusted
with pollen. By the time the sky pales in the east the moon has set; the first sun turns
the tops of the mesas red, and the flower, its night over, folds and droops.

    python night_bloom.py            interactive window (Space pause, R restart, S save frame, Esc quit)
    python night_bloom.py --png      save five stages
    python night_bloom.py --gif      save the whole night as a GIF
"""
import math

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))   # the shared engine: engine/pointillism.py

from pointillism import (Backdrop, Haze, Moon, Sun, bezier, glow, keyed, lit, mix, paint, run, sample_v, shades, sky,
                         ss, stamp, stars, unit, warm_glow)

T_END = 60.0
STAGES = [3.0, 16.0, 27.0, 36.5, 56.5]
TAU = 2 * math.pi
HORIZON = 0.665

PETAL = [(250, 248, 240), (244, 242, 232), (236, 236, 226), (252, 250, 246)]
OUTER = [(176, 120, 76), (196, 150, 96), (214, 182, 128), (150, 96, 64)]
STAMEN = [(240, 232, 196), (232, 222, 180)]
ANTHER = [(236, 204, 90), (226, 190, 70)]
STEM = [(54, 88, 52), (66, 104, 60), (80, 118, 70), (46, 74, 46)]
ROCK = [(176, 80, 48), (190, 96, 58), (160, 70, 44), (204, 112, 70)]
SAND = [(150, 112, 84), (170, 128, 94), (132, 98, 76)]
BRUSH = [(46, 50, 38), (56, 60, 44), (40, 42, 34)]
MESAS = [  # left, right, top (fractions), cliff and talus widths
    (0.53, 0.80, 0.505, 0.022, 0.075), (0.84, 0.97, 0.545, 0.018, 0.05), (-0.06, 0.17, 0.552, 0.026, 0.08),
    (0.31, 0.37, 0.578, 0.012, 0.045),
]

LOOKS = [  # dusk, night, deep night, before dawn, dawn, sunrise
    dict(t=0.0, sky=[(0.0, (40, 44, 96)), (0.30, (84, 72, 132)), (0.52, (196, 108, 118)), (0.66, (250, 150, 90))],
         west=(150, 70, 30), east=(0, 0, 0), rock=0.55, sun=0.0, stars=0.0, floor=0.55,
         amb=(0.62, 0.52, 0.60), key=(130, 80, 60)),
    dict(t=10.0, sky=[(0.0, (12, 16, 44)), (0.30, (22, 28, 70)), (0.52, (52, 50, 100)), (0.66, (96, 70, 110))],
         west=(40, 16, 10), east=(0, 0, 0), rock=0.30, sun=0.0, stars=0.7, floor=0.32,
         amb=(0.40, 0.44, 0.62), key=(90, 100, 130)),
    dict(t=30.0, sky=[(0.0, (6, 10, 30)), (0.30, (12, 18, 46)), (0.52, (24, 32, 66)), (0.66, (40, 46, 80))],
         west=(0, 0, 0), east=(0, 0, 0), rock=0.26, sun=0.0, stars=1.0, floor=0.28,
         amb=(0.36, 0.40, 0.58), key=(110, 120, 150)),
    dict(t=44.0, sky=[(0.0, (10, 16, 42)), (0.30, (22, 34, 74)), (0.52, (60, 70, 112)), (0.66, (130, 110, 140))],
         west=(0, 0, 0), east=(60, 40, 40), rock=0.32, sun=0.0, stars=0.6, floor=0.32,
         amb=(0.42, 0.44, 0.60), key=(70, 76, 100)),
    dict(t=52.0, sky=[(0.0, (36, 64, 120)), (0.30, (90, 120, 170)), (0.52, (220, 160, 150)), (0.66, (255, 176, 110))],
         west=(0, 0, 0), east=(150, 90, 50), rock=0.55, sun=0.5, stars=0.1, floor=0.50,
         amb=(0.66, 0.62, 0.70), key=(170, 110, 70)),
    dict(t=60.0, sky=[(0.0, (60, 110, 180)), (0.30, (120, 160, 210)), (0.52, (240, 200, 170)), (0.66, (255, 200, 130))],
         west=(0, 0, 0), east=(170, 110, 50), rock=0.75, sun=1.0, stars=0.0, floor=0.70,
         amb=(0.82, 0.78, 0.80), key=(220, 150, 80)),
]


def phases(t):
    return dict(
        lift=ss(t, 2.0, 12.0),
        open=ss(t, 13.0, 29.0),
        wilt=ss(t, 49.0, 59.0),
        moth=(t > 30.0) & (t < 43.5),
        night=ss(t, 4.0, 12.0) * (1 - ss(t, 44.0, 54.0)),
    )


class NightBloom:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.S = min(W, H)
        self.rng = np.random.default_rng(19)
        self.hz = HORIZON * H
        self.moon = Moon(np.random.default_rng(2), 0.026 * self.S, 3.4)
        self.sun = Sun(np.random.default_rng(5), 0.034 * self.S, 5.0)
        self.haze = Haze(W, H, max(2, round(4 * self.S / 1000)))
        self.amb_keys = [(L["t"], L["amb"]) for L in LOOKS]
        self.key_keys = [(L["t"], L["key"]) for L in LOOKS]
        self.build_mesas()
        self.build_stem()
        self.backdrop = Backdrop([L["t"] for L in LOOKS], [self.background(L) for L in LOOKS])
        yy = np.arange(H, dtype=np.float32)[:, None]
        self.sky_mask = ((yy < self.mtop[None, :]) & (self.stem_mask < 0.5)).astype(np.float32)
        self.build_flower()
        self.build_moth()

    # ---- the desert -----------------------------------------------------------------------------------
    def build_mesas(self):
        W, H = self.W, self.H
        u = np.arange(W) / W
        top = np.full(W, self.hz + 2.0)
        which = np.full(W, -1)
        for k, (xl, xr, tp, cw, tw) in enumerate(MESAS):
            d = np.maximum(xl - u, u - xr)                               # distance outside the flat top
            mid = tp + 0.62 * (HORIZON - tp)
            y = np.where(d <= 0, tp + 0.002 * np.sin(u * 300 + k),
                np.where(d < cw, tp + (mid - tp) * np.clip(d / cw, 0, 1) ** 0.8, mid + (HORIZON - mid) * np.clip((d - cw) / tw, 0, 1) ** 0.7))
            y = y + 0.003 * np.sin(u * 140 + 2 * k) * (d > 0)
            better = H * y < top
            top = np.where(better, H * y, top)
            which = np.where(better, k, which)
        self.mtop, self.mwhich = top.astype(np.float32), which

    def build_stem(self):
        """The cactus: a ribbed column and a smaller one leaning beside it."""
        W, H, S, rng = self.W, self.H, self.S, np.random.default_rng(7)
        xs, ys, col, rim = [], [], [], []
        for (x0, y0, x1, y1, wd, n) in ((0.40, 1.03, 0.445, 0.60, 0.066, 24000), (0.29, 1.03, 0.33, 0.76, 0.046, 12000)):
            s = rng.random(n)
            v = sample_v(rng, n, 0.10)
            P = [(x0 * W, y0 * H), (x0 * W + 0.01 * W, (y0 + y1) / 2 * H), (x1 * W - 0.005 * W, y1 * H + 0.05 * H), (x1 * W, y1 * H)]
            bx, by, dx, dy = bezier(P, s)
            nn = np.hypot(dx, dy) + 1e-6
            w = wd * S / 2 * np.where(s > 0.90, np.sqrt(np.clip(1 - ((s - 0.90) / 0.10) ** 2, 0, 1)), 1.0) \
                * (1 - 0.14 * np.exp(-((s - 0.55) / 0.03) ** 2))
            ribs = 7
            rib = np.cos(v * np.pi / 2 * ribs) ** 2                        # ridges and grooves
            x = bx - dy / nn * v * w
            y = by + dx / nn * v * w
            c = shades(rng, STEM, rng.random(n)) * (0.55 + 0.45 * rib)[:, None] * (0.75 + 0.35 * (-v + 1) / 2)[:, None]
            xs.append(x)
            ys.append(y)
            col.append(c)
            rim.append(np.clip(-v, 0, 1) ** 2)
            m = int(n / 170)                                              # areoles with pale spines
            s2 = rng.random(m)
            k = rng.integers(0, ribs, m)
            v2 = (k + 0.5) / ribs * 2 - 1
            bx, by, dx, dy = bezier(P, s2)
            nn = np.hypot(dx, dy) + 1e-6
            ax, ay = bx - dy / nn * v2 * wd * S / 2 * 0.95, by + dx / nn * v2 * wd * S / 2 * 0.95
            for j in range(3):
                a = rng.uniform(0, TAU, m)
                L = rng.uniform(0.002, 0.005, m) * S
                xs.append(ax + np.cos(a) * L)
                ys.append(ay + np.sin(a) * L)
                col.append(np.broadcast_to(np.array([128, 124, 112], np.float32), (m, 3)))
                rim.append(np.ones(m) * 0.5)
        self.stem = dict(x=np.concatenate(xs), y=np.concatenate(ys), col=np.concatenate(col).astype(np.float32),
                         rim=np.concatenate(rim))
        mask = np.zeros((H, W, 3), np.float32)
        stamp(mask, self.stem["x"], self.stem["y"], np.ones(3, np.float32), 1.5)
        self.stem_mask = mask[..., 0]
        self.attach = np.array([0.452 * W, 0.66 * H])                    # where the flower's tube grows from

    def background(self, L):
        W, H, S = self.W, self.H, self.S
        rng = np.random.default_rng(1234)
        img = sky(W, H, rng, L["sky"], self.hz + 3)
        warm_glow(img, 0.92 * W, self.hz, 0.35 * W, L["west"], 0.45)
        warm_glow(img, 0.10 * W, self.hz, 0.40 * W, L["east"], 0.45)
        stars(img, rng, int(1600 * W * H / 1e6), 0.6 * H, L["stars"])
        amb = np.asarray(L["amb"], np.float32)
        key = np.asarray(L["key"], np.float32)
        # mesas: banded red rock, the east-facing sides catching the dawn
        yy = np.arange(H, dtype=np.float32)[:, None]
        rock = (yy >= self.mtop[None, :]) & (yy < self.hz + 3)
        img[rock] = np.asarray(ROCK[0], np.float32) * L["rock"] * 0.8
        n = int(W * H * 0.10)
        x = rng.uniform(0, W, n)
        tp = np.interp(x, np.arange(W), self.mtop)
        y = tp + rng.random(n) * np.maximum(self.hz + 3 - tp, 0)
        k = self.mwhich[np.clip(x.astype(int), 0, W - 1)]
        band = 0.82 + 0.18 * np.sin(y / H * 260 + np.sin(x / W * 9) * 1.5)
        col = shades(rng, ROCK, rng.random(n)) * band[:, None]
        cx = np.array([(m[0] + m[1]) / 2 * W for m in MESAS] + [0])[k]
        hw = np.array([(m[1] - m[0]) / 2 * W for m in MESAS] + [1])[k]
        east = np.clip((cx - x) / hw, -1, 1)
        sheer = np.clip((y - tp) / (0.03 * H), 0, 1)
        litk = np.clip(0.5 + 0.5 * east, 0, 1) * (1 - 0.5 * sheer) + 0.6 * (y - tp < 0.006 * H)
        talus = (y - tp) > 0.05 * H
        col = col * L["rock"] * (0.8 + 0.2 * (1 - talus))[:, None]
        col = col * (0.85 + 0.25 * litk[:, None]) + key * (L["sun"] * 0.55 * litk)[:, None]
        stamp(img, x, y, col, rng.choice([1.0, 1.5, 2.0], n))
        # the desert floor, with brush and a pair of saguaros
        floor = yy >= self.hz
        d = np.clip((yy - self.hz) / (H - self.hz), 0, 1)
        fcol = np.asarray(SAND[0], np.float32) * L["floor"] * (1.0 - 0.35 * d[..., None])
        img[:] = np.where(floor[..., None], fcol, img)
        n = int(W * (H - self.hz) * 0.45)
        x, y = rng.uniform(0, W, n), self.hz + rng.random(n) * (H - self.hz)
        dd = (y - self.hz) / (H - self.hz)
        stamp(img, x, y, shades(rng, SAND, rng.random(n)) * (L["floor"] * (1.05 - 0.4 * dd))[:, None], 1.0 + dd)
        m = 150
        bx, by = rng.uniform(-0.02, 1.02, m) * W, self.hz + (rng.random(m) ** 1.6) * (H - self.hz)
        for j in range(m):                                                 # creosote and brush
            dd = (by[j] - self.hz) / (H - self.hz)
            r = (0.004 + 0.02 * dd) * S * rng.uniform(0.6, 1.3)
            q = int(20 + 9000 * (r / S) ** 1.3)
            a, rr = rng.uniform(np.pi, TAU, q), r * np.sqrt(rng.random(q))
            px, py = bx[j] + np.cos(a) * rr * 1.3, by[j] + np.sin(a) * rr * 0.9
            c = shades(rng, BRUSH, rng.random(q)) * amb * (1.3 - 0.4 * dd)
            c = c + key * (0.25 * np.clip(-(px - bx[j]) / r * (L["sun"] > 0) + 0.3, 0, 1))[:, None] * 0.4
            stamp(img, px, py, c, 1.0 + dd)
        for sx, sb, sh, arms in ((0.14, 0.725, 0.20, ((0.45, -1, 0.35), (0.62, 1, 0.30))), (0.86, 0.705, 0.13, ((0.55, 1, 0.4),))):
            self.saguaro(img, rng, sx * W, sb * H, sh * H, arms, amb, key * L["sun"])
        # the cactus in front, lit by the sky
        St = self.stem
        light = amb * 1.05 + 0.05
        c = St["col"] * light + key * (0.35 * St["rim"])[:, None] * (0.6 + 0.4 * L["sun"])
        stamp(img, St["x"], St["y"], c, 1.6 * S / 1000 + 0.3)
        return img

    def saguaro(self, img, rng, x0, y0, h, arms, amb, key):
        S = self.S
        w = 0.085 * h
        n = int(h * w * 2.2)
        s, v = rng.random(n), rng.uniform(-1, 1, n)
        top = np.where(s > 0.95, np.sqrt(np.clip(1 - ((s - 0.95) / 0.05) ** 2, 0, 1)), 1)
        x, y = x0 + v * w * top, y0 - s * h
        rib = 0.7 + 0.3 * np.cos(v * np.pi * 3) ** 2
        c = np.asarray(STEM[0], np.float32)[None] * rib[:, None] * amb * 0.9 + key[None] * (0.3 * np.clip(-v, 0, 1))[:, None]
        stamp(img, x, y, c, 1.5)
        for f, side, ln in arms:
            m = int(h * ln * w * 2.5)
            q, v = rng.random(m), rng.uniform(-1, 1, m)
            P = [(x0 + side * w * 0.8, y0 - f * h), (x0 + side * (w + 0.12 * h), y0 - f * h + 0.02 * h),
                 (x0 + side * (w + 0.13 * h), y0 - f * h - 0.05 * h), (x0 + side * (w + 0.12 * h), y0 - f * h - ln * h)]
            bx, by, dx, dy = bezier(P, q)
            nn = np.hypot(dx, dy) + 1e-6
            aw = w * 0.75
            x, y = bx - dy / nn * v * aw, by + dx / nn * v * aw
            c = np.asarray(STEM[0], np.float32)[None] * (0.7 + 0.3 * np.cos(v * np.pi * 2) ** 2)[:, None] * amb * 0.9 \
                + key[None] * (0.3 * np.clip(-v * side, 0, 1))[:, None]
            stamp(img, x, y, c, 1.5)

    # ---- the flower ------------------------------------------------------------------------------------
    def build_flower(self):
        rng, S = self.rng, self.S
        G = []
        for layer, (count, length, width, pal, r0) in enumerate(((21, 0.19, 0.016, OUTER, 0.022), (23, 0.145, 0.036, PETAL, 0.014))):
            nu, nv = (36, 15) if layer else (46, 7)
            m = nu * nv
            for k in range(count):
                u = np.clip((np.repeat(np.arange(nu), nv) + rng.uniform(0.15, 0.85, m)) / nu, 0, 1)
                v = np.clip((np.tile(np.arange(nv), nu) + rng.uniform(0.15, 0.85, m)) / nv * 2 - 1, -1, 1)
                G.append(dict(layer=np.full(m, layer), phi=np.full(m, TAU * k / count + rng.normal(0, 0.06) + 0.13 * layer),
                              u=u, v=v, L=np.full(m, length * S * rng.uniform(0.9, 1.08)),
                              w=np.full(m, width * S * rng.uniform(0.85, 1.15)), r0=np.full(m, r0 * S),
                              col=shades(rng, pal, rng.random(m) * 0.6 + 0.4 * u, 0.6) * rng.uniform(0.94, 1.04),
                              jit=np.full(m, rng.normal(0, 1))))
        self.petals = {k: np.concatenate([g[k] for g in G]) for k in G[0]}
        n = 260                                                            # stamens: filaments and anthers
        self.stamens = dict(phi=rng.uniform(0, TAU, n), r=rng.uniform(0.006, 0.02, n) * S, L=rng.uniform(0.045, 0.065, n) * S,
                            bend=rng.uniform(0.25, 0.6, n), col=shades(rng, STAMEN, rng.random(n)))
        n = 900                                                            # tube: long, scaly, a little hairy
        self.tube = dict(s=rng.random(n), v=sample_v(rng, n, 0.1), col=shades(rng, OUTER[3:] + STEM[1:3], rng.random(n)))
        self.scent = dict(ph=rng.uniform(0, TAU, 60), r=rng.random(60), t0=rng.uniform(0, 8, 60))

    def flower_pose(self, t, ph):
        """Centre of the flower, its axis, and the bend of the tube."""
        W, H, S = self.W, self.H, self.S
        o, wilt, lift = ph["open"], ph["wilt"], ph["lift"]
        bud_axis = unit((0.42, -0.90, 0.10))
        open_axis = unit((0.10, -0.42, 0.90))
        droop = unit((0.25, 0.75, 0.62))
        a = unit(bud_axis * (1 - o) + open_axis * o)
        a = unit(a * (1 - wilt) + droop * wilt)
        base = np.array([0.505 * W, (0.50 - 0.04 * lift) * H]) + np.array([0.01 * W, 0.07 * H]) * wilt
        sway = 0.004 * S * math.sin(0.7 * t)
        return base + np.array([sway, 0.0]), a

    def frame_of(self, a):
        e1 = unit(np.cross(a, [0, 0, 1.0]) if abs(a[2]) < 0.95 else np.cross(a, [1.0, 0, 0]))
        e2 = np.cross(a, e1)
        return e1, e2

    def draw_flower(self, dots, glows, t, ph, amb, key, ldir):
        S = self.S
        base, a = self.flower_pose(t, ph)
        e1, e2 = self.frame_of(a)
        o, wilt = float(ph["open"]), float(ph["wilt"])
        size = 0.62 + 0.38 * float(ss(t, 1.0, 20.0))
        P = self.petals
        outer = P["layer"] == 0
        # the opening angle along each petal: shut in a bud, then reflexed (outer) or cupped (inner)
        b0 = np.where(outer, 0.10 + 1.55 * o ** 1.2, 0.04 + 0.95 * o ** 1.5)
        curl = np.where(outer, -0.30 * (1 - o) + 0.55 * o, -0.25 * (1 - o) + 0.30 * o)
        b0 = b0 * (1 - 0.55 * wilt)
        curl = curl - 0.5 * wilt
        Lp = P["L"] * size * np.where(outer, 1.0, 0.82 + 0.18 * o) * (1 - 0.15 * wilt)
        wid = P["w"] * np.where(outer, 1.0, 0.45 + 0.55 * o) * (1 - 0.3 * wilt)
        u, v = P["u"], P["v"]
        th = b0 + curl * u + 0.06 * P["jit"]
        rad = P["r0"] * (0.5 + 0.5 * o) + Lp * u * np.sin(th) * np.where(outer, 1.0, 1.0)
        ax = Lp * u * np.cos(th)
        spat = (0.22 + 0.78 * np.sin(np.pi / 2 * np.clip(u / 0.72, 0, 1))) * np.sqrt(np.clip(1 - ((u - 0.72) / 0.28) ** 2, 0, 1) * (u > 0.72) + (u <= 0.72))
        spat = spat * (1 - 0.35 * np.clip((u - 0.92) / 0.08, 0, 1) * (1 - np.abs(v)))   # a little point at the tip
        shape = np.where(outer, np.sin(np.pi * np.clip(0.1 + 0.9 * u, 0, 1)) ** 0.9 * (1 - 0.6 * u ** 3), spat)
        lat = v * wid * shape
        cup = np.where(outer, 0.0, 0.18) * wid * v * v * o
        cp, sp = np.cos(P["phi"]), np.sin(P["phi"])
        d = cp[:, None] * e1[None] + sp[:, None] * e2[None]
        tg = -sp[:, None] * e1[None] + cp[:, None] * e2[None]
        pos = a[None] * (ax + cup)[:, None] + d * rad[:, None] + tg * lat[:, None]
        pos[:, 1] += (0.30 * wilt * Lp * u ** 2)                         # limp petals hang
        X, Y, Z = base[0] + pos[:, 0], base[1] + pos[:, 1], pos[:, 2]
        PX, PY = X, Y
        face = np.clip(0.55 + 0.45 * pos[:, 2] / (np.abs(pos[:, 2]).max() + 1e-6), 0, 1)        # turned toward us
        col = P["col"] * (1 - 0.45 * (1 - u) ** 2 * (~outer) * o)[:, None]  # shaded deep in the cup
        col = col * (1 - 0.22 * np.abs(v) ** 6)[:, None]                      # each petal's edge, so they read apart
        col = mix(col, np.array([200, 180, 120], np.float32), 0.45 * wilt)
        col = col * amb * 1.1 + key * (0.35 + 0.45 * face)[:, None] * 0.6
        dots.append((X, Y, col, np.maximum(1.0, np.where(outer, 2.4, 3.0) * S / 1000), Z))
        # stamens and the stigma rise out of the cup
        St = self.stamens
        show = float(ss(o, 0.35, 0.8)) * (1 - 0.6 * wilt)
        if show > 0.01:
            q = np.linspace(0, 1, 10)[None, :]
            L = St["L"][:, None] * show * 1.25
            ang = St["bend"][:, None] * q
            r = St["r"][:, None] + L * q * np.sin(ang + 0.15) * 0.6
            h = L * q * np.cos(ang)
            cp, sp = np.cos(St["phi"])[:, None, None], np.sin(St["phi"])[:, None, None]
            dd = cp * e1[None, None] + sp * e2[None, None]
            pos = a[None, None] * h[..., None] + dd * r[..., None]
            pos[..., 1] += 0.2 * wilt * L * q ** 2
            X, Y, Z = base[0] + pos[..., 0], base[1] + pos[..., 1], pos[..., 2] + 0.5
            col = np.repeat(St["col"][:, None, :], 10, 1)
            col[:, -2:] = np.array(ANTHER[0], np.float32)
            col = col.reshape(-1, 3) * amb * 1.1 + key * 0.45
            dots.append((X.ravel(), Y.ravel(), col, np.where(q.ravel() > 0.8, 1.6, 0.9)[None].repeat(len(St["phi"]), 0).ravel()
                         * S / 1000 + 0.4, Z.ravel()))
            k = 12                                                         # the stigma: a pale star on a long style
            sh = 0.10 * S * show
            q = np.linspace(0, 1, 30)
            sp_ = base[None] + (a[None, :2] * (sh * q)[:, None])
            dots.append((sp_[:, 0], sp_[:, 1] + 0.2 * wilt * sh * q ** 2, np.array([236, 236, 224]) * amb * 1.1 + key * 0.4,
                         1.6 * S / 1000 + 0.4, a[2] * sh * q + 1.0))
            tip = base + a[:2] * sh
            ang = np.linspace(0, TAU, k, endpoint=False)
            lobes = np.linspace(0, 1, 6)[None, :] * 0.012 * S * show
            lx = tip[0] + (np.cos(ang)[:, None] * e1[0] + np.sin(ang)[:, None] * e2[0]) * lobes
            ly = tip[1] + (np.cos(ang)[:, None] * e1[1] + np.sin(ang)[:, None] * e2[1]) * lobes
            dots.append((lx.ravel(), ly.ravel(), np.array([244, 244, 230]) * amb * 1.1 + key * 0.5, 1.4 * S / 1000 + 0.4,
                         a[2] * sh + 1.2))
        # the tube, from the stem to the back of the flower
        T = self.tube
        A0 = self.attach
        back = base - a[:2] * 0.012 * S
        P4 = [A0, A0 + np.array([0.04 * self.W, -0.01 * self.H]), back - a[:2] * 0.10 * S - np.array([0.0, -0.02 * self.H]) * 0, back]
        bx, by, dx, dy = bezier(P4, T["s"])
        nn = np.hypot(dx, dy) + 1e-6
        w = (0.008 + 0.006 * T["s"] ** 3) * S
        X, Y = bx - dy / nn * T["v"] * w, by + dx / nn * T["v"] * w
        col = T["col"] * amb * (0.8 + 0.3 * np.clip(-T["v"], 0, 1))[:, None] + key * 0.15
        dots.append((X, Y, col, 1.5 * S / 1000 + 0.4, np.full(len(X), -30.0)))
        # glowing a little under the moon
        if o > 0.2 and wilt < 0.9:
            g = float(ph["night"]) * o * (1 - wilt)
            tips = np.nonzero((~outer) & (u > 0.8))[0][::25]                  # a soft light round the petals
            self.haze.add(PX[tips], PY[tips], np.array([60, 64, 72]) * 0.12 * g, 30 * S / 1000)
        return base, a

    # ---- the hawk moth -----------------------------------------------------------------------------------
    def build_moth(self):
        rng = np.random.default_rng(41)
        n = 420
        s, v = rng.random(n), sample_v(rng, n, 0.15)               # body from head (0) to tail (1)
        w = 0.010 * np.sin(np.pi * np.clip(0.06 + 0.9 * s, 0, 1)) ** 0.7 * (1 + 0.35 * np.exp(-((s - 0.28) / 0.12) ** 2))
        col = shades(rng, [(110, 98, 88), (126, 114, 100), (92, 82, 74)], rng.random(n))
        band = (s > 0.42) & (np.sin(s * 38) > 0.4) & (np.abs(v) > 0.45)
        col[band] = shades(rng, [(226, 160, 70), (236, 176, 90)], rng.random(int(band.sum())))   # its orange spots
        self.moth_body = dict(s=s, v=v, w=w, col=col)
        self.wing = rng.random((2, 700))

    def moth_state(self, t, flower):
        """Moth position, heading and how far its tongue reaches."""
        W, H, S = self.W, self.H, self.S
        base, a = flower
        mouth = base + a[:2] * 0.03 * S
        hover = mouth + np.array([-0.12 * W, -0.02 * H])
        keys = [(30.0, (-0.10 * W, 0.20 * H)), (32.5, (hover[0] - 0.10 * W, hover[1] - 0.08 * H)),
                (34.0, tuple(hover)), (39.5, tuple(hover + np.array([0.02 * W, 0.01 * H]))),
                (41.0, (hover[0] + 0.05 * W, hover[1] - 0.12 * H)), (43.5, (1.12 * W, 0.08 * H))]
        p = np.array(keyed(t, keys), np.float64)
        p = p + 0.006 * S * np.array([math.sin(7.3 * t), math.sin(9.1 * t + 1)])
        drink = float(ss(t, 34.3, 35.0) * (1 - ss(t, 39.3, 40.0)))
        head = unit(mouth - p) if drink > 0 or t < 34.0 else unit(np.array(keyed(t + 0.2, keys)) - np.array(keyed(t, keys)) + 1e-6)
        return p, head, drink, mouth

    def draw_moth(self, img, dots, t, ph, flower, amb, key):
        if not ph["moth"]:
            return
        S = self.S
        p, hd, drink, mouth = self.moth_state(t, flower)
        side = np.array([-hd[1], hd[0]])
        B = self.moth_body
        L = 0.075 * S
        X = p[0] - hd[0] * B["s"] * L + side[0] * B["v"] * B["w"] * S * 1.4
        Y = p[1] - hd[1] * B["s"] * L + side[1] * B["v"] * B["w"] * S * 1.4
        col = B["col"] * (amb * 1.1 + 0.1) + key * 0.25
        dots.append((X, Y, col, 1.4 * S / 1000 + 0.4, np.full(len(X), 1e4)))
        if drink > 0:                                                 # the long tongue
            q = np.linspace(0, 1, 60) * drink
            tx = p[0] + (mouth[0] - p[0]) * q + 0.004 * S * np.sin(q * 6) * (1 - q)
            ty = p[1] + (mouth[1] - p[1]) * q
            dots.append((tx, ty, np.array([90, 80, 70]) * (amb * 1.2 + 0.2), 0.8 * S / 1000 + 0.3, np.full(60, 1e4 - 1)))
            r = np.random.default_rng(int(t * 15))                    # pollen in the air
            n = 14
            glow(img, mouth[0] + r.normal(0, 0.03, n) * S, mouth[1] + r.normal(0, 0.02, n) * S - 0.01 * S,
                 np.array([240, 210, 120]) * 0.6 * r.random((n, 1)), 1.2)
        # wings: a blur of beats, drawn as faint light over the sweep
        wq = self.wing
        base_ang = math.atan2(hd[1], hd[0])
        for sgn in (-1, 1):
            ang = base_ang + sgn * (1.9 + 0.5 * (wq[0] * 1.8 - 0.9))
            r = 0.085 * S * wq[1] ** 0.6
            wx = p[0] - hd[0] * 0.018 * S + np.cos(ang) * r
            wy = p[1] - hd[1] * 0.018 * S + np.sin(ang) * r
            edge = 0.6 + 0.4 * np.abs(wq[0] * 2 - 1) ** 2                 # denser at the ends of the beat
            glow(img, wx, wy, np.array([130, 120, 108])[None, :] * (0.22 * edge)[:, None] * (0.9 + key.mean() / 400), 1.6)
            self.haze.add(wx[::12], wy[::12], np.array([40, 36, 32]) * 0.25, 18 * S / 1000)

    # ---- the frame ---------------------------------------------------------------------------------
    def moon_pos(self, t):
        f = float(np.clip((t - 4.0) / 46.0, 0, 1))
        return self.W * (0.64 + 0.32 * f), self.H * (0.28 - 0.10 * math.sin(math.pi * min(f, 0.5)) + 1.45 * max(0.0, f - 0.5) ** 1.3)

    def render(self, t):
        ph = phases(t)
        img = self.backdrop.at(t)
        amb = keyed(t, self.amb_keys)
        key = keyed(t, self.key_keys)
        mx, my = self.moon_pos(t)
        mb = float(ss(t, 5.0, 10.0))
        if mb > 0 and my < self.hz + 0.03 * self.H:
            self.moon.draw(img, mx, my, np.array([250, 246, 228]) * mb + img[int(min(my, self.H - 1)), int(min(mx, self.W - 1))] * (1 - mb),
                           np.array([30, 32, 40]) * mb, self.sky_mask, self.sky_mask)
        sb = float(ss(t, 54.0, 60.0))
        if sb > 0:
            self.sun.draw(img, 0.12 * self.W, self.hz + 0.03 * self.H - 0.06 * self.H * sb, np.array([255, 236, 190]),
                          np.array([120, 70, 25]) * sb, self.sky_mask, self.sky_mask)
        ldir = (0.6, -0.8) if t < 46 else (-0.9, -0.3)
        dots = []
        flower = self.draw_flower(dots, None, t, ph, amb, key, ldir)
        self.draw_moth(img, dots, t, ph, flower, amb, key)
        paint(img, dots)
        self.haze.apply(img, 1.0)
        return np.clip(img, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    run(NightBloom, "Queen of the Night", "night_bloom", T_END, STAGES)
