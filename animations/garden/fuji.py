"""
fuji.py - a day and a night of Mount Fuji, as a living woodblock print. A seamless one-minute loop.

Flat bands of colour, ink key-lines and washi paper grain, with the light painted in dots.
We look west across a lake to the mountain, past a cherry tree and a vermilion torii standing
in the water, so the minute can pass through Fuji's famous faces in their true order:

  night        climbers' headlamps zigzag up the trail in a chain of sparks; paper lanterns
               drift on the lake under the Milky Way
  first light  Akafuji - the dawn turns the mountain crimson, running down from the summit
  day          a lens-shaped cap cloud (kasa-gumo) settles on the peak, a hanging cloud drifts
               beside it, cranes fly across, and when the wind drops the lake holds a perfect
               upside-down Fuji
  sunset       Diamond Fuji - the sun sinks exactly into the crater and flares like a jewel
  dusk         fireworks bloom over the lake and the water doubles every one

    python fuji.py            the print, living (Space pause, R restart, S save frame, Esc quit)
    python fuji.py --png      save five moments
    python fuji.py --gif      save one seamless loop as a GIF
"""
import math

import numpy as np
from PIL import Image

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))   # the shared engine: engine/pointillism.py

from pointillism import Haze, Sun, bezier, glow, keyed, run, ss, stamp

T_END = 60.0
TAU = 2 * math.pi
STAGES = [3.0, 17.0, 31.0, 44.6, 53.0]
T_DIAMOND = 44.6

N0 = dict(top=(6, 8, 26), mid=(16, 22, 56), low=(40, 46, 88), ink=(4, 6, 16), rock=((32, 36, 62), (16, 20, 38)),
          snow=((134, 142, 184), (72, 82, 124)), hill1=(14, 22, 32), hill2=(9, 14, 22), water=(8, 12, 28), stars=1.0,
          kasumi=(52, 58, 100), ka=0.40, cl=(120, 126, 166), cs=(60, 64, 100), amb=(0.30, 0.34, 0.52), key=(30, 40, 80))
LOOKS = [
    dict(N0, t=0.0),
    dict(t=10.5, top=(18, 24, 64), mid=(72, 72, 128), low=(216, 148, 146), ink=(10, 12, 30), rock=((62, 58, 94), (36, 36, 62)),
         snow=((194, 188, 216), (124, 122, 162)), hill1=(26, 32, 50), hill2=(18, 22, 38), water=(36, 36, 66), stars=0.35,
         kasumi=(206, 156, 176), ka=0.50, cl=(230, 190, 200), cs=(130, 110, 150), amb=(0.50, 0.48, 0.62), key=(120, 80, 100)),
    dict(t=17.0, top=(46, 82, 152), mid=(164, 164, 204), low=(255, 208, 172), ink=(64, 16, 18), rock=((214, 62, 42), (150, 40, 44)),
         snow=((255, 190, 170), (244, 146, 138)), hill1=(46, 58, 64), hill2=(32, 40, 50), water=(96, 100, 146), stars=0.0,
         kasumi=(255, 222, 216), ka=0.60, cl=(255, 206, 188), cs=(206, 146, 156), amb=(0.86, 0.72, 0.68), key=(255, 150, 110)),
    dict(t=24.0, top=(28, 74, 160), mid=(112, 162, 216), low=(216, 232, 244), ink=(28, 38, 70), rock=((98, 108, 148), (62, 72, 110)),
         snow=((252, 252, 255), (222, 230, 246)), hill1=(40, 86, 76), hill2=(28, 62, 58), water=(64, 104, 148), stars=0.0,
         kasumi=(255, 255, 255), ka=0.88, cl=(255, 255, 255), cs=(188, 200, 224), amb=(0.95, 0.95, 1.0), key=(255, 240, 220)),
    dict(t=33.0, top=(18, 56, 146), mid=(80, 140, 210), low=(198, 222, 240), ink=(24, 34, 66), rock=((86, 100, 142), (54, 64, 102)),
         snow=((252, 252, 255), (216, 226, 246)), hill1=(38, 88, 74), hill2=(26, 64, 58), water=(56, 100, 150), stars=0.0,
         kasumi=(255, 255, 255), ka=0.88, cl=(255, 255, 255), cs=(186, 198, 224), amb=(1.0, 1.0, 1.0), key=(255, 250, 235)),
    dict(t=40.0, top=(48, 64, 132), mid=(198, 160, 172), low=(255, 206, 142), ink=(255, 172, 92), rock=((74, 60, 86), (50, 44, 66)),
         snow=((214, 190, 198), (150, 130, 152)), hill1=(42, 48, 54), hill2=(30, 34, 42), water=(116, 98, 112), stars=0.0,
         kasumi=(255, 214, 182), ka=0.55, cl=(255, 216, 186), cs=(172, 142, 154), amb=(0.82, 0.72, 0.68), key=(255, 190, 120)),
    dict(t=T_DIAMOND, top=(40, 40, 98), mid=(208, 122, 128), low=(255, 174, 94), ink=(255, 196, 110), rock=((44, 34, 58), (32, 28, 46)),
         snow=((120, 96, 118), (80, 66, 92)), hill1=(30, 30, 44), hill2=(22, 22, 34), water=(110, 74, 80), stars=0.0,
         kasumi=(255, 172, 134), ka=0.50, cl=(255, 180, 124), cs=(124, 82, 102), amb=(0.66, 0.56, 0.60), key=(255, 150, 80)),
    dict(t=50.0, top=(20, 20, 60), mid=(110, 66, 112), low=(224, 114, 102), ink=(10, 8, 20), rock=((34, 30, 50), (24, 22, 38)),
         snow=((88, 78, 110), (60, 54, 86)), hill1=(18, 18, 30), hill2=(12, 12, 22), water=(36, 28, 48), stars=0.3,
         kasumi=(124, 84, 124), ka=0.45, cl=(150, 100, 130), cs=(70, 50, 80), amb=(0.46, 0.40, 0.54), key=(170, 90, 90)),
    dict(N0, t=56.0, same=0.0),
    dict(N0, t=60.0, same=0.0),
]
BANDS = [  # mist bands of a print: centre height, thickness, left end, length (fractions of the frame), phase
    (0.505, 0.034, -0.06, 0.44, 0.0), (0.556, 0.030, 0.56, 0.52, 1.7), (0.100, 0.022, 0.60, 0.28, 2.2),
]
SHELLS = [  # launch time, x, burst height (fractions), first and last colour, size
    (49.4, 0.20, 0.30, (255, 190, 80), (255, 240, 180), 1.0), (50.1, 0.80, 0.24, (255, 90, 150), (255, 200, 230), 1.0),
    (50.9, 0.32, 0.36, (90, 255, 150), (220, 255, 220), 0.85), (51.5, 0.69, 0.33, (255, 190, 80), (255, 120, 60), 1.1),
    (52.3, 0.13, 0.20, (255, 80, 70), (255, 230, 200), 0.9), (52.9, 0.87, 0.30, (90, 200, 255), (230, 240, 255), 0.95),
    (53.6, 0.27, 0.26, (255, 200, 90), (255, 160, 60), 1.2), (54.2, 0.74, 0.20, (255, 110, 190), (255, 220, 240), 1.0),
    (55.2, 0.50, 0.075, (255, 215, 120), (255, 255, 230), 1.6),
]


def night(t):
    return float(1 - ss(t, 8, 13) + ss(t, 50, 55))


class Fuji:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.S = S = min(W, H)
        self.yw = int(round(0.62 * H))
        self.cx, self.yp, self.R, self.u0 = 0.5 * W, 0.175 * H, 0.62 * W, 0.058
        self.haze = Haze(W, H, max(2, round(4 * S / 1000)))
        self.sun = Sun(np.random.default_rng(5), 0.034 * S, 5.0)
        self.xidx = np.arange(W)
        self.keys = {k: [(L["t"], L[k]) for L in LOOKS] for k in ("amb", "key", "water", "kasumi", "ka", "cl", "cs", "low")}
        self.build_land()
        self.build_looks()
        self.build_fg()
        self.build_paper()
        self.build_motion()

    # ---- the land: the mountain, its snow, the foothills --------------------------------------------
    def build_land(self):
        W, H = self.W, self.H
        xs = np.arange(W, dtype=np.float64)
        u = np.abs(xs - self.cx) / self.R
        h = np.minimum(np.clip((1 - u) / (1 - self.u0), 0, None) ** 1.85, 1.0)
        h = h - (u < self.u0 * 1.1) * 0.010 * (0.5 + 0.5 * np.sin(xs / W * TAU * 38))     # the notched crater rim
        h = h + 0.030 * np.exp(-((xs / W - 0.70) / 0.035) ** 2)                            # the Hoei shoulder
        self.mtop = (self.yw - h * (self.yw - self.yp)).astype(np.float32)
        rows = np.arange(H, dtype=np.float32)[:, None]
        self.mtn = (rows >= self.mtop[None, :]) & (rows < self.yw)
        self.grel = (np.clip((self.yw - rows) / (self.yw - self.yp), 0, 1) * np.ones((1, W), np.float32)).astype(np.float32)
        X = xs[None, :] - self.cx
        Y = rows - (self.yp - 0.06 * H)
        theta = np.arctan2(X, Y)
        fing = 0.5 + 0.5 * np.sin(theta * 70 + 2.5 * np.sin(theta * 17) + 1.5 * np.sin(theta * 5))
        line = 0.60 + 0.025 * np.sin(xs / W * TAU * 9)[None, :]
        k = np.clip((self.grel - 0.30) / (line - 0.30), 0, 1)
        self.snow = self.mtn & ((self.grel > line) | (fing > 1.0 - 0.95 * k ** 1.3))
        self.hatch = (0.88 + 0.12 * (np.sin(theta * 160 + np.sin(theta * 31) * 2) > 0.55)).astype(np.float32)
        f = xs / W
        self.h1 = (H * (0.548 + 0.012 * np.sin(f * TAU * 1.3 + 0.5) + 0.006 * np.sin(f * TAU * 3.7))
                   - 0.006 * H * np.abs(np.sin(f * TAU * 31))).astype(np.float32)
        self.h2 = (H * (0.585 + 0.009 * np.sin(f * TAU * 2.1 + 2.0)) - 0.007 * H * np.abs(np.sin(f * TAU * 23 + 1))).astype(np.float32)
        self.hill1 = (rows >= self.h1[None, :]) & (rows < self.yw)
        self.hill2 = (rows >= self.h2[None, :]) & (rows < self.yw)
        self.sky = (rows < self.mtop[None, :]) & (rows < self.h1[None, :])
        self.skym = self.sky.astype(np.float32)
        self.gmask = np.clip(self.skym + 0.45 * self.mtn, 0, 1).astype(np.float32)
        up = np.zeros_like(self.mtn)
        up[1:] = self.mtn[:-1]
        edge = self.mtn & ~up
        edge |= self.mtn & ~(np.roll(self.mtn, 1, 1) & np.roll(self.mtn, -1, 1))
        e2 = edge.copy()
        e2[1:] |= edge[:-1]
        self.m_edge = e2 & ~self.hill1
        sn = self.snow
        se = sn & ~(np.roll(sn, 1, 0) & np.roll(sn, -1, 0) & np.roll(sn, 1, 1) & np.roll(sn, -1, 1))
        self.s_edge = se & ~self.m_edge & ~self.hill1
        t1 = np.zeros_like(self.hill1)
        t1[1:] = self.hill1[1:] & ~self.hill1[:-1]
        self.h_edge = t1 | np.roll(t1, 1, 0)

    # ---- the looks: the same print, inked for different hours ---------------------------------------------
    def build_looks(self):
        W, H = self.W, self.H
        rng = np.random.default_rng(11)
        ri, ci = np.arange(H)[:, None] / H, np.arange(W)[None, :] / W
        self.wob = (0.010 * np.sin(ci * TAU * 2.3 + ri * 9) + 0.006 * np.sin(ci * TAU * 7.1 - ri * 21)).astype(np.float32)
        n = int(2600 * W * H / 1e6)
        sx, sy = rng.uniform(0, W - 1, n), rng.random(n) ** 1.3 * (self.yw - 1)
        b = rng.random(n) ** 3
        ok = self.sky[sy.astype(int), sx.astype(int)]
        self.star = (sx[ok], sy[ok], b[ok])
        m = int(16000 * W * H / 1e6)
        a = rng.uniform(-0.2, 1.2, m)
        mx, my = a * W, H * (0.44 - 0.40 * a) + rng.normal(0, 0.05, m) * H
        ok = (my > 0) & (my < self.yw - 1) & (mx > 0) & (mx < W - 1)
        ok[ok] = self.sky[my[ok].astype(int), mx[ok].astype(int)]
        self.mw = (mx[ok], my[ok], rng.random(int(ok.sum())))
        cache, imgs = {}, []
        for L in LOOKS:
            if "same" in L:
                imgs.append(cache[L["same"]])
                continue
            cache[L["t"]] = self.look(L)
            imgs.append(cache[L["t"]])
        self.look_t = [L["t"] for L in LOOKS]
        self.looks = imgs

    def look(self, L):
        W, H, yw = self.W, self.H, self.yw
        rng = np.random.default_rng(1234)
        rows = np.arange(H, dtype=np.float32)[:, None]
        g = np.clip(rows / yw + self.wob, 0, 1)
        q = np.floor(g * 15) / 15                                            # flat bands of colour
        top, mid, low = (np.asarray(L[k], np.float32) for k in ("top", "mid", "low"))
        col = np.where((q < 0.55)[..., None], top + (mid - top) * (q / 0.55)[..., None],
                       mid + (low - mid) * ((q - 0.55) / 0.45)[..., None])
        band = np.clip(1 - rows / yw / 0.16, 0, 1) ** 1.4                    # the dark band across the top
        img = (col * (1 - 0.55 * band[..., None]) + top * 0.2 * band[..., None]).astype(np.float32)
        if L["stars"] > 0:
            x, y, b = self.star
            glow(img, x, y, np.array([220, 225, 255])[None] * ((0.15 + 0.85 * b) * L["stars"])[:, None], 1.0 + 0.5 * (b > 0.5))
            x, y, r = self.mw
            glow(img, x, y, np.array([200, 200, 240])[None] * ((0.05 + 0.10 * r) * L["stars"])[:, None], 1.0)
        gm = self.grel[..., None]
        rt, rb = (np.asarray(c, np.float32) for c in L["rock"])
        st, sb = (np.asarray(c, np.float32) for c in L["snow"])
        rock = (rb + (rt - rb) * gm) * self.hatch[..., None]
        snow = sb + (st - sb) * gm
        img = np.where(self.mtn[..., None], np.where(self.snow[..., None], snow, rock), img).astype(np.float32)
        ink = np.asarray(L["ink"], np.float32)
        img[self.s_edge] = rock[self.s_edge] * 0.72
        img[self.m_edge] = ink
        for mask, top_y, c in ((self.hill1, self.h1, L["hill1"]), (self.hill2, self.h2, L["hill2"])):
            d = np.clip((rows - top_y[None, :]) / (yw - top_y[None, :] + 1), 0, 1)
            hc = np.asarray(c, np.float32) * (0.85 + 0.45 * d[..., None])
            img = np.where(mask[..., None], hc, img).astype(np.float32)
        n = int(W * 0.07 * H * 0.9)                                          # trees on the hills, as dots
        x = rng.uniform(0, W, n)
        which = rng.random(n) < 0.5
        y = np.where(which, np.interp(x, np.arange(W), self.h1), np.interp(x, np.arange(W), self.h2)) + rng.random(n) ** 2 * 0.03 * H
        c = np.where(which[:, None], np.asarray(L["hill1"], np.float32), np.asarray(L["hill2"], np.float32))
        stamp(img, x, np.minimum(y, yw - 1), c * (0.75 + 0.5 * rng.random((n, 1))), rng.choice([1.0, 1.5, 2.0], n))
        img[self.h_edge] = ink * 0.8 + np.asarray(L["hill1"], np.float32) * 0.2
        img[yw:] = np.asarray(L["water"], np.float32)
        n = int(W * yw * 0.3)                                                 # the grain of the baren
        x, y = rng.uniform(0, W, n), rng.uniform(0, yw, n)
        stamp(img, x, y, img[y.astype(int), x.astype(int)] * (0.94 + 0.12 * rng.random((n, 1))), rng.choice([1.0, 1.5], n))
        return img

    def look_at(self, t):
        ts = self.look_t
        i = int(np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 2))
        a, b = self.looks[i], self.looks[i + 1]
        if a is b:
            return a.copy()
        w = float(ss(t, ts[i], ts[i + 1]))
        if i == 1:                                                           # first light runs down from the summit
            rows = np.arange(self.H, dtype=np.float32)[:, None, None] / self.yw
            wr = np.clip(w * 1.8 - 0.8 * np.clip(rows, 0, 1), 0, 1)
            return a + (b - a) * wr
        return a + (b - a) * np.float32(w)

    # ---- the near things: shore, cherry tree, torii -------------------------------------------------------
    def build_fg(self):
        W, H, S = self.W, self.H, self.S
        rng = np.random.default_rng(21)
        alb = np.zeros((H, W, 3), np.float32)
        msk = np.zeros((H, W, 3), np.float32)
        rimm = np.zeros((H, W, 3), np.float32)
        kind = np.zeros((H, W, 3), np.float32)
        tor = []

        def put(x, y, col, rad, rim, k):
            x = np.atleast_1d(np.asarray(x, np.float32))
            n = len(x)
            stamp(alb, x, y, np.broadcast_to(np.asarray(col, np.float32), (n, 3)), rad)
            stamp(msk, x, y, np.ones(3, np.float32), rad)
            stamp(rimm, x, y, np.broadcast_to(np.asarray(rim, np.float32).reshape(-1, 1), (n, 3)), rad)
            stamp(kind, x, y, np.full(3, k, np.float32), rad)
            if k == 3:
                tor.append((x, np.broadcast_to(np.asarray(y, np.float32), (n,)), np.broadcast_to(np.asarray(col, np.float32), (n, 3))))

        # the shore, bottom left
        xs = np.arange(W)
        ysh = H * (0.835 + 0.20 * (xs / (0.40 * W)) ** 1.7) + 0.006 * H * np.sin(xs / W * 47)
        rows = np.arange(H)[:, None]
        ys_, xs_ = np.nonzero(rows >= ysh[None, :])
        d = np.clip((ys_ - ysh[xs_]) / (0.12 * H), 0, 1)
        col = np.array([56, 46, 40], np.float32) * (1.15 - 0.45 * d)[:, None] * (0.85 + 0.3 * rng.random((len(d), 1)))
        put(xs_ + rng.random(len(xs_)), ys_ + rng.random(len(ys_)), col, 1.0, np.clip(1 - (ys_ - ysh[xs_]) / (0.012 * H), 0, 1), 0)
        n = 1400                                                              # grass along its edge
        gx = rng.uniform(0, 0.42 * W, n)
        gy = np.interp(gx, xs, ysh) + rng.random(n) * 0.025 * H
        for q in np.linspace(0, 1, 7):
            lean = rng.normal(0, 0.4, n)
            put(gx + lean * q * 6 * S / 1000, gy - q * (5 + 9 * rng.random(n)) * S / 1000,
                np.array([66, 92, 52], np.float32) * (0.7 + 0.5 * rng.random((n, 1))), 1.0, np.full(n, q), 0)
        # the cherry tree, framing the top left
        branches = [
            ([(0.05, 1.03), (0.03, 0.80), (0.12, 0.62), (0.13, 0.42)], 0.030, 0.017),
            ([(0.13, 0.43), (0.15, 0.25), (0.24, 0.12), (0.37, 0.05)], 0.016, 0.004),
            ([(0.13, 0.46), (0.06, 0.32), (0.04, 0.16), (0.08, 0.01)], 0.013, 0.004),
            ([(0.20, 0.16), (0.24, 0.22), (0.27, 0.29), (0.29, 0.36)], 0.007, 0.002),
            ([(0.27, 0.10), (0.29, 0.04), (0.32, 0.00), (0.35, -0.03)], 0.006, 0.002),
            ([(0.12, 0.56), (0.17, 0.52), (0.22, 0.49), (0.27, 0.51)], 0.008, 0.003),
        ]
        anchors = []
        for bi, (P, w0, w1) in enumerate(branches):
            P = [(a * W, b * H) for a, b in P]
            n = int(4200 * (w0 / 0.015) ** 0.7)
            s = rng.random(n)
            x, y, dx, dy = bezier(P, s)
            nn = np.hypot(dx, dy) + 1e-6
            v = rng.uniform(-1, 1, n)
            wd = (w0 + (w1 - w0) * s) * W / 2
            px, py = x - dy / nn * v * wd, y + dx / nn * v * wd
            bark = np.array([48, 34, 34], np.float32) * (0.7 + 0.5 * rng.random((n, 1)))
            bark[np.abs(v) > 0.82] = (18, 12, 14)                             # the key line
            put(px, py, bark, 1.5, np.clip(-dx / nn * v, 0, 1) ** 2, 1)
            if bi > 0:
                for s0 in np.linspace(0.2, 1.0, 12 if bi in (1, 2) else 7):
                    x0, y0, _, _ = bezier(P, np.array([s0]))
                    anchors.append((float(x0[0]), float(y0[0])))
        for x0, y0 in anchors:                                                # twigs and blossom clusters
            for _ in range(3):
                a = rng.uniform(-math.pi, math.pi * 0.3)
                ln = rng.uniform(0.008, 0.035) * W
                q = np.linspace(0, 1, 10)
                tx, ty = x0 + np.cos(a) * ln * q, y0 + np.sin(a) * ln * q + 0.3 * ln * q ** 2
                put(tx, ty, np.array([40, 28, 30], np.float32), 1.0, q * 0, 1)
                r = rng.uniform(0.014, 0.030) * W
                m = int(120 * (r / (0.02 * W)) ** 2) + 30
                ang, rr = rng.uniform(0, TAU, m), r * np.sqrt(rng.random(m)) * (0.75 + 0.25 * np.sin(rng.uniform(0, TAU)))
                bx, by = tx[-1] + np.cos(ang) * rr, ty[-1] + np.sin(ang) * rr * 0.8
                pal = np.array([(250, 214, 224), (244, 190, 206), (255, 238, 242), (232, 160, 184)], np.float32)
                c = pal[rng.integers(0, 4, m)] * (1 - 0.18 * (by > ty[-1]))[:, None]
                put(bx, by, c, rng.choice([1.5, 2.0, 2.5], m) * S / 1000 + 0.3, np.clip((ty[-1] - by) / r, 0, 1), 2)
        # the torii, standing in the water at the right
        xt, yb, ht = 0.80 * W, 0.845 * H, 0.20 * H
        self.tor_base = yb
        verm = np.array([214, 62, 40], np.float32)
        black = np.array([30, 24, 28], np.float32)

        def block(xc, half, y0, y1, col, curve=0.0, lean=0.0, dens=1.0):
            n = int(max(30, 2 * half * abs(y1 - y0) * 1.4 * dens))
            v = rng.uniform(-1, 1, n)
            y = rng.uniform(y0, y1, n)
            f = (y - y1) / (y0 - y1 + 1e-6)
            x = xc + lean * f + v * half
            y = y - curve * (v ** 4)
            c = col * (0.88 + 0.18 * rng.random((n, 1)))
            c[np.abs(v) > 0.9] *= 0.55
            put(x, y, c, 1.2, np.clip(1 - (y - y0 + curve * v ** 4) / 4.0, 0, 1) * 0.6, 3)

        for sgn in (-1, 1):
            block(xt + sgn * 0.047 * W, 0.0075 * W, yb - 0.86 * ht, yb, verm, lean=-sgn * 0.005 * W)
            block(xt + sgn * 0.047 * W, 0.0085 * W, yb - 0.035 * ht, yb, black)
        block(xt, 0.074 * W, yb - 0.705 * ht, yb - 0.66 * ht, verm)          # the tie beam
        block(xt, 0.006 * W, yb - 0.86 * ht, yb - 0.70 * ht, verm)            # the centre strut
        block(xt, 0.080 * W, yb - 0.905 * ht, yb - 0.86 * ht, verm, curve=0.012 * H)
        block(xt, 0.094 * W, yb - 0.975 * ht, yb - 0.905 * ht, black, curve=0.020 * H)
        m = msk[..., 0] > 0.5
        self.fg_idx = np.flatnonzero(m)
        self.fg_alb = alb.reshape(-1, 3)[self.fg_idx]
        self.fg_rim = rimm[..., 0].ravel()[self.fg_idx]
        self.fg_kind = np.rint(kind[..., 0].ravel()[self.fg_idx]).astype(np.int8)
        tx, ty, tc = (np.concatenate(a) for a in zip(*tor))
        ry = 2 * yb - ty
        ok = (ty < yb) & (ry < H - 1)
        self.tor_ref = (np.rint(tx[ok]).astype(int).clip(0, W - 1), np.rint(ry[ok]).astype(int), tc[ok])

    # ---- paper, and the print's cartouche and seal --------------------------------------------------------
    def build_paper(self):
        W, H = self.W, self.H
        rng = np.random.default_rng(99)
        small = rng.normal(0, 1, (H // 6 + 2, W // 6 + 2)).astype(np.float32)
        mott = np.asarray(Image.fromarray(small).resize((W, H), Image.BILINEAR))[:H, :W]
        grain = 1 + 0.030 * mott + 0.020 * rng.normal(0, 1, (H, W)).astype(np.float32)
        fib = np.zeros((H, W), np.float32)
        n = int(2200 * W * H / 1e6)
        x0, y0 = rng.uniform(0, W, n), rng.uniform(0, H, n)
        a = rng.uniform(0, TAU, n)
        ln = rng.uniform(6, 34, n)
        q = np.linspace(0, 1, 24)[None, :]
        fx = np.clip((x0[:, None] + np.cos(a)[:, None] * ln[:, None] * q).astype(int), 0, W - 1)
        fy = np.clip((y0[:, None] + np.sin(a)[:, None] * ln[:, None] * q + 2 * np.sin(q * 6)).astype(int), 0, H - 1)
        np.add.at(fib, (fy.ravel(), fx.ravel()), np.repeat(rng.choice([-0.05, 0.04], n), 24))
        self.paper = ((grain + fib)[..., None] * np.array([1.0, 0.99, 0.955], np.float32)).astype(np.float32)
        # a cartouche of invented characters and a red seal, top right, as on a print
        ov = np.zeros((H, W, 3), np.float32)
        om = np.zeros((H, W, 3), np.float32)
        x0, x1, y0, y1 = int(0.905 * W), int(0.955 * W), int(0.035 * H), int(0.235 * H)
        ov[y0:y1, x0:x1] = (236, 214, 160)
        om[y0:y1, x0:x1] = 1
        ov[y0:y0 + 2, x0:x1] = ov[y1 - 2:y1, x0:x1] = ov[y0:y1, x0:x0 + 2] = ov[y0:y1, x1 - 2:x1] = (40, 30, 30)
        cell = (x1 - x0) * 0.7

        def strokes(cx, cy, size, col, k):
            for _ in range(k):
                kind = rng.integers(0, 3)
                if kind == 0:
                    yy = cy + rng.uniform(-0.4, 0.4) * size
                    xa, xb = cx + rng.uniform(-0.45, -0.1) * size, cx + rng.uniform(0.1, 0.45) * size
                    pts = (np.linspace(xa, xb, 30), np.full(30, yy))
                elif kind == 1:
                    xx = cx + rng.uniform(-0.35, 0.35) * size
                    pts = (np.full(30, xx), np.linspace(cy + rng.uniform(-0.45, -0.1) * size, cy + rng.uniform(0.1, 0.45) * size, 30))
                else:
                    pts = (np.linspace(cx - 0.35 * size, cx + 0.35 * size, 30) * 1, np.linspace(cy - 0.3 * size, cy + 0.35 * size, 30))
                stamp(ov, pts[0], pts[1], np.asarray(col, np.float32), 1.0)
                stamp(om, pts[0], pts[1], np.ones(3, np.float32), 1.0)

        for i in range(5):
            strokes((x0 + x1) / 2, y0 + (i + 0.7) * (y1 - y0) / 5.4, cell, (40, 30, 30), int(rng.integers(3, 5)))
        sx0, sy0, sz = int(0.913 * W), int(0.250 * H), int(0.034 * W)
        ov[sy0:sy0 + sz, sx0:sx0 + sz] = (192, 40, 32)
        om[sy0:sy0 + sz, sx0:sx0 + sz] = 1
        strokes(sx0 + sz / 2, sy0 + sz / 2, sz * 0.8, (240, 220, 200), 4)
        m = om[..., 0] > 0.5
        self.ov_idx = np.flatnonzero(m)
        self.ov_col = ov.reshape(-1, 3)[self.ov_idx]

    # ---- things that move ----------------------------------------------------------------------------
    def build_motion(self):
        W, H, S = self.W, self.H, self.S
        rng = np.random.default_rng(31)
        n = 360                                                               # climbers, in little groups
        grp = rng.random(60)
        self.lamp_s0 = (np.repeat(grp, 6) + rng.normal(0, 0.004, 360))[:n] % 1.0
        self.lamp_ph = rng.uniform(0, TAU, n)
        self.huts = np.array([0.04, 0.17, 0.30, 0.43, 0.56, 0.69, 0.82, 0.985])
        n = 26                                                                # paper lanterns on the water
        d = rng.random(n) ** 0.8
        self.lant = dict(x=rng.uniform(0.16, 0.70, n) * W, d=d, y=self.yw + (0.05 + 0.88 * d) * (H - self.yw),
                         ph=rng.uniform(0, TAU, (2, n)))
        n = 110                                                               # cherry petals
        self.pet = dict(x0=rng.uniform(0.06, 0.48, n) * W, y0=rng.uniform(0.02, 0.42, n) * H, dx=rng.uniform(0.25, 0.75, n) * W,
                        dy=rng.uniform(0.25, 0.60, n) * H, k=rng.integers(2, 5, n), p=rng.random(n), ph=rng.uniform(0, TAU, n))
        n = 600                                                               # stylised ripple lines on the lake
        d = rng.random(n) ** 0.7
        self.rip = dict(x=rng.uniform(0, W, n), y=self.yw + 4 + d * (H - self.yw - 6), d=d, ph=rng.uniform(0, TAU, n),
                        ln=(5 + 38 * d ** 1.5) * S / 1000)
        self.bands = []
        for yf, hf, x0f, wf, ph in BANDS:
            h, w = max(3, int(hf * H)), int(wf * W)
            yy, xx = np.mgrid[0:h, 0:w]
            r = h / 2
            dy = np.abs(yy + 0.5 - r)
            ins = ~((xx < r) & ((xx + 0.5 - r) ** 2 + dy ** 2 > r * r)) & ~((xx > w - r) & ((xx + 0.5 - (w - r)) ** 2 + dy ** 2 > r * r))
            soft = np.clip(np.minimum(yy + 1, h - yy) / (0.18 * h), 0, 1)                 # crisp, barely softened edges
            self.bands.append((int(yf * H - h / 2), int(x0f * W), ins, (1.06 - 0.16 * yy / h)[..., None].astype(np.float32),
                               ph, soft.astype(np.float32)))
        self.shells = []
        for tl, xf, yf, c0, c1, size in SHELLS:
            m = int(240 * size)
            v = rng.normal(0, 1, (3, m))
            v /= np.linalg.norm(v, axis=0)
            self.shells.append(dict(tl=tl, x=xf * W, y=yf * H, c0=np.array(c0, np.float32), c1=np.array(c1, np.float32),
                                    size=size, dx=v[0], dy=v[1], id=rng.uniform(0, 100, m)))

    def wind(self, t):
        return 0.30 + 0.15 * math.sin(TAU * 2 * t / T_END) ** 2 + 0.55 * math.exp(-((t - 27.0) / 2.2) ** 2) \
            - 0.25 * math.exp(-((t - 32.5) / 2.0) ** 2)

    def trail(self, s):
        g = 0.40 + 0.585 * s
        y = self.yw - g * (self.yw - self.yp)
        hw = self.R * (1 - (1 - self.u0) * g ** (1 / 1.85))
        tri = 2 * np.abs((s * 9) % 1 - 0.5) * 2 - 1
        return self.cx + 0.30 * hw + 0.30 * hw * tri, y

    # ---- drawing ----------------------------------------------------------------------------------------
    def sun_xy(self, t):
        a = T_DIAMOND - t
        return self.cx - 0.010 * self.W * a, self.yp + 0.012 * self.H - 0.034 * self.H * a

    def draw_sun(self, img, t):
        if not 34.0 < t < 47.5:
            return
        sx, sy = self.sun_xy(t)
        warm = float(ss(t, 38, T_DIAMOND))
        core = np.array([255, 240, 200]) * (1 - warm) + np.array([255, 206, 140]) * warm
        k = float(ss(t, 34, 37) * (1 - ss(t, 46, 47.5)))
        self.sun.draw(img, sx, sy, core, np.array([170, 110, 50]) * k, self.skym, self.gmask)
        fd = math.exp(-((t - T_DIAMOND) / 0.9) ** 2)                          # Diamond Fuji
        if fd > 0.02:
            r = np.random.default_rng(7)
            ang = np.linspace(0, TAU, 16, endpoint=False) + 0.1
            ln = (0.10 + 0.16 * r.random(16)) * self.S * (0.6 + 0.4 * fd)
            q = np.linspace(0.05, 1, 70)[None, :]
            rx = self.cx + np.cos(ang)[:, None] * ln[:, None] * q
            ry = self.yp + np.sin(ang)[:, None] * ln[:, None] * q * 0.8
            b = np.broadcast_to(((1 - q) ** 1.5 * fd)[..., None] * np.array([255, 230, 180])[None, None] * 0.7, rx.shape + (3,))
            ok = ry < self.yw
            glow(img, rx[ok], ry[ok], b[ok], 1.2)
            glow(img, np.array([self.cx]), np.array([self.yp]), np.array([255, 250, 230]) * 2.5 * fd, 3.0)
            self.haze.add(np.full(8, self.cx), np.full(8, self.yp), np.array([255, 200, 120]) * 0.9 * fd, 40 * self.S / 1000)
            self.haze.add(np.full(4, self.cx), np.full(4, 2 * self.yw - self.yp), np.array([255, 180, 110]) * 0.4 * fd,
                          30 * self.S / 1000)

    def draw_climbers(self, img, t):
        nt = night(t)
        if nt > 0.01:
            s = (self.lamp_s0 + t / T_END) % 1.0
            x, y = self.trail(s)
            end = np.clip(s / 0.04, 0, 1) * np.clip((1 - s) / 0.03, 0, 1)
            b = nt * end * (0.7 + 0.3 * np.sin(TAU * 5 * t / T_END + self.lamp_ph))
            b = b * (1 - 0.6 * ss(s, 0.75, 1.0))                                  # the chain thins out near the top
            glow(img, x, y, np.array([255, 196, 110])[None] * (b * 0.9)[:, None], 1.0)
            self.haze.add(x[::4], y[::4], np.array([255, 150, 60])[None] * (b[::4] * 0.18)[:, None], 8 * self.S / 1000)
        hut = float(1 - ss(t, 14, 18) + ss(t, 47, 51))
        if hut > 0.01:
            x, y = self.trail(self.huts)
            glow(img, x + 3, y, np.array([255, 170, 80]) * 0.9 * hut, 1.4)

    def lens(self, img, xc, yc, w, h, lit, shade, ink, alpha):
        W, H = self.W, self.H
        xa, xb = max(int(xc - w / 2) - 1, 0), min(int(xc + w / 2) + 2, W)
        ya, yb = max(int(yc - h) - 2, 0), min(int(yc + h) + 3, self.yw)
        if xa >= xb or ya >= yb:
            return
        yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
        a = np.clip(1 - ((xx - xc) / (w / 2)) ** 2, 0, 1)
        half = h / 2 * a ** 0.65 * (1 + 0.06 * np.sin(xx * 0.15 + yc))
        top, bot = yc - half * 0.9, yc + half * 1.05
        ins = (yy >= top) & (yy <= bot) & (a > 0)
        v = np.clip((yy - top) / (bot - top + 1e-3), 0, 1)
        col = lit + (shade - lit) * v[..., None] ** 1.3
        edge = ins & ~(np.roll(ins, 1, 0) & np.roll(ins, -1, 0) & np.roll(ins, 1, 1) & np.roll(ins, -1, 1))
        col[edge] = ink
        reg = img[ya:yb, xa:xb]
        reg[ins] = reg[ins] * (1 - alpha) + col[ins] * alpha

    def draw_clouds(self, img, t):
        g = float(ss(t, 23.5, 26.5) * (1 - ss(t, 34.5, 37.5)))
        if g < 0.01:
            return
        lit, shade = keyed(t, self.keys["cl"]), keyed(t, self.keys["cs"])
        ink = shade * 0.55
        W, H, cx, yp = self.W, self.H, self.cx, self.yp
        br = 1 + 0.04 * math.sin(0.9 * t)
        al = min(1.0, g * 1.6)
        for dx, dy, w, h in ((0.0, 0.004, 0.22, 0.042), (0.004, -0.024, 0.17, 0.020), (-0.006, -0.043, 0.12, 0.013)):
            self.lens(img, cx + dx * W, yp + dy * H, w * W * g ** 0.5 * br, h * H * g ** 0.6, lit, shade, ink, al)
        hx = cx + 0.27 * W + 0.05 * W * (t - 23.5) / 14                       # a hanging cloud drifting beside it
        for dy, w, h in ((0.07, 0.13, 0.022), (0.052, 0.09, 0.011)):
            self.lens(img, hx, yp + dy * H, w * W * g ** 0.5, h * H * g ** 0.6, lit, shade, ink, al)

    def draw_kasumi(self, img, t):
        c = keyed(t, self.keys["kasumi"])
        a = float(keyed(t, self.keys["ka"]))
        W = self.W
        for y0, x0, ins, sh, ph, soft in self.bands:
            x = int(x0 + 0.05 * W * math.sin(TAU * t / T_END + ph))
            h, w = ins.shape
            xa, xb = max(x, 0), min(x + w, W)
            ya, yb = max(y0, 0), min(y0 + h, self.yw)
            if xa >= xb or ya >= yb:
                continue
            reg = img[ya:yb, xa:xb]
            m = ins[ya - y0:yb - y0, xa - x:xb - x]
            al = (soft[ya - y0:yb - y0, xa - x:xb - x] * a)[m][:, None]
            tgt = np.broadcast_to(c * sh[ya - y0:yb - y0], (yb - ya, w, 3))[:, xa - x:xb - x]
            reg[m] = reg[m] * (1 - al) + tgt[m] * al

    def draw_cranes(self, img, t):
        a = (t - 26.0) / 10.0
        if not 0 < a < 1:
            return
        W, H, S = self.W, self.H, self.S
        lx, ly = -0.10 * W + 1.20 * W * a, 0.425 * H + 0.02 * H * math.sin(a * 4)
        L = 0.064 * S
        white, black, red = np.array([242, 242, 236], np.float32), np.array([22, 22, 26], np.float32), np.array([220, 30, 30], np.float32)
        r = np.random.default_rng(3)
        for i, (ox, oy) in enumerate(((0, 0), (1, 1), (1, -1), (2, 2), (2, -2), (3, 3), (3, -3))):
            x0, y0 = lx - ox * 0.06 * W, ly + oy * 0.022 * H
            flap = math.sin(TAU * 1.5 * t + i * 0.9)
            pts = []
            n = 220
            ang, rr = r.uniform(0, TAU, n), np.sqrt(r.random(n))
            pts.append((0.22 * rr * np.cos(ang), 0.06 * rr * np.sin(ang), white))
            q = np.linspace(0, 1, 22)
            pts.append((0.18 + 0.37 * q, 0.02 + 0.03 * q, black))                # neck
            pts.append((np.array([0.58, 0.60, 0.62]), np.array([0.05, 0.05, 0.045]), white))
            pts.append((np.array([0.58]), np.array([0.075]), red))
            pts.append((0.62 + 0.09 * q[:8], 0.045 - 0.01 * q[:8], np.array([150, 140, 110], np.float32)))
            pts.append((-0.18 - 0.42 * q, -0.02 - 0.03 * q, black))              # legs trailing
            for far, fl in ((0.55, math.sin(TAU * 1.5 * t + i * 0.9 + 0.6)), (1.0, flap)):
                sp, ch = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 8))      # long, narrow wings
                chord = 0.17 * (1 - sp) ** 0.7 + 0.05
                u = 0.08 - 0.06 * sp - ch * chord
                w_ = 0.02 + fl * 0.95 * sp ** 1.1 + 0.04 * sp * (1 - abs(fl))
                col = np.where(((sp > 0.80) | ((ch > 0.7) & (sp > 0.25)))[..., None], black, white) * far
                pts.append((u.ravel(), w_.ravel(), col.reshape(-1, 3)))
            X = np.concatenate([p[0] for p in pts])
            Y = np.concatenate([p[1] for p in pts])
            C = np.concatenate([np.broadcast_to(p[2], (len(np.atleast_1d(p[0])), 3)) for p in pts])
            stamp(img, x0 + X * L, y0 - Y * L, C, 1.0 + 0.5 * S / 1000)

    def draw_fireworks(self, img, t):
        if not 49.0 < t < 59.0:
            return
        W, H, S = self.W, self.H, self.S
        for sh in self.shells:
            a = t - sh["tl"]
            if a < 0 or a > 4.0:
                continue
            if a < 0.9:                                                       # the shell climbing
                p = a / 0.9
                y = self.yw - (self.yw - sh["y"]) * (1 - (1 - p) ** 2)
                q = np.linspace(0, 0.12, 8)
                glow(img, np.full(8, sh["x"]) + np.sin(q * 50) * 1.5, y + q * (self.yw - sh["y"]) * 0.4,
                     np.array([255, 200, 140])[None] * (1 - q / 0.12)[:, None] * 0.9, 1.0)
                continue
            a -= 0.9
            R = 0.16 * S * sh["size"]
            out = []
            for lag, k in ((0.0, 1.0), (0.06, 0.55), (0.12, 0.3)):
                aa = max(a - lag, 0)
                rr = R * (1 - math.exp(-aa / 0.45))
                x = sh["x"] + sh["dx"] * rr
                y = sh["y"] + sh["dy"] * rr + 0.028 * S * aa * aa
                b = math.exp(-aa / 1.2) * k * (aa < 3.0) * 1.8
                if aa > 1.3:
                    b = b * (0.4 + 0.6 * (np.sin(80 * aa + sh["id"]) > 0))
                col = sh["c0"] + (sh["c1"] - sh["c0"]) * min(aa / 2.0, 1)
                out.append((x, y, np.broadcast_to(col * b, (len(sh["dx"]), 3)) if np.ndim(b) == 0 else col[None] * b[:, None]))
            x = np.concatenate([o[0] for o in out])
            y = np.concatenate([o[1] for o in out])
            c = np.concatenate([o[2] for o in out])
            ok = (y < self.yw - 1) & (y > 0) & (x > 0) & (x < W)
            glow(img, x[ok], y[ok], c[ok], 1.5)
            f = math.exp(-a / 0.18)
            if f > 0.02:
                self.haze.add(np.full(4, sh["x"]), np.full(4, sh["y"]), sh["c1"] * 0.8 * f, 50 * S / 1000)
                self.haze.add(np.full(4, sh["x"]), np.full(4, 2 * self.yw - sh["y"]), sh["c1"] * 0.35 * f, 40 * S / 1000)

    def draw_lake(self, img, t):
        W, H, S, yw = self.W, self.H, self.S, self.yw
        rows = np.arange(yw, H)
        dist = (rows - yw) / (H - yw)
        src = 2 * yw - rows - 1
        w = self.wind(t)
        amp = (0.4 + 7 * dist ** 1.2) * S / 1000 * (0.25 + 1.3 * w)
        dx = np.rint(amp * (np.sin(rows * 0.55 + TAU * 7 * t / T_END) + 0.6 * np.sin(rows * 0.23 - TAU * 5 * t / T_END + 1.3))).astype(int)
        cols = np.clip(self.xidx[None, :] + dx[:, None], 0, W - 1)
        refl = img[src[:, None], cols]
        k = (0.80 - 0.25 * dist)[:, None, None].astype(np.float32)
        water = keyed(t, self.keys["water"]).astype(np.float32)
        img[yw:] = refl * k + water * (1 - k)
        R_ = self.rip                                                       # ripple lines, like a print's waves
        low = keyed(t, self.keys["low"])
        q = np.linspace(-1, 1, 7)[None, :]
        x = R_["x"][:, None] + 0.01 * W * np.sin(TAU * 2 * t / T_END + R_["ph"])[:, None] + R_["ln"][:, None] * q
        y = R_["y"][:, None] - R_["ln"][:, None] * 0.14 * (1 - q ** 2)
        b = (0.10 + 0.55 * w) * (0.5 + 0.5 * np.sin(TAU * 3 * t / T_END + R_["ph"])) * (0.3 + 0.7 * R_["d"])
        glow(img, x.ravel(), y.ravel(), np.repeat((low[None] * 0.45 * b[:, None]), 7, 0), 1.0)
        mist = float(ss(t, 12, 15) * (1 - ss(t, 22, 26)))                     # morning mist lying on the water
        if mist > 0.01:
            c = keyed(t, self.keys["kasumi"])
            yy = np.arange(yw - int(0.02 * H), yw + int(0.05 * H))
            edge = 0.5 + 0.5 * np.sin(np.arange(W) / W * TAU * 3 + 0.6 * t)[None, :]
            v = (yy[:, None] - (yw - 0.02 * H)) / (0.07 * H)
            al = np.clip(np.sin(np.pi * v) * (0.6 + 0.4 * edge), 0, 1) * 0.38 * mist
            img[yy] = img[yy] * (1 - al[..., None]) + c * al[..., None]
        return dx

    def draw_lanterns(self, img, t, dx):
        nt = night(t)
        if nt < 0.01:
            return
        Ln, W, S = self.lant, self.W, self.S
        x = Ln["x"] + 0.05 * W * np.sin(TAU * t / T_END + Ln["ph"][0]) + 0.01 * W * np.sin(TAU * 3 * t / T_END + Ln["ph"][1])
        y = Ln["y"]
        r = (1.0 + 3.0 * Ln["d"]) * S / 1000
        fl = 0.85 + 0.15 * np.sin(TAU * 13 * t / T_END + Ln["ph"][1] * 5)
        for ox, oy, c in ((0, -1.5, (255, 220, 150)), (0, 0, (255, 170, 80)), (-1, 0, (255, 150, 70)), (1, 0, (255, 150, 70))):
            glow(img, x + ox * r, y + oy * r, np.array(c)[None] * (nt * fl)[:, None] * 1.1, np.maximum(r * 0.8, 1.0))
        for j in range(1, 9):                                                  # their light trembling in the water
            yy = y + j * r * 1.4
            row = np.clip((yy - self.yw).astype(int), 0, len(dx) - 1)
            glow(img, x + dx[row] * 0.6, yy, np.array([255, 150, 60])[None] * (nt * fl * 0.5 * (1 - j / 9))[:, None], 1.0)
        self.haze.add(x, y, np.array([255, 140, 60])[None] * (0.5 * nt * fl)[:, None], (14 + 20 * Ln["d"]) * S / 1000)

    def draw_torii_ref(self, img, t, dx, amb):
        x, ry, c = self.tor_ref
        row = np.clip(ry - self.yw, 0, len(dx) - 1)
        rx = np.clip(x + dx[row], 0, self.W - 1)
        col = c * amb * 0.5 + 40 * night(t) * np.array([1.0, 0.6, 0.3])
        img[ry, rx] = img[ry, rx] * 0.35 + col * 0.65

    def draw_fg(self, img, t, amb, key):
        back = float(ss(t, 36, 40) * (1 - ss(t, 46, 49)))
        nt = night(t)
        c = self.fg_alb * amb + key[None] * (self.fg_rim * (0.25 + 0.75 * back))[:, None] * 0.45
        warm = ((self.fg_kind == 2) | (self.fg_kind == 3)).astype(np.float32)   # lantern light on blossom and torii
        c += np.array([90, 50, 26], np.float32)[None] * (warm * 0.5 * nt)[:, None]
        img.reshape(-1, 3)[self.fg_idx] = c

    def draw_petals(self, img, t, amb, key):
        P = self.pet
        f = (t * P["k"] / T_END + P["p"]) % 1.0
        on = (f > 0.03) & (f < 0.95)
        f, idx = f[on], np.nonzero(on)[0]
        x = P["x0"][idx] + P["dx"][idx] * f + 0.02 * self.W * np.sin(TAU * f * 3 + P["ph"][idx])
        y = P["y0"][idx] + P["dy"][idx] * f + 0.01 * self.H * np.sin(TAU * f * 5 + P["ph"][idx])
        a = TAU * f * 4 + P["ph"][idx]
        col = np.array([248, 200, 214], np.float32) * np.maximum(amb, 0.45) + key * 0.15
        s = 1.4 * self.S / 1000
        for o in (-1, 0, 1):
            stamp(img, x + o * s * np.cos(a), y + o * s * np.sin(a) * 0.5, col, 1.0 + 0.4 * self.S / 1000)

    # ---- the frame ---------------------------------------------------------------------------------
    def render(self, t):
        t = t % T_END
        img = self.look_at(t)
        amb = keyed(t, self.keys["amb"])
        key = keyed(t, self.keys["key"])
        self.draw_sun(img, t)
        self.draw_climbers(img, t)
        self.draw_clouds(img, t)
        self.draw_kasumi(img, t)
        self.draw_cranes(img, t)
        self.draw_fireworks(img, t)
        dx = self.draw_lake(img, t)
        self.draw_lanterns(img, t, dx)
        self.draw_torii_ref(img, t, dx, amb)
        self.draw_fg(img, t, amb, key)
        self.draw_petals(img, t, amb, key)
        self.haze.apply(img, 1.0)
        img.reshape(-1, 3)[self.ov_idx] = self.ov_col
        img *= self.paper
        return np.clip(img, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    run(Fuji, "Fuji - a day and a night", "fuji", T_END, STAGES, wrap=True)
