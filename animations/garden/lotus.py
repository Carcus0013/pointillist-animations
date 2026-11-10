"""
lotus.py - a lotus on a pond through a summer rain, in dots.

Rain hisses on a grey pond and rings spread across the water. Rolled lotus pads unfurl
on the surface; others rise on stalks, cupped like bowls, holding beads of water. A bud
pushes up out of the pond. The rain stops, the clouds open, a rainbow hangs over the far
trees, and everything doubles in the still water. The lotus opens - pink-tipped petals
around a ring of golden stamens and a green seed pod - dragonflies dart in and a koi
slides by under the surface. As the sun goes down the petals let go one by one and float
away, leaving the seed pod nodding over its own reflection.

    python lotus.py            interactive window (Space pause, R restart, S save frame, Esc quit)
    python lotus.py --png      save five stages
    python lotus.py --gif      save the whole film as a GIF
"""
import math

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))   # the shared engine: engine/pointillism.py

from pointillism import (Backdrop, Sun, bezier, cloud_streaks, glow, join, keyed, lights, lit, mix, paint, run,
                         sample_u, sample_v, shades, sky, ss, stamp, unit)

T_END = 60.0
STAGES = [6.0, 20.0, 33.0, 46.0, 57.0]

PAD = [(50, 100, 88), (66, 120, 100), (88, 140, 114), (114, 160, 126)]
PAD_VEIN = [(136, 178, 148), (150, 190, 156)]
PAD_RIM = [(96, 122, 80), (110, 130, 84)]
PETAL = [(252, 246, 236), (252, 224, 230), (248, 184, 206), (238, 128, 168), (222, 86, 138)]
POD_TOP = [(214, 212, 112), (196, 204, 96), (224, 220, 130)]
POD_SIDE = [(116, 158, 70), (98, 140, 60)]
POD_OLD = [(132, 104, 58), (108, 84, 48), (96, 92, 52)]
STEM = [(70, 112, 62), (86, 130, 70), (60, 98, 56)]
STAMEN = [(255, 206, 70), (250, 180, 44), (255, 222, 110)]
BANDS = [(255, 70, 70), (255, 150, 60), (255, 230, 90), (120, 220, 110), (90, 160, 255), (110, 100, 220), (170, 100, 220)]
CLOUDS = [(0.20, 0.10, 0.26, 0.030, 9000), (0.62, 0.07, 0.30, 0.026, 9000), (0.82, 0.24, 0.20, 0.022, 6000),
          (0.36, 0.27, 0.24, 0.020, 6000), (0.10, 0.35, 0.16, 0.014, 3500), (0.66, 0.38, 0.18, 0.014, 3500)]

# pads: x, waterline y, radius, height above the water (0 = floating), when it unfurls
PADS = [
    (0.27, 0.74, 0.068, 0.10, 1.0), (0.69, 0.71, 0.058, 0.13, 4.0), (0.60, 0.80, 0.075, 0.045, 7.0),
    (0.16, 0.88, 0.090, 0.0, 0.0), (0.84, 0.90, 0.100, 0.0, 2.5), (0.40, 0.64, 0.050, 0.0, 5.0),
    (0.88, 0.63, 0.042, 0.0, 8.0), (0.08, 0.66, 0.045, 0.0, 3.5),
]
# petal layers, outside in: count, length, width, open angle, bud angle
LAYERS = [(6, 1.00, 0.56, 1.16, 0.24), (8, 0.94, 0.52, 0.86, 0.16), (8, 0.84, 0.48, 0.56, 0.08)]

LOOKS = [
    dict(t=0.0, sky=[(0.0, (88, 96, 112)), (0.30, (112, 120, 134)), (0.55, (146, 152, 162))],
         lit=[(150, 156, 168), (164, 170, 180)], shade=[(78, 84, 98), (92, 98, 112)],
         shore=[(30, 42, 42), (40, 54, 52), (52, 66, 60)], mist=(110, 116, 124), water=(56, 70, 80),
         amb=(0.72, 0.76, 0.82), key=(40, 40, 44)),
    dict(t=14.0, sky=[(0.0, (104, 116, 138)), (0.30, (140, 150, 168)), (0.55, (186, 188, 194))],
         lit=[(196, 198, 204), (210, 210, 214)], shade=[(110, 116, 132), (124, 130, 146)],
         shore=[(34, 50, 46), (46, 64, 56), (60, 80, 64)], mist=(120, 124, 130), water=(66, 84, 96),
         amb=(0.82, 0.85, 0.90), key=(70, 66, 60)),
    dict(t=25.0, sky=[(0.0, (64, 124, 198)), (0.30, (132, 176, 224)), (0.55, (214, 224, 226))],
         lit=[(255, 252, 244), (246, 246, 250)], shade=[(170, 182, 206), (190, 198, 218)],
         shore=[(34, 66, 44), (52, 90, 54), (80, 116, 66)], mist=(100, 110, 112), water=(46, 88, 108),
         amb=(1.0, 1.0, 1.0), key=(110, 100, 80)),
    dict(t=38.0, sky=[(0.0, (76, 116, 186)), (0.30, (170, 176, 204)), (0.55, (255, 210, 150))],
         lit=[(255, 224, 176), (255, 212, 160)], shade=[(160, 156, 188), (176, 166, 194)],
         shore=[(40, 58, 40), (62, 82, 48), (96, 106, 60)], mist=(120, 104, 80), water=(56, 82, 100),
         amb=(0.94, 0.88, 0.80), key=(190, 136, 70)),
    dict(t=50.0, sky=[(0.0, (46, 38, 96)), (0.28, (164, 78, 118)), (0.45, (246, 128, 80)), (0.55, (255, 186, 110))],
         lit=[(255, 150, 104), (255, 126, 110)], shade=[(96, 54, 104), (120, 62, 110)],
         shore=[(30, 26, 40), (44, 34, 50), (60, 44, 56)], mist=(110, 70, 60), water=(60, 52, 80),
         amb=(0.70, 0.56, 0.62), key=(230, 124, 64)),
    dict(t=58.0, sky=[(0.0, (18, 20, 56)), (0.28, (54, 46, 96)), (0.55, (140, 86, 110))],
         lit=[(150, 110, 130), (130, 100, 130)], shade=[(40, 36, 70), (52, 44, 80)],
         shore=[(14, 16, 28), (20, 22, 36), (28, 28, 44)], mist=(46, 38, 56), water=(26, 28, 50),
         amb=(0.46, 0.44, 0.60), key=(110, 70, 70)),
]


def petal_width(u):
    return np.sin(np.pi * np.clip(u, 0, 1) ** 1.25) ** 0.6


class Lotus:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.rng = np.random.default_rng(31)
        self.yw = int(0.56 * H)                 # far edge of the pond
        self.fx, self.fwl = 0.48 * W, 0.70 * H  # where the lotus stalk leaves the water
        self.Lp = 0.085 * H                     # petal length
        self.r0 = 0.030 * H                     # seed pod radius
        self.sun = Sun(np.random.default_rng(6), 0.040 * H, 5.0)
        self.amb_keys = [(L["t"], L["amb"]) for L in LOOKS]
        self.key_keys = [(L["t"], L["key"]) for L in LOOKS]
        self.water_keys = [(L["t"], L["water"]) for L in LOOKS]
        self.build_backdrop()
        self.build_pads()
        self.build_flower()
        self.build_falls()
        self.build_events()
        self.build_fx()

    # ---- the pond --------------------------------------------------------------------------
    def shore_top(self, x):
        xn = x / self.W
        return self.yw - self.H * (0.030 + 0.016 * (0.5 + 0.5 * np.sin(xn * 2 * np.pi * 1.6 + 0.8))
                                   + 0.012 * np.abs(np.sin(xn * 2 * np.pi * 9.0 + 1.3)) ** 0.6
                                   + 0.006 * np.abs(np.sin(xn * 2 * np.pi * 23 + 0.2)))

    def build_backdrop(self):
        W, H = self.W, self.H
        self.backdrop = Backdrop([L["t"] for L in LOOKS], [self.background(L) for L in LOOKS])
        yy = np.arange(H, dtype=np.float32)[:, None]
        top = self.shore_top(np.arange(W, dtype=np.float32))
        self.sky_mask = (yy < top[None, :]).astype(np.float32)
        self.glow_mask = (yy < self.yw).astype(np.float32) * np.ones((1, W), np.float32)
        self.xs = np.arange(W)
        self.rows = np.arange(self.yw, H)

    def background(self, L):
        W, H, yw = self.W, self.H, self.yw
        rng = np.random.default_rng(77)
        img = sky(W, H, rng, L["sky"], yw)
        cloud_streaks(img, rng, CLOUDS, L["lit"], L["shade"])
        # the far shore: a bank of trees
        xs = np.arange(W, dtype=np.float32)
        top = self.shore_top(xs)
        yy = np.arange(H, dtype=np.float32)[:, None]
        img[(yy >= top[None, :]) & (yy < yw)] = np.asarray(L["shore"][0], np.float32) * 0.9
        n = int(W * 0.06 * H * 0.9)
        x = rng.uniform(0, W, n)
        tp = np.interp(x, xs, top)
        d = rng.random(n)
        y = tp + d * (yw - tp)
        col = shades(rng, L["shore"], (1 - d) * 0.8 + 0.2 * rng.random(n), 0.6)
        stamp(img, x, y, col, rng.choice([1.0, 1.5, 2.0, 2.5], n, p=[0.2, 0.35, 0.3, 0.15]))
        n = int(W * 3.0)                                            # mist along the far bank
        glow(img, rng.uniform(0, W, n), yw - np.abs(rng.normal(0, 0.02 * H, n)), np.asarray(L["mist"]) * 0.14, 3.0)
        # the water mirrors everything above it, darker further out towards the viewer
        rows = np.arange(yw, H)
        src = np.clip(2 * yw - rows, 0, H - 1)
        k = (0.70 - 0.32 * (rows - yw) / (H - yw))[:, None, None]
        img[yw:] = img[src] * k + np.asarray(L["water"], np.float32) * (1 - k)
        n = int(W * (H - yw) * 0.06)                               # horizontal ripple texture
        x, y = rng.uniform(0, W, n), rng.uniform(yw, H, n)
        base = img[y.astype(int), np.clip(x.astype(int), 0, W - 1)] * (1 + rng.normal(0, 0.08, (n, 1)))
        stretch = 1.0 + 2.0 * (y - yw) / (H - yw)
        for dx in (-1.0, 0.0, 1.0):
            stamp(img, x + dx * stretch, y, base, 1.0)
        return img

    def ripple(self, y, t, rain):
        d = (np.asarray(y, np.float64) - self.yw) / (self.H - self.yw)
        return (1.2 + 4.0 * d) * (1 + 1.4 * rain) * (0.6 * np.sin(0.11 * y + 1.9 * t) + 0.4 * np.sin(0.037 * y - 1.1 * t + 2))

    # ---- build the plants once ------------------------------------------------------------------
    def build_pads(self):
        rng = self.rng
        self.pads = []
        for x, wl, R, h, start in PADS:
            n = int(1700 * (R / 0.08) ** 2)
            r = np.sqrt(rng.random(n))
            th = rng.uniform(0, 2 * np.pi, n)
            vein = np.abs(np.sin(th * 9)) < 0.07
            col = shades(rng, PAD, 0.3 + 0.6 * r + rng.normal(0, 0.1, n), 0.5)
            col[vein] = shades(rng, PAD_VEIN, rng.random(int(vein.sum())))
            col[r < 0.12] = np.array([170, 196, 160], np.float32) * (1 + rng.normal(0, 0.05, (int((r < 0.12).sum()), 1)))
            rim = r > 0.94
            col[rim] = shades(rng, PAD_RIM, rng.random(int(rim.sum())))
            nd = 3 if h > 0 else 2                                   # beads of rain water
            drops = dict(r=rng.uniform(0.0, 0.25 if h > 0 else 0.7, nd), th=rng.uniform(0, 2 * np.pi, nd),
                         ph=rng.uniform(0, 6.28, nd), w=rng.uniform(1.0, 2.4, nd))
            self.pads.append(dict(x=x * self.W, wl=wl * self.H, R=R * self.H, h=h * self.H, start=start,
                                  r=r, th=th, col=col, rad=rng.choice([1.5, 2.0], n, p=[0.5, 0.5]), drops=drops,
                                  stalk=shades(rng, STEM, rng.random(160))))

    def build_flower(self):
        rng = self.rng
        parts, k = [], 0
        for layer, (cnt, ln, wd, ao, ab) in enumerate(LAYERS):
            for i in range(cnt):
                d = 240
                u, v = sample_u(rng, d, petal_width), sample_v(rng, d, 0.12)
                live = shades(rng, PETAL, u * 0.95 + rng.normal(0, 0.06, d) - 0.12 * layer, 0.3)
                live *= rng.uniform(0.86, 1.08)                          # each petal its own tone
                vein = np.abs(np.mod((v + 1) * 5, 1.0) - 0.5) < 0.06
                live[vein] = live[vein] * np.array([0.98, 0.9, 0.94])
                rim = np.abs(v) > 0.8                                    # a deeper edge keeps petals apart
                live[rim] = live[rim] * np.array([0.88, 0.74, 0.82])
                if layer == 0:                                       # green-flushed outer petal bases
                    base = u < 0.18
                    live[base] = mix(live[base], np.array([170, 186, 130], np.float32), 0.6)
                parts.append(dict(k=np.full(d, k), layer=np.full(d, layer),
                                  th=np.full(d, 2 * np.pi * (i + 0.5 * layer) / cnt + rng.normal(0, 0.05)),
                                  ln=np.full(d, ln * rng.uniform(0.95, 1.05)), wd=np.full(d, wd), ao=np.full(d, ao),
                                  ab=np.full(d, ab), u=u, v=v, live=live,
                                  rad=rng.choice([1.0, 1.5, 2.0], d, p=[0.35, 0.5, 0.15])))
                k += 1
        self.petals = join(parts)
        self.npetal = k
        order = np.argsort(np.array([0] * 6 + [1] * 8 + [2] * 8) + rng.uniform(0, 0.8, k))
        self.petal_fall = np.empty(k)
        self.petal_fall[order] = 42.0 + 10.0 * np.arange(k) / k        # outer petals let go first
        n = 150
        self.stamens = dict(th=rng.uniform(0, 2 * np.pi, n), ln=rng.uniform(0.6, 1.0, n), rn=rng.random(n),
                            anther=shades(rng, STAMEN, rng.random(n)))
        n_top, n_side = 320, 220
        hole_r = np.r_[0.0, np.full(6, 0.45), np.full(12, 0.8)]
        hole_a = np.r_[0.0, np.arange(6) * np.pi / 3, np.arange(12) * np.pi / 6 + 0.26]
        self.pod = dict(top_r=np.sqrt(rng.random(n_top)), top_a=rng.uniform(0, 2 * np.pi, n_top),
                        side_h=rng.random(n_side), side_a=rng.uniform(0, 2 * np.pi, n_side),
                        top_col=shades(rng, POD_TOP, rng.random(n_top)), side_col=shades(rng, POD_SIDE, rng.random(n_side)),
                        old_top=shades(rng, POD_OLD, rng.random(n_top)), old_side=shades(rng, POD_OLD, rng.random(n_side)),
                        hole_r=hole_r, hole_a=hole_a)
        n = 900
        self.stalk = dict(s=rng.random(n), lat=sample_v(rng, n, 0.2), col=shades(rng, STEM, rng.random(n)))

    def flower_pose(self, t):
        H = self.H
        grow = float(ss(t, 6.0, 20.0))
        bow = float(ss(t, 50.0, 58.0))
        h = 0.25 * H * grow
        sway = 0.004 * H * math.sin(0.7 * t)
        top = (self.fx + sway + 0.03 * H * bow, self.fwl - h + 0.025 * H * bow)
        P = ((self.fx, self.fwl + 0.02 * H), (self.fx - 0.01 * H, self.fwl - 0.4 * h),
             (top[0] - 0.012 * H * (1 - bow), self.fwl - 0.8 * h), top)
        return dict(top=top, P=P, tilt=0.38 + 0.55 * bow, hs=0.25 + 0.75 * float(ss(t, 8.0, 20.0)), grow=grow)

    @staticmethod
    def tilt(x, y, z, cx, cy, tau):
        ct, st = math.cos(tau), math.sin(tau)
        return cx + x, cy - (y * ct - z * st), y * st + z * ct

    def petal_points(self, g, t, fp):
        """Positions of petal dots (dict of per-dot arrays) at time t, in flower space."""
        layer, u, v, th = g["layer"], g["u"], g["v"], g["th"]
        o = ss(t, 22.0 + 1.6 * layer, 30.0 + 1.6 * layer)
        alpha = g["ab"] + (g["ao"] - g["ab"]) * o
        kappa = -0.95 * (1 - o) - 0.45 * o                              # tips always curve in: a cup
        L = self.Lp * fp["hs"] * g["ln"]
        Wd = L * g["wd"]
        am = alpha + kappa * u * 0.5
        ct, st, ca, sa = np.cos(th), np.sin(th), np.cos(am), np.sin(am)
        w = petal_width(u)
        bulge = 0.10 * L * np.sin(np.pi * u) * (1 - o)                 # a plump bud
        r = self.r0 * fp["hs"] + L * u * sa + bulge
        y = L * u * ca
        lat, cup = v * Wd * w * 0.5, 0.28 * Wd * w * v ** 2
        x = ct * r - st * lat + cup * (-ca * ct)
        z = st * r + ct * lat + cup * (-ca * st)
        y = y + cup * sa
        return x, y, z, o

    def build_falls(self):
        """Where each petal is when it lets go, and where it lands on the water."""
        rng, K = self.rng, self.npetal
        P = self.petals
        cx, cy = np.zeros(K), np.zeros(K)
        for k in range(K):
            T = self.petal_fall[k]
            fp = self.flower_pose(T)
            m = P["k"] == k
            g = {key: val[m] for key, val in P.items()}
            x, y, z, _ = self.petal_points(g, T, fp)
            X, Y, _ = self.tilt(x, y, z, fp["top"][0], fp["top"][1], fp["tilt"])
            cx[k], cy[k] = X.mean(), Y.mean()
        lx = self.fx + rng.normal(0, 0.11, K) * self.W
        ly = self.fwl + rng.uniform(0.01, 0.14, K) * self.H
        self.falls = dict(T=self.petal_fall, cx=cx, cy=cy, lx=lx, ly=ly, dur=rng.uniform(1.4, 2.4, K),
                          ph=rng.uniform(0, 6.28, K), a0=rng.uniform(0, 2 * np.pi, K),
                          vx=self.W * rng.uniform(-0.006, 0.010, K), spin=rng.uniform(-0.15, 0.15, K))

    def build_events(self):
        """Every ring on the water: rain, petals landing, drips, the koi."""
        rng, W, H, yw = self.rng, self.W, self.H, self.yw
        ev = []
        t = 0.0
        while t < 21.0:
            t += rng.exponential(1.0 / (70.0 * (1 - float(ss(t, 12.0, 20.0))) + 0.6))
            ev.append((t, rng.uniform(0, W), yw + (H - yw) * rng.random() ** 0.8, 0.5 + 0.5 * rng.random(), 1.3))
        F = self.falls
        for k in range(self.npetal):
            ev.append((F["T"][k] + F["dur"][k], F["lx"][k], F["ly"][k], 1.0, 2.2))
        for t0, i in ((26.0, 0), (34.5, 1), (47.0, 2)):                 # beads rolling off the raised pads
            p = PADS[i]
            ev.append((t0, p[0] * W + p[2] * H * 0.9, p[1] * H + 0.01 * H, 0.9, 1.8))
        kx, ky = self.koi_head(37.5)
        ev.append((37.5, kx, ky, 1.2, 2.6))
        ev = np.array(ev, np.float64)
        self.events = dict(t=ev[:, 0], x=ev[:, 1], y=ev[:, 2], s=ev[:, 3], life=ev[:, 4])

    def build_fx(self):
        rng, W, H = self.rng, self.W, self.H
        n = 380
        self.rain = dict(x=rng.uniform(0, W, n), ph=rng.random(n), sp=rng.uniform(1.6, 2.4, n))
        n = 500
        self.glitter = dict(rel=rng.random(n) ** 1.3, side=rng.uniform(-1, 1, n), ph=rng.uniform(0, 6.28, n),
                            w=rng.uniform(2.0, 6.0, n))
        # the rainbow and its reflection never change shape, so paint them once and fade them in and out
        layer = np.zeros((H, W, 3), np.float32)
        cx, cy, R0 = 0.16 * W, 0.84 * H, 0.62 * H
        for i, col in enumerate(BANDS):
            ang = rng.uniform(0.06, 0.94, 1500) * np.pi
            r = R0 - i * 0.0062 * H
            x, y = cx + r * np.cos(ang), cy - r * np.sin(ang)
            ok = y < self.shore_top(x) - 2
            glow(layer, x[ok], y[ok], np.array(col) * 0.14, 3.0)
            yr = 2 * self.yw - y
            okr = ok & (yr < H)
            glow(layer, x[okr] + self.ripple(yr[okr], 0.0, 0.0), yr[okr], np.array(col) * 0.06, 3.0)
        rows, cols = np.nonzero(layer.sum(2) > 0)
        self.bow_box = (rows.min(), rows.max() + 1, cols.min(), cols.max() + 1)
        self.bow = layer[self.bow_box[0]:self.bow_box[1], self.bow_box[2]:self.bow_box[3]].copy()
        n = 900                                                         # the koi
        s = rng.random(n) ** 0.9
        v = rng.uniform(-1, 1, n)
        patch = np.sin(11 * s + 2.5 * v + 1.0) + 0.5 * np.sin(23 * s - 4 * v) > 0.15
        col = np.where(patch[:, None], shades(rng, [(240, 112, 40), (226, 84, 34)], rng.random(n)),
                       shades(rng, [(244, 238, 228), (230, 226, 216)], rng.random(n)))
        self.koi = dict(s=s, v=v, col=col)
        self.flies = []                                                  # two dragonflies' flight plans
        for i in range(2):
            ts = [23.0 + 1.5 * i]
            while ts[-1] < 54:
                ts.append(ts[-1] + rng.uniform(1.0, 2.6))
            pts = np.c_[rng.uniform(0.08, 0.92, len(ts)) * W, rng.uniform(0.34, 0.66, len(ts)) * H]
            pts[0] = (-0.06 * W if i == 0 else 1.06 * W, 0.4 * H)
            pts[-1] = (1.08 * W if i == 0 else -0.08 * W, 0.3 * H)
            self.flies.append(dict(t=np.array(ts), p=pts, ph=rng.uniform(0, 6.28)))

    # ---- moving parts ------------------------------------------------------------------------
    def koi_head(self, t):
        p = (t - 30.0) / 16.0
        return -0.1 * self.W + 1.25 * self.W * p, 0.885 * self.H + 0.02 * self.H * math.sin(2 * math.pi * p)

    def fly_pos(self, F, t):
        ts, ps = F["t"], F["p"]
        if t <= ts[0]:
            return ps[0], ps[1] - ps[0]
        if t >= ts[-1]:
            return ps[-1], ps[-1] - ps[-2]
        k = int(np.searchsorted(ts, t, side="right")) - 1
        s = (t - ts[k]) / (ts[k + 1] - ts[k])
        e = float(ss(s, 0.0, 0.32))                                    # dash, then hover
        p = ps[k] + (ps[k + 1] - ps[k]) * e
        p = p + 0.003 * self.H * np.array([math.sin(13 * t + F["ph"]), math.cos(11 * t + F["ph"])]) * (1 - e * (1 - e) * 4)
        return p, ps[k + 1] - ps[k]

    # ---- drawing ------------------------------------------------------------------------------
    def render(self, t):
        W, H, yw = self.W, self.H, self.yw
        img = self.backdrop.at(t)
        rain = float(1 - ss(t, 12.0, 20.0))
        water = keyed(t, self.water_keys)

        sx, sy = 0.80 * W, H * (0.25 + 0.305 * float(ss(t, 18.0, 58.0)))
        vis = float(ss(t, 16.0, 23.0))
        warm = float(ss(t, 36.0, 56.0))
        if vis > 0.01:
            core = np.array([255, 248, 230]) * (1 - warm) + np.array([255, 150, 80]) * warm
            gcol = (np.array([120, 110, 90]) * (1 - warm) + np.array([170, 80, 30]) * warm) * vis
            self.sun.draw(img, sx, sy, core, gcol, self.sky_mask, self.glow_mask)

        # the water shivers: shift each row a little
        sh = np.rint(self.ripple(self.rows, t, rain)).astype(np.int64)
        img[yw:] = img[self.rows[:, None], np.clip(self.xs[None, :] + sh[:, None], 0, W - 1)]

        fp = self.flower_pose(t)
        cx, cy = fp["top"]
        ldir = unit(np.array([0.0, -1.0]) * (1 - vis) + unit((sx - cx, sy - cy)) * vis)
        light = (keyed(t, self.amb_keys), keyed(t, self.key_keys), ldir)

        dots, refl, glows = [], [], []
        self.draw_pads(dots, refl, glows, t, light)
        self.draw_flower(dots, refl, t, fp, light)
        self.draw_floaters(dots, t, light)
        self.draw_koi(dots, t, water)
        for x, y, col, rad, wl in refl:                                  # reflections, mirrored at each waterline
            yr = 2 * wl - y
            m = (yr > wl) & (yr > yw) & (yr < H)
            if m.any():
                c = np.asarray(col)[m] * 0.55 + water * 0.28
                dots.append((x[m] + self.ripple(yr[m], t, rain), yr[m], c, np.broadcast_to(rad, x.shape)[m], -1e8 + yr[m]))
        paint(img, dots)

        self.draw_water_light(img, t, rain, vis, warm, sx)
        self.draw_flies(img, t)
        lights(img, glows)
        return np.clip(img, 0, 255).astype(np.uint8)

    def draw_pads(self, dots, refl, glows, t, light):
        for P in self.pads:
            uf = float(ss(t, P["start"], P["start"] + 7.0))
            if uf < 0.01:
                continue
            raised = P["h"] > 0
            R0 = P["R"] * (0.45 + 0.55 * uf)
            R = R0 * (1 + 0.04 * np.sin(7 * P["th"]))                   # a gently wavy rim
            r = P["r"] * R
            x0, z0 = r * np.cos(P["th"]), r * np.sin(P["th"])
            x = x0 * (0.12 + 0.88 * uf)                                  # unrolling from both edges
            y = (0.30 * R * P["r"] ** 2 if raised else 0.05 * R * P["r"] ** 6) + (1 - uf) * 0.3 * R * (np.abs(x0) / R) ** 2
            h = P["h"] * float(ss(t, P["start"], P["start"] + 5.0))
            pcx, pcy = P["x"], P["wl"] - h
            tau = 0.40 if raised else 0.30
            X, Y, Z = self.tilt(x, y, z0, pcx, pcy, tau)
            col = lit(P["col"], X, Y, pcx, pcy, R0, 0.3, light)
            if raised:
                dots.append((X, Y, col, P["rad"], P["wl"] * 10 + Z))
                refl.append((X, Y, col, P["rad"], P["wl"]))
                s = np.linspace(0, 1, len(P["stalk"]))                   # the stalk it stands on
                sx, sy = pcx + 0.004 * self.H * np.sin(s * 3), P["wl"] - h * s
                sx = sx + np.where(np.arange(len(s)) % 2, 1.5, -1.5)
                scol = lit(P["stalk"], sx, sy, pcx, sy, 3.0, 0.3, light)
                dots.append((sx, sy, scol, 1.5, P["wl"] * 10 - 50))
                refl.append((sx, sy, scol, 1.5, P["wl"]))
            else:
                dots.append((X, Y, col, P["rad"], -1e6 + P["wl"]))
            if uf > 0.9:                                                 # beads of water that catch the light
                D = P["drops"]
                dr = D["r"] * R0
                xd, zd = dr * np.cos(D["th"]), dr * np.sin(D["th"])
                yd = (0.30 * R0 * D["r"] ** 2 if raised else 0.0) + 0.006 * self.H
                Xd, Yd, _ = self.tilt(xd, yd + 0 * xd, zd, pcx, pcy, tau)
                dots.append((Xd, Yd, np.array([214, 230, 244]) * light[0], 2.2, (P["wl"] * 10 + 500) if raised else (-1e6 + P["wl"] + 1)))
                tw = np.clip(np.sin(t * D["w"] + D["ph"]), 0, 1) ** 8
                glows.append((Xd - 1, Yd - 1, np.array([255, 255, 255]) * (0.25 + 0.8 * tw)[:, None], 2.2))

    def draw_flower(self, dots, refl, t, fp, light):
        if fp["grow"] < 0.03:
            return
        H = self.H
        cx, cy = fp["top"]
        tau, hs = fp["tilt"], fp["hs"]
        S = self.stalk                                                  # the stalk out of the water
        x, y, dx, dy = bezier(fp["P"], S["s"])
        nrm = np.hypot(dx, dy) + 1e-6
        w = 0.0055 * H
        X, Y = x - dy / nrm * S["lat"] * w, y + dx / nrm * S["lat"] * w
        col = lit(S["col"], X, Y, x, y, w, 0.4, light)
        m = Y < self.fwl
        dots.append((X[m], Y[m], col[m], 1.5, self.fwl * 10 - 40))
        refl.append((X[m], Y[m], col[m], 1.5, self.fwl))

        late = float(ss(t, 46.0, 58.0))
        pod = self.pod                                                  # the seed pod
        ps = hs * (1 + 0.2 * late)
        rt, hp = self.r0 * ps, 0.75 * self.r0 * ps
        a, r = pod["top_a"], pod["top_r"] * rt
        Xt, Yt, Zt = self.tilt(r * np.cos(a), hp + 0 * r, r * np.sin(a), cx, cy, tau)
        ct = mix(pod["top_col"], pod["old_top"], late)
        hgt = pod["side_h"]
        rs = (0.27 + 0.73 * hgt) * rt
        Xs, Ys, Zs = self.tilt(rs * np.cos(pod["side_a"]), hgt * hp, rs * np.sin(pod["side_a"]), cx, cy, tau)
        cs = mix(pod["side_col"], pod["old_side"], late)
        Xh, Yh, Zh = self.tilt(pod["hole_r"] * rt * 0.8 * np.cos(pod["hole_a"]), hp + 0.5 + 0 * pod["hole_r"],
                               pod["hole_r"] * rt * 0.8 * np.sin(pod["hole_a"]), cx, cy, tau)
        hole = np.array([236, 220, 120]) * (1 - late) + np.array([56, 46, 30]) * late
        Xp, Yp, Zp = np.r_[Xt, Xs, Xh], np.r_[Yt, Ys, Yh], np.r_[Zt, Zs, Zh + 1]
        cp = np.r_[ct, cs, np.tile(hole, (len(Xh), 1))]
        rp = np.r_[np.full(len(Xt), 1.5), np.full(len(Xs), 1.5), np.full(len(Xh), 2.4)]
        cp = lit(cp, Xp, Yp, cx, cy - hp, rt, 0.35, light)
        m = Yp < self.fwl
        dots.append((Xp[m], Yp[m], cp[m], rp[m], self.fwl * 10 + Zp[m]))
        refl.append((Xp[m], Yp[m], cp[m], rp[m], self.fwl))

        o_all = float(ss(t, 24.0, 31.0))
        St = self.stamens                                               # a ring of golden stamens
        alive = St["rn"] > ss(t, 47.0, 53.0)
        if o_all > 0.05 and alive.any():
            wth = float(ss(t, 45.0, 51.0))
            n = int(alive.sum())
            th, ln = St["th"][alive], St["ln"][alive] * 0.03 * H * hs * (0.5 + 0.5 * o_all)
            al = 0.35 + 0.75 * o_all + 0.6 * wth
            f = np.array([0.3, 0.55, 0.8, 1.0])
            rr = 0.95 * rt + (ln[:, None] * f) * math.sin(al)
            yy = 0.15 * hp + (ln[:, None] * f) * math.cos(al)
            Xs_, Ys_, Zs_ = self.tilt(rr * np.cos(th)[:, None], yy, rr * np.sin(th)[:, None], cx, cy, tau)
            fil = np.array([250, 236, 190]) * (1 - wth) + np.array([150, 110, 60]) * wth
            col = np.concatenate([np.tile(fil, (n * 3, 1)).reshape(n, 3, 3),
                                  (St["anther"][alive] * (1 - wth) + np.array([140, 100, 50]) * wth)[:, None, :]], 1)
            col = lit(col.reshape(-1, 3), Xs_.ravel(), Ys_.ravel(), cx, cy, rt * 2, 0.3, light)
            rad = np.tile([1.0, 1.0, 1.0, 1.6], n)
            dots.append((Xs_.ravel(), Ys_.ravel(), col, rad, self.fwl * 10 + Zs_.ravel() + 0.5))
            refl.append((Xs_.ravel(), Ys_.ravel(), col, rad, self.fwl))

        P = self.petals                                                 # the petals still attached
        keep = t < self.petal_fall[P["k"]]
        if keep.any():
            g = {key: val[keep] for key, val in P.items()}
            x, y, z, o = self.petal_points(g, t, fp)
            X, Y, Z = self.tilt(x, y, z, cx, cy, tau)
            col = g["live"] * (0.78 + 0.22 * np.clip(Z / (self.Lp * hs) * 0.5 + 0.5, 0, 1))[:, None]
            col = mix(col, np.array([150, 170, 110], np.float32), 0.45 * (1 - o) ** 2 * (1 - g["u"]))
            col = lit(col, X, Y, cx, cy - 0.4 * self.Lp, self.Lp * hs, 0.35, light, wash=0.15)
            m = Y < self.fwl
            dots.append((X[m], Y[m], col[m], g["rad"][m], self.fwl * 10 + Z[m]))
            refl.append((X[m], Y[m], col[m], g["rad"][m], self.fwl))

    def draw_floaters(self, dots, t, light):
        """Petals falling, then floating away on the water."""
        F, P = self.falls, self.petals
        f = t - F["T"][P["k"]]
        m = f > 0
        if not m.any():
            return
        k, f, u, v = P["k"][m], f[m], P["u"][m], P["v"][m]
        dur = F["dur"][k]
        e = np.clip(f / dur, 0, 1)
        air = f < dur
        x = F["cx"][k] + (F["lx"][k] - F["cx"][k]) * e + 0.02 * self.W * np.sin(3 * f + F["ph"][k]) * air
        y = F["cy"][k] + (F["ly"][k] - F["cy"][k]) * e ** 2
        x = x + F["vx"][k] * np.maximum(f - dur, 0)
        y = y + 0.002 * self.H * np.sin(0.8 * t + F["ph"][k]) * ~air
        a = F["a0"][k] + np.where(air, 2.0 * f, 2.0 * dur + F["spin"][k] * (f - dur))
        flat = np.where(air, 0.3 + 0.7 * np.abs(np.cos(3 * f + F["ph"][k])), 0.32)
        L = self.Lp * P["ln"][m] * 0.95
        Wd = L * P["wd"][m] * petal_width(u) * 0.5
        uu, vv = (u - 0.5) * L, v * Wd
        X = x + np.cos(a) * uu - np.sin(a) * vv
        Y = y + (np.sin(a) * uu + np.cos(a) * vv) * flat
        col = lit(P["live"][m] * np.where(air, 1.0, 0.9)[:, None], X, Y, x, y, L, 0.3, light, wash=0.2)
        dots.append((X, Y, col, P["rad"][m], np.where(air, 1e5, -1e6 + F["ly"][k] + 0.5)))

    def draw_koi(self, dots, t, water):
        if not 30.0 < t < 46.0:
            return
        H = self.H
        hx, hy = self.koi_head(t)
        K = self.koi
        s, v = K["s"], K["v"]
        Lk = 0.11 * H
        sway = 0.014 * H * np.sin(9 * s - 6.0 * t) * s ** 1.2         # the body swings more towards the tail
        rise = float(ss(t, 36.5, 37.5) * (1 - ss(t, 38.0, 39.5)))     # comes up to the surface for a moment
        sx = hx - s * Lk
        sy = hy + sway * 0.6
        w = 0.026 * H * np.sin(np.pi * np.clip(0.06 + 0.9 * (1 - s), 0, 1) ** 0.6) * (1 - 0.65 * s)
        w = np.where(s > 0.86, 0.024 * H * (s - 0.86) / 0.14, w)          # the tail fin fans out
        X = sx + 0.12 * w * v
        Y = sy + w * v * 0.55
        col = K["col"] * (0.55 + 0.3 * rise) + water * (0.45 - 0.3 * rise)
        dots.append((X, Y, col, 2.0, -3e8 + Y))

    def draw_water_light(self, img, t, rain, vis, warm, sx):
        W, H, yw = self.W, self.H, self.yw
        E = self.events                                                 # rings on the water
        age = t - E["t"]
        m = (age >= 0) & (age < E["life"])
        if m.any():
            a = np.linspace(0, 2 * np.pi, 28, endpoint=False)
            depth = 0.35 + 0.65 * (E["y"][m] - yw) / (H - yw)
            ag = age[m]
            b = E["s"][m] * (1 - ag / E["life"][m]) ** 1.5
            for scale, bb in ((1.0, 1.0), (0.6, 0.6)):
                r = (2 + 45 * ag) * depth * E["s"][m] * W / 1000 * scale
                x = E["x"][m][:, None] + r[:, None] * np.cos(a)
                y = E["y"][m][:, None] + r[:, None] * 0.28 * np.sin(a)
                glow(img, x.ravel(), y.ravel(), np.array([230, 236, 244]) * 0.5 * np.repeat(b * bb, 28)[:, None], 1.2)
        if rain > 0.01:                                                 # rain streaks
            R = self.rain
            y = ((R["ph"] + t * R["sp"]) % 1.15 - 0.1) * H
            x = (R["x"] + 0.12 * y) % W
            k = np.arange(6)
            xs = (x[:, None] - k * 6 * 0.12).ravel()
            ys = (y[:, None] - k * 6.0).ravel()
            glow(img, xs, ys, np.array([210, 218, 232]) * (0.55 * rain * np.tile(1 - k / 6, len(x)))[:, None], 1.0)
        if vis > 0.01:                                                  # the sun's path across the water
            G = self.glitter
            y = yw + 2 + G["rel"] * (H - yw - 2)
            x = sx + G["side"] * (0.008 * W + 0.09 * W * G["rel"])
            tw = np.clip(np.sin(t * G["w"] + G["ph"]), 0, 1) ** 6
            col = np.array([255, 236, 190]) * (1 - warm) + np.array([255, 170, 90]) * warm
            glow(img, x, y, col * (vis * (0.12 + 0.9 * tw) * (1 - 0.5 * np.abs(G["side"])))[:, None], 1.3)
        rb = float(ss(t, 19.0, 24.0) * (1 - ss(t, 32.0, 38.0)))         # the rainbow, and its reflection
        if rb > 0.01:
            y0, y1, x0, x1 = self.bow_box
            img[y0:y1, x0:x1] += self.bow * np.float32(rb)

    def draw_flies(self, img, t):
        if not 22.0 < t < 56.0:
            return
        H = self.H
        for F in self.flies:
            p, d = self.fly_pos(F, t)
            hd = unit(d + 1e-6)
            pv = np.array([-hd[1], hd[0]])
            seg = 0.0032 * H
            k = np.arange(-1, 11)                                        # head, thorax, long abdomen
            bx, by = p[0] - hd[0] * k * seg, p[1] - hd[1] * k * seg
            col = np.where((k <= 0)[:, None], [[30, 90, 118]], np.where((k % 3 == 0)[:, None], [[20, 60, 80]], [[60, 172, 214]]))
            rad = np.where(k <= 0, 2.6, 1.5)
            stamp(img, bx, by, col.astype(np.float32), rad)
            flick = 0.6 + 0.4 * math.sin(90 * t)
            j = np.arange(1, 9) / 8
            for ang, ln in ((1.25, 0.024), (-1.25, 0.024), (1.85, 0.021), (-1.85, 0.021)):
                wd = hd * math.cos(ang) + pv * math.sin(ang)
                glow(img, p[0] - hd[0] * seg + wd[0] * ln * H * j, p[1] - hd[1] * seg + wd[1] * ln * H * j,
                     np.array([200, 215, 235]) * 0.28 * flick, 1.6)


if __name__ == "__main__":
    run(Lotus, "Lotus - summer rain on the pond", "lotus", T_END, STAGES)
