"""
fireflies.py - a meadow at dusk filling with fireflies, in dots.

The sun has just gone. The sky still burns orange behind a line of trees and a lone oak,
and the tall grass sways dark against it. One by one fireflies wake and blink in the grass.
Through the blue hour there are more and more, and their flashes begin to fall into step:
waves of light roll across the meadow until the whole field flashes as one. The moon comes
up through the trees, a fox trots through the grass - pausing to look straight at you, eyes
shining - and the fireflies lift out of its way.

    python fireflies.py            interactive window (Space pause, R restart, S save frame, Esc quit)
    python fireflies.py --png      save five stages
    python fireflies.py --gif      save the whole evening as a GIF
"""
import math

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))   # the shared engine: engine/pointillism.py

from pointillism import (Backdrop, Haze, Moon, bezier, glow, keyed, mix, paint, run, sample_v, shades, sky, ss,
                         stamp, stars, warm_glow)

T_END = 60.0
STAGES = [5.0, 18.0, 31.4, 43.0, 55.75]
TAU = 2 * math.pi

GRASS = [(34, 58, 30), (46, 74, 36), (60, 90, 44), (80, 104, 52)]
DRY = [(104, 98, 60), (124, 114, 70), (140, 128, 80)]
SEED = [(150, 140, 100), (170, 160, 118), (190, 180, 136)]
LACE = [(232, 228, 214), (246, 244, 232), (220, 216, 196)]
LEAF = [(20, 30, 26), (26, 38, 30), (32, 46, 34)]
BARK = [(40, 34, 32), (52, 44, 40), (62, 52, 46)]
FIRE = np.array([200, 255, 110], np.float32)
HALO = np.array([64, 112, 18], np.float32)

LOOKS = [  # the evening, as the keys the backdrop cross-fades between
    dict(t=0.0, sky=[(0.0, (40, 46, 100)), (0.25, (92, 80, 140)), (0.45, (196, 116, 120)), (0.60, (252, 160, 84))],
         glow=(140, 70, 20), trees=(40, 34, 52), meadow=((92, 92, 70), (40, 50, 34)), stars=0.0, mist=0.0,
         amb=(0.62, 0.52, 0.56), key=(150, 80, 40), ldir=(-1.0, -0.15)),
    dict(t=13.0, sky=[(0.0, (22, 28, 74)), (0.25, (48, 56, 116)), (0.45, (116, 96, 140)), (0.60, (200, 128, 112))],
         glow=(70, 34, 12), trees=(26, 26, 44), meadow=((62, 68, 70), (28, 38, 32)), stars=0.25, mist=0.2,
         amb=(0.42, 0.42, 0.56), key=(80, 56, 50), ldir=(-1.0, -0.2)),
    dict(t=27.0, sky=[(0.0, (10, 14, 40)), (0.25, (18, 26, 62)), (0.45, (34, 44, 88)), (0.60, (62, 64, 104))],
         glow=(10, 6, 4), trees=(14, 16, 30), meadow=((36, 44, 56), (16, 24, 24)), stars=0.8, mist=0.5,
         amb=(0.26, 0.29, 0.42), key=(30, 34, 52), ldir=(-0.4, -1.0)),
    dict(t=42.0, sky=[(0.0, (8, 12, 34)), (0.25, (16, 24, 58)), (0.45, (32, 44, 86)), (0.60, (54, 62, 100))],
         glow=(0, 0, 0), trees=(12, 16, 30), meadow=((40, 50, 64), (16, 26, 28)), stars=1.0, mist=0.8,
         amb=(0.26, 0.30, 0.44), key=(52, 60, 86), ldir=(-0.3, -1.0)),
    dict(t=60.0, sky=[(0.0, (10, 16, 40)), (0.25, (20, 30, 66)), (0.45, (38, 52, 94)), (0.60, (60, 72, 108))],
         glow=(0, 0, 0), trees=(14, 18, 34), meadow=((46, 58, 72), (18, 30, 32)), stars=1.0, mist=0.9,
         amb=(0.30, 0.34, 0.48), key=(70, 80, 110), ldir=(-0.1, -1.0)),
]


def flashes(t, tau, period, count, gap=0.5):
    """Brightness of a firefly that flashes `count` times, `gap` seconds apart, every `period`."""
    u = np.mod(t - tau, period)
    k = np.floor(u / gap)
    q = u - k * gap
    p = np.where(q < 0.06, q / 0.06, np.exp(-(q - 0.06) / 0.11))
    return p * (k < count)


class Fireflies:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.rng = np.random.default_rng(7)
        self.S = min(W, H)
        self.amb_keys = [(L["t"], L["amb"]) for L in LOOKS]
        self.key_keys = [(L["t"], L["key"]) for L in LOOKS]
        self.dir_keys = [(L["t"], L["ldir"]) for L in LOOKS]
        self.moon = Moon(np.random.default_rng(3), 0.030 * H, 3.2)
        self.haze = Haze(W, H, max(2, round(5 * min(W, H) / 1000)))
        self.build_treeline()
        self.build_oak()
        self.build_backdrop()
        self.build_grass()
        self.build_flies()
        self.build_fox()

    # ---- the far side: sky, trees, the oak ------------------------------------------------------
    def build_treeline(self):
        W, H = self.W, self.H
        rng = np.random.default_rng(31)
        u = np.arange(W) / W
        base = 0.588 - 0.016 * np.sin(u * TAU * 1.3 + 0.4) - 0.010 * np.sin(u * TAU * 3.1 + 1.1)
        top = np.zeros(W)
        for _ in range(90):                                    # round crowns
            c, w, h = rng.uniform(-0.05, 1.05), rng.uniform(0.010, 0.034), rng.uniform(0.006, 0.034)
            top = np.maximum(top, h * np.sqrt(np.clip(1 - ((u - c) / w) ** 2, 0, 1)))
        for _ in range(22):                                    # a few pointed firs
            c, w, h = rng.uniform(0, 1), rng.uniform(0.006, 0.012), rng.uniform(0.03, 0.062)
            top = np.maximum(top, h * np.clip(1 - np.abs(u - c) / w, 0, 1) ** 1.2)
        self.ttop = (H * (base - top)).astype(np.float32)

    def build_oak(self):
        W, H, rng = self.W, self.H, np.random.default_rng(41)
        S = self.S
        xs, ys, rads, rim, kind = [], [], [], [], []
        bx, by, fx, fy = 0.80 * W, 0.662 * H, 0.796 * W, 0.535 * H
        n = 2600                                                # trunk with a flared foot
        s, v = rng.random(n), rng.uniform(-1, 1, n)
        wid = 0.019 * S * (1 + 0.9 * np.exp(-s * 9)) * (1 - 0.2 * s)
        xs.append(bx + (fx - bx) * s + v * wid)
        ys.append(by + (fy - by) * s)
        rads.append(rng.choice([1.0, 1.5, 2.0], n))
        rim.append(np.clip(-v, 0, 1) ** 2)
        kind.append(np.zeros(n))
        clumps = []
        for i in range(46):                                     # a broad crown, its lower limbs drooping
            a = rng.uniform(0, TAU)
            r = rng.random() ** 0.45
            cx = 0.79 * W + 0.19 * W * r * math.cos(a)
            cy = 0.405 * H + 0.105 * H * r * math.sin(a) + 0.05 * H * (r * math.cos(a)) ** 4
            clumps.append((cx, cy, rng.uniform(0.020, 0.040) * S))
        for cx, cy, cr in clumps[::2]:                          # limbs out to the clumps
            P = [(fx, fy), (fx + (cx - fx) * 0.3, fy - 0.03 * H), (cx - (cx - fx) * 0.25, cy + 0.02 * H), (cx, cy)]
            m = 420
            s = rng.random(m)
            x, y, dx, dy = bezier(P, s)
            nn = np.hypot(dx, dy) + 1e-6
            v = rng.uniform(-1, 1, m)
            wd = 0.0075 * S * (1 - 0.7 * s)
            xs.append(x - dy / nn * v * wd)
            ys.append(y + dx / nn * v * wd)
            rads.append(rng.choice([1.0, 1.5], m))
            rim.append(np.clip(-v, 0, 1) ** 2 * 0.6)
            kind.append(np.zeros(m))
        for cx, cy, cr in clumps:                               # leafy masses, airy at the edges
            m = int(3600 * (cr / (0.04 * S)) ** 2)
            a = rng.uniform(0, TAU, m)
            edge = 1 + 0.22 * np.sin(5 * a + cx) + 0.14 * np.sin(11 * a + cy)   # ragged outline
            r = rng.random(m) ** 0.55 * edge
            x = cx + np.cos(a) * r * cr * 1.2
            y = cy + np.sin(a) * r * cr * 0.8
            sub = rng.random(m) < 0.18                            # little sprays poking out
            x[sub] += rng.normal(0, 0.35, int(sub.sum())) * cr
            y[sub] += rng.normal(0, 0.25, int(sub.sum())) * cr
            xs.append(x)
            ys.append(y)
            rads.append(rng.choice([1.0, 1.5, 2.0], m, p=[0.3, 0.45, 0.25]))
            rim.append(np.clip((-(x - cx) * 0.8 - (y - cy)) / cr, 0, 1.4))
            kind.append(np.ones(m))
        self.oak = dict(x=np.concatenate(xs), y=np.concatenate(ys), rad=np.concatenate(rads),
                        rim=np.concatenate(rim), kind=np.concatenate(kind))
        self.oak["tone"] = rng.random(len(self.oak["x"]))
        self.oak_crown = (0.79 * W, 0.405 * H, 0.2 * W, 0.15 * H)

    def build_backdrop(self):
        self.backdrop = Backdrop([L["t"] for L in LOOKS], [self.background(L) for L in LOOKS])
        W, H = self.W, self.H
        mask = np.zeros((H, W, 3), np.float32)
        O = self.oak
        stamp(mask, O["x"], O["y"], np.ones(3, np.float32), O["rad"])
        yy = np.arange(H, dtype=np.float32)[:, None]
        self.sky_mask = ((yy < self.ttop[None, :]) & (mask[..., 0] < 0.5)).astype(np.float32)
        self.glow_mask = np.clip(self.sky_mask + 0.25, 0, 1)

    def background(self, L):
        W, H = self.W, self.H
        rng = np.random.default_rng(1234)                         # the same dots in every look
        img = sky(W, H, rng, L["sky"], 0.63 * H)
        warm_glow(img, 0.22 * W, 0.60 * H, 0.32 * W, L["glow"], 0.42)
        stars(img, rng, int(1500 * W * H / 1e6), 0.55 * H, L["stars"])
        amb = np.asarray(L["amb"], np.float32)
        tcol = np.asarray(L["trees"], np.float32)                 # the treeline, leafy along its top
        yy = np.arange(H, dtype=np.float32)[:, None]
        fill = (yy >= self.ttop[None, :]) & (yy < 0.66 * H)
        img[fill] = tcol * 0.9
        n = int(W * 0.08 * H * 0.6)
        x = rng.uniform(0, W, n)
        y = np.interp(x, np.arange(W), self.ttop) + rng.random(n) ** 1.6 * 0.08 * H - rng.random(n) * 3
        stamp(img, x, y, tcol * (0.85 + 0.35 * rng.random((n, 1))), rng.choice([1.0, 1.5, 2.0], n))
        far, near = (np.asarray(c, np.float32) for c in L["meadow"])   # the far meadow, in short strokes
        mtop = H * (0.612 + 0.005 * np.sin(np.arange(W) / W * TAU * 2.2))
        img[yy >= mtop[None, :]] = far * 0.8
        for row in range(2):
            n = int(W * H * (0.12 if row == 0 else 0.06))
            x = rng.uniform(0, W, n)
            d = rng.random(n) ** (0.8 if row == 0 else 1.0)
            y0 = np.interp(x, np.arange(W), mtop) + d * (1.0 * H - np.interp(x, np.arange(W), mtop))
            ln = (1.5 + 9 * d ** 1.8) * H / 1000 * rng.uniform(0.6, 1.4, n)
            k = rng.random(n)
            col = (far + (near - far) * (d ** 0.7)[:, None]) * (0.85 + 0.35 * rng.random((n, 1)))
            lean = rng.normal(0, 0.25, n)
            stamp(img, x + lean * ln * k, y0 - ln * k, col * (0.9 + 0.3 * k[:, None]), 1.0 + 0.8 * d)
        if L["mist"] > 0:                                          # a little ground mist under the trees
            n = int(W * H * 0.05)
            x, y = rng.uniform(0, W, n), H * (0.600 + 0.10 * rng.random(n) ** 1.5)
            fade = np.sin(np.pi * (y / H - 0.600) / 0.10) ** 2
            glow(img, x, y, np.array([50, 56, 76]) * L["mist"] * 0.035 * fade[:, None], 1.5)
        else:
            rng.uniform(0, 1, int(W * H * 0.05) * 2)
        O = self.oak                                               # the oak, dark against the sky
        key = np.asarray(L["key"], np.float32)
        base = np.where(O["kind"][:, None] > 0, shades(rng, LEAF, O["tone"], 0.6), shades(rng, BARK, O["tone"], 0.6))
        col = base * amb + key * 0.22 * O["rim"][:, None]
        stamp(img, O["x"], O["y"], col, O["rad"])
        return img

    # ---- grass, seed heads, wild carrot ------------------------------------------------------------
    def build_grass(self):
        rng, W, H = self.rng, self.W, self.H
        n = 620
        d = rng.random(n) ** 1.45                                 # 0 far .. 1 near
        bx = rng.uniform(-0.03, 1.03, n) * W
        by = H * (0.63 + 0.44 * d)
        h = H * (0.022 + 0.36 * d ** 2.3) * rng.uniform(0.55, 1.25, n)
        rad0 = np.clip((1.0 + 3.6 * d ** 1.6) * H / 1000, 0.8, 2.6)
        dry = rng.random(n) < 0.22
        col = np.where(dry[:, None], shades(rng, DRY, rng.random(n), 0.6), shades(rng, GRASS, d, 0.9))
        B = dict(bx=bx, by=by, h=h, d=d, lean=rng.normal(0, 0.12, n), bend=rng.normal(0, 0.10, n),
                 ph=rng.uniform(0, TAU, n), ph2=rng.uniform(0, TAU, n), amp=rng.uniform(0.6, 1.4, n),
                 rim=rng.uniform(0.2, 1.0, n), col=col)
        idx, s, rad, lat, tone, dcol = [], [], [], [], [], []
        for i in range(n):                                        # the blades: one line of dots each
            m = max(4, int(h[i] / (0.9 + 0.75 * rad0[i])))
            si = np.linspace(0, 1, m) + rng.uniform(0, 0.5 / m, m)
            idx.append(np.full(m, i))
            s.append(np.clip(si, 0, 1))
            rad.append(np.maximum(rad0[i] * (1 - 0.72 * si), 0.8))
            lat.append(np.zeros(m))
            tone.append(np.ones(m))
            dcol.append(np.broadcast_to(col[i], (m, 3)))
        tall = np.nonzero((h > 0.07 * H) & (rng.random(n) < 0.45))[0]
        for i in tall:                                            # timothy spikes and oat-like sprays
            px = H / 1000
            if rng.random() < 0.6:                                # a slim fuzzy spike
                m = int(16 + 30 * d[i])
                si = rng.uniform(0.87, 1.0, m)
                idx.append(np.full(m, i))
                s.append(si)
                rad.append(np.full(m, max(0.8, (0.8 + 0.9 * d[i]) * px * 1.4)))
                lat.append(rng.uniform(-1, 1, m) * (0.8 + 2.2 * d[i]) * px)
                tone.append(np.full(m, 1.25))
                dcol.append(shades(rng, SEED, rng.random(m), 0.7))
            else:                                                  # little spikelets dangling off the tip
                k, q = int(rng.integers(4, 8)), 4
                m = k * q
                sp = np.repeat(rng.uniform(0.78, 1.0, k), q)
                side = np.repeat(rng.choice([-1, 1], k) * rng.uniform(0.4, 1.0, k), q)
                step = np.tile(np.arange(1, q + 1), k) * np.repeat(rng.uniform(0.6, 1.3, k), q)
                ln = (1.6 + 2.6 * d[i]) * px
                idx.append(np.full(m, i))
                s.append(sp)
                rad.append(np.full(m, max(0.8, (0.7 + 0.8 * d[i]) * px * 1.4)))
                lat.append(side * step * ln * 0.7)
                tone.append(np.full(m, 1.3))
                dcol.append(shades(rng, SEED, rng.random(m), 0.7))
        self.blades = B
        self.gd = dict(i=np.concatenate(idx), s=np.concatenate(s), rad=np.concatenate(rad),
                       lat=np.concatenate(lat), tone=np.concatenate(tone), col=np.concatenate(dcol).astype(np.float32))
        g = self.gd
        g["droop"] = np.where(g["tone"] > 1.27, np.abs(g["lat"]) * 1.3, 0.0)       # spikelets hang down
        # wild carrot: straight stems topped with flat white lace
        k = 9
        lx = rng.uniform(0.05, 0.95, k) * W
        ld = rng.uniform(0.35, 0.95, k)
        ly = H * (0.63 + 0.44 * ld)
        lh = H * (0.05 + 0.22 * ld ** 1.5)
        florets = []
        for j in range(k):
            cs = rng.uniform(-1, 1, (22, 2))
            cs = cs[np.hypot(cs[:, 0], cs[:, 1]) < 1][:14]
            m = 11
            cx = np.repeat(cs[:, 0], m) + rng.normal(0, 0.09, len(cs) * m)
            cz = np.repeat(cs[:, 1], m) + rng.normal(0, 0.09, len(cs) * m)
            florets.append((np.full(len(cx), j), cx, cz))
        self.lace = dict(x=lx, y=ly, h=lh, d=ld, ph=rng.uniform(0, TAU, k), a=0.020 * H * (0.5 + ld),
                         b=0.007 * H * (0.5 + ld))
        j, cx, cz = (np.concatenate(a) for a in zip(*florets))
        self.lace_f = dict(j=j.astype(int), u=cx, v=cz, col=shades(rng, LACE, rng.random(len(j)), 0.8))

    def sway(self, x, ph, ph2, amp, t):
        W = self.W
        gust = 0.5 + 0.5 * np.sin(0.55 * t - 2.4 * x / W)
        return amp * (0.035 * np.sin(1.3 * t + 3.0 * x / W + ph) + 0.02 * np.sin(2.1 * t + ph2) + 0.05 * gust ** 3)

    def draw_grass(self, dots, t, amb, key):
        B, g = self.blades, self.gd
        sw = self.sway(B["bx"], B["ph"], B["ph2"], B["amp"], t)
        i, s = g["i"], g["s"]
        h, cur = B["h"][i], B["bend"][i] + sw[i]
        x = B["bx"][i] + h * (B["lean"][i] * s + cur * s * s) + g["lat"]
        y = B["by"][i] - h * s * (1 - 0.25 * np.abs(cur) * s) + g["droop"]
        rimk = B["rim"][i] * s ** 0.8 * g["tone"]
        col = g["col"] * amb + key * (0.45 * rimk)[:, None]
        dots.append((x, y, col, g["rad"], B["by"][i]))
        La, f = self.lace, self.lace_f            # wild carrot
        lsw = self.sway(La["x"], La["ph"], La["ph"] * 1.7, 0.6, t)
        for j in range(len(La["x"])):
            m = 60
            s = np.linspace(0, 1, m)
            x = La["x"][j] + La["h"][j] * lsw[j] * s * s
            y = La["y"][j] - La["h"][j] * s
            dots.append((x, y, np.array(GRASS[1], np.float32) * amb, 0.8 + 1.0 * La["d"][j], La["y"][j]))
        j = f["j"]
        tx = La["x"][j] + La["h"][j] * lsw[j]
        ty = La["y"][j] - La["h"][j]
        x = tx + f["u"] * La["a"][j]
        y = ty - La["b"][j] * 0.9 * (1 - (f["u"] ** 2 + f["v"] ** 2)) + f["v"] * La["b"][j] * 0.6
        col = f["col"] * amb * 0.95 + key * 0.55
        dots.append((x, y, col, np.maximum(0.8, (0.8 + 1.0 * La["d"][j]) * self.S / 1000 * 1.3), La["y"][j] + 0.01))

    # ---- fireflies ----------------------------------------------------------------------------------
    def build_flies(self):
        rng, W, H = self.rng, self.W, self.H
        n = 440
        where = rng.choice(3, n, p=[0.72, 0.14, 0.14])            # grass, the oak, the air above
        hy = np.where(where == 0, H * (0.60 + 0.38 * rng.random(n) ** 1.2),
                      np.where(where == 1, H * (0.31 + 0.19 * rng.random(n)), H * (0.47 + 0.14 * rng.random(n))))
        hx = np.where(where == 1, W * (0.64 + 0.30 * rng.random(n)), rng.uniform(-0.02, 1.02, n) * W)
        near = np.clip((hy / H - 0.58) / 0.40, 0, 1)
        near = np.where(where == 1, 0.25, near)
        F = dict(hx=hx, hy=hy, near=near, wake=2.0 + 30.0 * rng.random(n) ** 1.4 + 8.0 * (where == 1),
                 p=rng.uniform(0, TAU, (4, n)), w=rng.uniform(0.12, 0.30, (2, n)),
                 period=rng.uniform(2.2, 4.6, n), count=rng.choice([1, 2, 2, 3], n), tau=rng.uniform(0, 6, n),
                 jit=rng.normal(0, 0.12, n), size=0.6 + 1.1 * near * rng.uniform(0.8, 1.2, n))
        self.flies = F

    def draw_flies(self, img, t, fox):
        F, W, H = self.flies, self.W, self.H
        awake = ss(t, F["wake"], F["wake"] + 2.0)
        x = F["hx"] + 0.030 * W * np.sin(F["w"][0] * t + F["p"][0]) + 0.008 * W * np.sin(0.9 * t + F["p"][1])
        y = F["hy"] + 0.018 * H * np.sin(F["w"][1] * t + F["p"][2]) + 0.005 * H * np.sin(1.4 * t + F["p"][3])
        sync, one = float(ss(t, 22.0, 40.0)), float(ss(t, 46.0, 52.0))
        b = (1 - sync) * flashes(t, F["tau"], F["period"], F["count"])
        if sync > 0:                      # in step: a wave rolls left to right, then all at once
            delay = (1 - one) * (1.7 * F["hx"] / W + 0.4 * F["hy"] / H) + F["jit"] * (1 - 0.6 * one)
            b = b + sync * flashes(t, 0.6 + delay, 5.0, 5)
        if fox is not None:               # the fox: fireflies lift out of its way and flare
            fx, fy, L = fox
            dx, dy = x - fx, y - (fy - 0.5 * L)
            e = np.exp(-(dx * dx + dy * dy) / (1.6 * L) ** 2)
            x = x + np.sign(dx) * 0.35 * L * e
            y = y - 0.9 * L * e
            b = np.maximum(b, e * (0.55 + 0.45 * np.sin(23 * t + F["p"][0])))
        b = b * awake
        on = b > 0.03
        if not on.any():
            return
        x, y, b, sz = x[on], y[on], b[on], F["size"][on]
        self.haze.add(x, y, HALO * (b ** 1.4 * sz)[:, None], (8 + 11 * sz) * self.S / 1000)
        self.haze.apply(img, 1.0)
        glow(img, x, y, FIRE * (b * (0.55 + 0.45 * sz))[:, None], 0.9 + 1.0 * sz * self.S / 1000)
        stamp(img, x[b > 0.5], y[b > 0.5], np.array([250, 255, 220], np.float32), 0.8 + 0.5 * (sz[b > 0.5] > 1.0))

    # ---- the fox -------------------------------------------------------------------------------------
    def build_fox(self):
        rng = np.random.default_rng(77)
        parts = []

        def ellipse(n, cx, cy, a, b, col, rot=0.0, part=0):
            r, ang = np.sqrt(rng.random(n)), rng.uniform(0, TAU, n)
            u, v = r * np.cos(ang), r * np.sin(ang)
            c, s = math.cos(rot), math.sin(rot)
            x, y = cx + a * u * c - b * v * s, cy + a * u * s + b * v * c
            parts.append((x, y, col(u, v), np.clip(-v, 0, 1) ** 1.5, np.full(n, part)))

        russet = lambda u, v: mix(shades(rng, [(150, 70, 32), (172, 84, 38), (188, 100, 46)], rng.random(len(u)), 0.7),
                                  np.array([196, 176, 150], np.float32), ss(v, 0.35, 0.9))
        ellipse(760, 0.02, -0.57, 0.40, 0.15, russet, -0.05)                 # body
        ellipse(320, 0.30, -0.58, 0.14, 0.15, russet)                        # haunch
        ellipse(460, -0.36, -0.62, 0.17, 0.19, lambda u, v: mix(
            shades(rng, [(160, 76, 34), (180, 90, 42)], rng.random(len(u))), np.array([230, 222, 206], np.float32),
            ss(v + 0.6 * u, 0.1, 0.6)), 0.5)                                 # neck and white bib
        ellipse(300, -0.63, -0.84, 0.12, 0.10, russet, 0.0, 1)                # head
        n = 260                                                              # snout
        u = rng.random(n)
        v = rng.uniform(-1, 1, n)
        sx = -0.70 - 0.22 * u
        sy = -0.85 + 0.04 * u + v * 0.055 * (1 - u) ** 0.8
        col = mix(shades(rng, [(176, 86, 40), (190, 100, 48)], rng.random(n)), np.array([232, 224, 210], np.float32),
                  ss(v, 0.0, 0.5))
        col[u > 0.93] = (24, 20, 20)
        parts.append((sx, sy, col, np.clip(-v, 0, 1), np.full(n, 2)))
        for ex in (-0.57, -0.67):                                           # ears, black-backed
            n = 120
            u, v = rng.random(n), rng.uniform(-1, 1, n)
            x = ex + 0.05 * v * (1 - u) + 0.03 * u
            y = -0.90 - 0.19 * u
            col = np.where((u > 0.5)[:, None], np.array([30, 24, 22], np.float32), np.array([168, 80, 36], np.float32))
            parts.append((x, y, col.astype(np.float32), np.ones(n) * 0.7, np.full(n, 1)))
        n = 900                                                              # the brush
        u, v = rng.random(n), sample_v(rng, n, 0.2)
        tw = 0.03 + 0.085 * np.sin(np.pi * np.clip(0.08 + 0.95 * u, 0, 1)) ** 0.8 * ss(u, 0.0, 0.35)
        col = shades(rng, [(150, 70, 32), (170, 84, 40), (120, 60, 30)], rng.random(n), 0.7)
        col[u > 0.84] = shades(rng, [(226, 220, 206), (240, 236, 224)], rng.random(int((u > 0.84).sum())))
        parts.append((u, v * tw, col, np.clip(-v, 0, 1), np.full(n, 3)))
        x, y, col, rim, part = (np.concatenate(a) for a in zip(*parts))
        self.fox = dict(x=x, y=y, col=col.astype(np.float32), rim=rim, part=part.astype(int))

    def fox_state(self, t):
        """Where the fox is, how far it has walked, how much it has turned its head to us."""
        a, p0, p1, b = 35.0, 41.0, 44.6, 51.5
        x0, xm, x1 = 1.10, 0.56, -0.12
        if t < a or t > b:
            return None
        if t < p0:
            f = (t - a) / (p0 - a)
            x = x0 + (xm - x0) * (f * (1.6 - 0.6 * f))
        elif t < p1:
            x = xm
        else:
            f = (t - p1) / (b - p1)
            x = xm + (x1 - xm) * (f * f * 0.4 + 0.6 * f)
        turn = float(ss(t, p0 + 0.3, p0 + 1.0) * (1 - ss(t, p1 - 0.6, p1)))
        return x * self.W, 0.885 * self.H, 0.125 * self.S, turn

    def draw_fox(self, dots, glows, t, amb, key):
        st = self.fox_state(t)
        if st is None:
            return None
        fx, fy, L, turn = st
        dist = (1.10 * self.W - fx) / L                      # walked so far (in body lengths)
        moving = float(1 - ss(t, 40.4, 41.0) * (1 - ss(t, 44.6, 45.2)))
        phi = dist * TAU / 0.95
        Fx = self.fox
        x, y, part = Fx["x"].copy(), Fx["y"].copy(), Fx["part"]
        y = y - 0.02 * abs(math.sin(phi)) * moving
        head = part >= 1                                     # the head lifts and turns to us
        hp = np.array([-0.48, -0.74])
        lift = 0.18 * turn
        hx, hy = x[head] - hp[0], y[head] - hp[1]
        c, s = math.cos(lift), math.sin(lift)
        x[head], y[head] = hp[0] + hx * c + hy * s, hp[1] - hx * s + hy * c
        sn = part == 2
        x[sn] = -0.70 + (x[sn] + 0.70) * (1 - 0.65 * turn)
        tail = part == 3                                      # the brush, held out and swaying
        u, v = Fx["x"][tail], Fx["y"][tail]
        sw = 0.05 * math.sin(phi * 0.5 + 0.7) * moving + 0.03 * math.sin(1.3 * t)
        P = [(0.38, -0.64), (0.62, -0.62 + sw), (0.86, -0.46 + 2 * sw), (1.04, -0.34 + 3 * sw)]
        bx, by, dx, dy = bezier(P, u)
        nn = np.hypot(dx, dy) + 1e-6
        x[tail], y[tail] = bx - dy / nn * v, by + dx / nn * v
        base = Fx["col"] * (amb * 1.15 + 0.05)
        col = base + key * 1.1 * Fx["rim"][:, None]
        X, Y = fx + x * L, fy + y * L
        dots.append((X, Y, col, 1.6 * L / 105, fy - 0.1))
        legs = [(-0.32, 0.0, 1), (-0.36, math.pi, 0), (0.30, math.pi, 1), (0.34, 0.0, 0)]   # trot: diagonal pairs
        for hx0, ph, nearside in legs:
            sw = math.sin(phi + ph) * moving
            lift = max(0.0, math.cos(phi + ph)) * 0.09 * moving
            hip = np.array([hx0, -0.52])
            foot = np.array([hx0 + 0.20 * sw, -lift])
            knee = (hip + foot) / 2 + np.array([-0.05 - 0.05 * lift / 0.09 * (hx0 < 0), 0.0])
            s = np.linspace(0, 1, 30)
            px = np.where(s < 0.5, hip[0] + (knee[0] - hip[0]) * s * 2, knee[0] + (foot[0] - knee[0]) * (s * 2 - 1))
            py = np.where(s < 0.5, hip[1] + (knee[1] - hip[1]) * s * 2, knee[1] + (foot[1] - knee[1]) * (s * 2 - 1))
            lc = np.where((s > 0.45)[:, None], np.array([26, 22, 22], np.float32), np.array([150, 72, 34], np.float32))
            shade = 1.0 if nearside else 0.6
            dots.append((fx + px * L, fy + py * L, lc * amb * 1.25 * shade + key * 0.3 * shade,
                         (3.0 - 1.4 * s) * L / 105, fy + (0.2 if nearside else -0.3)))
        if turn > 0.2:                                        # eyeshine
            ey = fy + (-0.88 - 0.10 * turn) * L
            for ex in (-0.73, -0.64):
                X = fx + (ex + 0.02 * (1 - turn)) * L
                glows.append((X, ey, np.array([170, 255, 160]) * (turn - 0.2) / 0.8, 1.6))
        return fx, fy, L

    # ---- the frame ---------------------------------------------------------------------------------
    def moon_pos(self, t):
        f = float(np.clip((t - 22.0) / 38.0, 0, 1))
        W, H = self.W, self.H
        return W * (0.30 + 0.12 * f), H * (0.62 - 0.42 * (1 - (1 - f) ** 1.6)), f

    def render(self, t):
        img = self.backdrop.at(t)
        amb = keyed(t, self.amb_keys)
        key = keyed(t, self.key_keys)
        mx, my, f = self.moon_pos(t)
        if f > 0:
            warm = 1 - ss(f, 0.0, 0.5)
            core = np.array([255, 196, 140]) * warm + np.array([250, 246, 230]) * (1 - warm)
            self.moon.draw(img, mx, my, core, np.array([40, 42, 56]) * ss(f, 0.0, 0.15), self.sky_mask, self.glow_mask)
        dots, glows = [], []
        self.draw_grass(dots, t, amb, key)
        fox = self.draw_fox(dots, glows, t, amb, key)
        paint(img, dots)
        for x, y, c, r in glows:
            glow(img, x, y, c, r)
        self.draw_flies(img, t, fox)
        return np.clip(img, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    run(Fireflies, "Firefly meadow", "fireflies", T_END, STAGES)
