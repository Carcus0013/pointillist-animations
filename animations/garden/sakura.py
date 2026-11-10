"""
sakura.py - a cherry branch from winter snow to a storm of petals, in dots.

A winter night: snow falls on a bare, gnarled branch under the moon and settles along
its upper edges. Dawn comes, the snow melts away and the buds swell, red then pink.
Blossoms burst open in a wave from the trunk out to the twig tips until the branch is
clouded in pink, and a white-eye drops in to hop from cluster to cluster. Then the wind
rises: the bird flees, the petals tear loose and whirl across the golden afternoon in
a blizzard, and by sunset fresh bronze-green leaves are unfolding where the flowers were.

    python sakura.py            interactive window (Space pause, R restart, S save frame, Esc quit)
    python sakura.py --png      save five stages
    python sakura.py --gif      save the whole film as a GIF
"""
import math

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))   # the shared engine: engine/pointillism.py

from pointillism import (Backdrop, cloud_streaks, glow, hill_layers, keyed, lit, mix, moon, paint, run, sample_v,
                         shades, sky, ss, stamp, stars, unit, warm_glow)

T_END = 60.0
STAGES = [5.0, 17.0, 31.0, 45.0, 57.0]

BARK = [(38, 28, 30), (52, 40, 40), (66, 52, 50)]
BARK_HI = [(104, 90, 88), (122, 108, 104)]
LENTICEL = (136, 124, 120)
SNOW = [(226, 234, 248), (240, 244, 252), (210, 220, 240)]
PETAL = [(255, 238, 244), (255, 216, 228), (250, 190, 210), (236, 140, 176)]     # edge -> heart
BUD = [(176, 52, 78), (200, 78, 104), (222, 108, 140)]
LEAF_YOUNG = [(172, 98, 66), (186, 122, 72)]
LEAF_GREEN = [(124, 182, 78), (150, 202, 98)]

LOOKS = [
    dict(t=0.0, sky=[(0.0, (8, 12, 34)), (0.50, (20, 30, 64)), (0.85, (46, 56, 96))], moon=1.0, stars=1.0,
         snow=1.0, far=(48, 60, 96), near=(30, 38, 66), mist=(60, 70, 110), glow=None,
         lit=[(120, 130, 170), (100, 110, 150)], shade=[(30, 36, 66), (40, 46, 78)],
         amb=(0.42, 0.48, 0.70), key=(50, 56, 78), dir=(0.6, -0.8)),
    dict(t=13.0, sky=[(0.0, (34, 38, 88)), (0.45, (112, 98, 152)), (0.85, (238, 176, 172))], moon=0.3, stars=0.3,
         snow=0.9, far=(110, 100, 150), near=(76, 70, 110), mist=(200, 170, 180), glow=((90, 50, 40), 0.15, 0.82, 0.30),
         lit=[(255, 190, 180), (240, 176, 176)], shade=[(90, 80, 130), (104, 90, 140)],
         amb=(0.60, 0.58, 0.76), key=(115, 76, 68), dir=(-0.8, -0.5)),
    dict(t=23.0, sky=[(0.0, (110, 156, 210)), (0.45, (186, 200, 232)), (0.85, (255, 228, 222))], moon=0.0, stars=0.0,
         snow=0.6, far=(150, 160, 200), near=(120, 136, 170), mist=(240, 236, 240), glow=None,
         lit=[(255, 250, 250), (250, 246, 250)], shade=[(196, 204, 226), (206, 212, 232)],
         amb=(0.94, 0.94, 1.0), key=(100, 94, 88), dir=(-0.6, -0.8)),
    dict(t=35.0, sky=[(0.0, (80, 146, 220)), (0.45, (156, 198, 240)), (0.85, (226, 236, 246))], moon=0.0, stars=0.0,
         snow=0.5, far=(140, 160, 200), near=(110, 136, 160), mist=(236, 240, 246), glow=None,
         lit=[(255, 255, 255), (248, 250, 255)], shade=[(190, 204, 230), (204, 214, 236)],
         amb=(1.0, 1.0, 1.0), key=(76, 75, 70), dir=(-0.2, -1.0)),
    dict(t=46.0, sky=[(0.0, (96, 136, 200)), (0.45, (214, 198, 196)), (0.85, (255, 210, 150))], moon=0.0, stars=0.0,
         snow=0.45, far=(150, 140, 170), near=(120, 110, 140), mist=(250, 220, 190), glow=((110, 70, 20), 0.85, 0.80, 0.30),
         lit=[(255, 226, 180), (255, 214, 170)], shade=[(170, 160, 190), (180, 168, 196)],
         amb=(0.95, 0.88, 0.80), key=(153, 120, 78), dir=(0.7, -0.6)),
    dict(t=56.0, sky=[(0.0, (56, 48, 108)), (0.40, (198, 108, 140)), (0.65, (255, 150, 110)), (0.85, (255, 196, 140))],
         moon=0.0, stars=0.0, snow=0.4, far=(140, 90, 130), near=(100, 64, 100), mist=(250, 170, 150),
         glow=((150, 70, 30), 0.88, 0.84, 0.35), lit=[(255, 160, 120), (255, 140, 120)], shade=[(110, 64, 110), (124, 70, 116)],
         amb=(0.72, 0.58, 0.66), key=(204, 120, 72), dir=(0.85, -0.45)),
]
CLOUDS = [(0.25, 0.12, 0.22, 0.012, 3500), (0.70, 0.22, 0.20, 0.010, 3000), (0.45, 0.34, 0.18, 0.008, 2400)]


def storm(t):
    return ss(t, 40.0, 43.0) * (1 - ss(t, 52.0, 57.0))


class Sakura:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.rng = np.random.default_rng(41)
        self.amb_keys = [(L["t"], L["amb"]) for L in LOOKS]
        self.key_keys = [(L["t"], L["key"]) for L in LOOKS]
        self.dir_keys = [(L["t"], unit(L["dir"])) for L in LOOKS]
        self.build_branch()
        self.backdrop = Backdrop([L["t"] for L in LOOKS], [self.background(L) for L in LOOKS])
        self.build_blossoms()
        self.build_leaves()
        self.build_bird()
        self.build_fx()

    # ---- backdrop -------------------------------------------------------------------------------
    def background(self, L):
        W, H = self.W, self.H
        rng = np.random.default_rng(99)
        img = sky(W, H, rng, L["sky"], 0.86 * H)
        if L["glow"]:
            col, gx, gy, gr = L["glow"]
            warm_glow(img, gx * W, gy * H, gr * H, col, 0.6)
        stars(img, rng, 420, 0.6 * H, L["stars"])
        moon(img, rng, 0.82 * W, 0.14 * H, 0.04 * H, L["moon"])
        cloud_streaks(img, rng, CLOUDS, L["lit"], L["shade"])
        # a far mountain with a snow cap, then nearer ridges in the mist
        xs = np.arange(W, dtype=np.float32)
        xn = xs / W
        top = H * (0.745 - 0.15 * np.exp(-((xn - 0.66) / 0.12) ** 2) - 0.04 * np.exp(-((xn - 0.28) / 0.16) ** 2)
                   - 0.006 * np.sin(xn * 40))
        yy = np.arange(H, dtype=np.float32)[:, None]
        far = np.asarray(L["far"], np.float32)
        img[(yy >= top[None, :]) & (yy < 0.88 * H)] = far * 0.85
        n = int(W * 0.25 * H * 0.5)
        x = rng.uniform(0, W, n)
        tp = np.interp(x, xs, top)
        d = rng.random(n)
        y = tp + d * (0.88 * H - tp)
        col = far * (1.08 - 0.25 * d)[:, None] * (1 + rng.normal(0, 0.06, (n, 1)))
        peak = float(top.min())                                          # everything above the snowline is white
        snowline = peak + (0.035 + 0.04 * L["snow"]) * H + 0.008 * H * np.sin(x * 0.05) + rng.normal(0, 0.004 * H, n)
        capped = y < snowline
        col[capped] = shades(rng, SNOW, rng.random(int(capped.sum()))) * (0.55 + 0.45 * far.mean() / 150)
        stamp(img, x, y, col.astype(np.float32), rng.choice([1.0, 1.5, 2.0], n, p=[0.3, 0.45, 0.25]))
        n = int(W * 4)
        glow(img, rng.uniform(0, W, n), 0.80 * H + rng.normal(0, 0.03 * H, n), np.asarray(L["mist"]) * 0.10, 3.0)
        hill_layers(img, rng, [(0.85, 0.03, L["near"], 1.3, 3.1, 0.4), (0.93, 0.025, tuple(np.asarray(L["near"]) * 0.75), 0.9, 2.7, 2.0)], H)
        n = int(W * 5)
        glow(img, rng.uniform(0, W, n), 0.90 * H + rng.normal(0, 0.03 * H, n), np.asarray(L["mist"]) * 0.10, 3.0)
        # the thick old wood near the trunk does not sway, so it is painted once per look
        B, still = self.bark, self.bark["w"] <= 0
        light = (np.asarray(L["amb"], np.float32), np.asarray(L["key"], np.float32), unit(L["dir"]))
        stamp(img, B["x"][still], B["y"][still],
              np.clip(lit(B["col"][still], B["x"][still], B["y"][still], B["cx"][still], B["cy"][still],
                          B["half"][still], 0.45, light), 0, 255), B["rad"][still])
        return img

    # ---- the branch -------------------------------------------------------------------------------
    def build_branch(self):
        W, H = self.W, self.H
        rng = np.random.default_rng(3)
        segs, tips = [], []
        probs = (0.30, 0.42, 0.40, 0.30)

        def grow(x, y, ang, length, thick, depth, dist):
            steps = max(3, int(length / (0.026 * H)))
            sl = length / steps
            for s in range(steps):
                ang += rng.normal(0, 0.15 if depth else 0.09)
                if depth >= 2:
                    ang += (-math.pi / 2 - ang) * 0.04                   # twigs reach for the light
                x1, y1 = x + math.cos(ang) * sl, y + math.sin(ang) * sl
                th1 = max(thick * (1 - 0.5 / steps), 0.0012 * H)
                segs.append((x, y, x1, y1, thick, th1, depth, dist))
                dist += sl
                if not (-0.05 * W < x1 < 1.05 * W and -0.05 * H < y1 < 0.95 * H):
                    break
                if depth < 4 and s >= 1 and rng.random() < probs[depth]:
                    side = 1 if rng.random() < 0.5 else -1
                    grow(x1, y1, ang + side * rng.uniform(0.45, 0.95), length * rng.uniform(0.32, 0.55),
                         th1 * 0.62, depth + 1, dist)
                x, y, thick = x1, y1, th1
            tips.append((x, y, depth, dist, ang))

        grow(-0.03 * W, 0.70 * H, -0.36, 0.95 * W, 0.052 * H, 0, 0.0)
        grow(0.10 * W, 0.66 * H, -1.05, 0.36 * W, 0.024 * H, 1, 0.12 * W)
        self.segs, self.tips = segs, tips
        self.maxdist = max(s[7] + math.hypot(s[2] - s[0], s[3] - s[1]) for s in segs)

        xs, ys, cxs, cys, half, cols, wts = [], [], [], [], [], [], []
        sx, sy, sthr, swt = [], [], [], []
        for x0, y0, x1, y1, t0, t1, depth, dist in segs:
            sl = max(math.hypot(x1 - x0, y1 - y0), 1e-3)
            th = 0.5 * (t0 + t1)
            n = int(sl * th * 0.21) + 3
            a = rng.random(n)
            l = sample_v(rng, n, 0.15)
            nx, ny = -(y1 - y0) / sl, (x1 - x0) / sl
            cx, cy = x0 + (x1 - x0) * a, y0 + (y1 - y0) * a
            thick = t0 + (t1 - t0) * a
            top = -l * (1 if ny > 0 else -1)                             # >0 on the upper side
            col = shades(rng, BARK, rng.random(n), 0.6)
            col[top > 0.55] = shades(rng, BARK_HI, rng.random(int((top > 0.55).sum())))
            if depth <= 1:                                               # bands of lenticels across the bark
                band = (np.mod((dist + a * sl) / (0.011 * H), 1.0) < 0.13) & (np.abs(l) < 0.85)
                col[band] = np.asarray(LENTICEL, np.float32) * (1 + rng.normal(0, 0.05, (int(band.sum()), 1)))
            xs.append(cx + nx * l * thick / 2)
            ys.append(cy + ny * l * thick / 2)
            cxs.append(cx)
            cys.append(cy)
            half.append(thick / 2 + 0.5)
            cols.append(col)
            wts.append(self.weight(dist + a * sl))
            if depth <= 3 and abs(ny) > 0.35:                            # where snow can settle
                ns = int(sl * (0.9 if depth < 2 else 0.5)) + 1
                a = rng.random(ns)
                sgn = -1.0 if ny > 0 else 1.0                            # the l that points up
                lift = sgn * (1.0 + rng.random(ns) * min(th * 0.8, 0.012 * H) / max(th, 1e-3) * 2)
                thick = t0 + (t1 - t0) * a
                sx.append(x0 + (x1 - x0) * a + nx * lift * thick / 2)
                sy.append(y0 + (y1 - y0) * a + ny * lift * thick / 2)
                sthr.append(rng.random(ns))
                swt.append(self.weight(dist + a * sl))
        self.bark = dict(x=np.concatenate(xs), y=np.concatenate(ys), cx=np.concatenate(cxs), cy=np.concatenate(cys),
                         half=np.concatenate(half), col=np.concatenate(cols), w=np.concatenate(wts))
        self.bark["rad"] = rng.choice([1.0, 1.5, 2.0], len(self.bark["x"]), p=[0.3, 0.5, 0.2])
        moving = self.bark["w"] > 0                                      # the rest is painted into the backdrop
        self.moving = {k: v[moving] for k, v in self.bark.items()}
        self.snowcap = dict(x=np.concatenate(sx), y=np.concatenate(sy), thr=np.concatenate(sthr), w=np.concatenate(swt))
        self.snowcap["col"] = shades(rng, SNOW, rng.random(len(self.snowcap["x"])))

    def weight(self, d):
        """How much a point this far along the wood sways: nothing for the inner third."""
        return np.clip((np.asarray(d, np.float64) / self.maxdist - 0.3) / 0.7, 0, 1) ** 1.5

    def sway(self, t, w, x, y):
        A = self.H * (0.003 + 0.012 * float(storm(t)) * (0.75 + 0.25 * math.sin(2.1 * t)))
        return (w * A * np.sin(1.1 * t + x * 0.004 + y * 0.002),
                w * A * 0.5 * np.sin(0.9 * t + x * 0.003 + 1.0))

    # ---- blossoms ---------------------------------------------------------------------------------
    def build_blossoms(self):
        rng, W, H = self.rng, self.W, self.H
        sites = []
        for x0, y0, x1, y1, t0, t1, depth, dist in self.segs:
            if (depth >= 2 and rng.random() < 0.45) or (depth == 1 and rng.random() < 0.10):
                sites.append((x1, y1, dist, math.atan2(y1 - y0, x1 - x0)))
        for x, y, depth, dist, ang in self.tips:
            if depth >= 1:
                sites.append((x, y, dist, ang))
        self.sites = np.array(sites)
        bl = []
        for x, y, dist, ang in sites:
            for _ in range(rng.integers(1, 4)):
                bl.append((x + rng.normal(0, 0.009 * H), y + rng.normal(0, 0.009 * H), dist))
        bl = np.array(bl)
        if len(bl) > 270:
            bl = bl[rng.choice(len(bl), 270, replace=False)]
        nb = len(bl)
        n3 = np.c_[rng.uniform(-1, 1, nb), rng.uniform(-1, 1, nb), rng.uniform(0.3, 1.0, nb)]
        n3 /= np.linalg.norm(n3, axis=1, keepdims=True)
        e1 = np.c_[n3[:, 1], -n3[:, 0], np.zeros(nb)]
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True) + 1e-9
        e2 = np.cross(n3, e1)
        frac = bl[:, 2] / self.maxdist
        self.bl = dict(x=bl[:, 0], y=bl[:, 1], w=self.weight(bl[:, 2]), n=n3, e1=e1, e2=e2,
                       R=0.0145 * H * rng.uniform(0.85, 1.15, nb), rot=rng.uniform(0, 2 * np.pi, nb),
                       T=19.5 + 9.0 * frac + rng.uniform(0, 2.5, nb), z=np.where(rng.random(nb) < 0.2, -1.0, 1.0) + rng.random(nb) * 0.5,
                       white=rng.random(nb) < 0.35)

        # every dot of every blossom, in the blossom's own unit disc
        per_petal, nc, nst = 12, 5, 8
        bid, kind, pk, lx, ly, rr = [], [], [], [], [], []
        for b in range(nb):
            for k in range(5):
                phi = self.bl["rot"][b] + 2 * np.pi * k / 5
                r = 0.22 + 0.78 * np.sqrt(rng.random(per_petal))
                hw = 0.58 * np.sin(np.pi * np.minimum(1, r * 0.95) ** 0.8) + 0.08
                psi = rng.uniform(-1, 1, per_petal) * hw
                notch = (r > 0.84) & (np.abs(psi) < 0.11)                  # the little notch at the tip
                r[notch] = 0.8
                bid.append(np.full(per_petal, b))
                kind.append(np.zeros(per_petal, int))
                pk.append(np.full(per_petal, k))
                lx.append(r * np.cos(phi + psi))
                ly.append(r * np.sin(phi + psi))
                rr.append(r)
            a = rng.uniform(0, 2 * np.pi, nc + nst)
            r = np.r_[rng.uniform(0, 0.15, nc), rng.uniform(0.32, 0.5, nst)]
            bid.append(np.full(nc + nst, b))
            kind.append(np.r_[np.ones(nc, int), np.full(nst, 2)])
            pk.append(np.full(nc + nst, -1))
            lx.append(r * np.cos(a))
            ly.append(r * np.sin(a))
            rr.append(r)
        bid, kind, pk = np.concatenate(bid), np.concatenate(kind), np.concatenate(pk)
        lx, ly, rr = np.concatenate(lx), np.concatenate(ly), np.concatenate(rr)
        col = shades(rng, PETAL, np.clip(0.1 + 0.9 * (1 - rr) ** 1.4, 0, 1), 0.35)
        wht = self.bl["white"][bid]
        col[wht] = mix(col[wht], np.array([255, 248, 250], np.float32), 0.5)
        col[kind == 1] = np.array([214, 80, 120], np.float32)
        col[kind == 2] = np.array([250, 222, 130], np.float32)
        self.bdots = dict(b=bid, kind=kind, k=pk, lx=lx, ly=ly, r=rr, col=col,
                          bud=shades(rng, BUD, rng.random(len(bid))),
                          rad=np.where(kind == 0, rng.choice([1.0, 1.5, 2.0], len(bid), p=[0.3, 0.5, 0.2]), 1.2))

        # when each petal lets go: a few drift down in full bloom, most go in the storm
        npet = nb * 5
        pb, pk_ = np.repeat(np.arange(nb), 5), np.tile(np.arange(5), nb)
        early = rng.random(npet) < 0.08
        T = np.where(early, 32.0 + 9.0 * rng.random(npet), 41.5 + 9.0 * rng.random(npet) ** 0.8)
        T = np.maximum(T, self.bl["T"][pb] + 3.0)
        self.pfall = T.reshape(nb, 5)
        # where each petal starts its flight
        phi = self.bl["rot"][pb] + 2 * np.pi * pk_ / 5
        R = self.bl["R"][pb]
        P = R[:, None] * 0.6 * (np.cos(phi)[:, None] * self.bl["e1"][pb] + np.sin(phi)[:, None] * self.bl["e2"][pb])
        x0, y0 = self.bl["x"][pb] + P[:, 0], self.bl["y"][pb] - P[:, 1]
        dx, dy = np.zeros(npet), np.zeros(npet)
        for i in range(npet):
            dx[i], dy[i] = self.sway(T[i], self.bl["w"][pb[i]], x0[i], y0[i])
        col = mix(shades(rng, PETAL[:2], rng.random(npet), 0.6), np.array([255, 248, 250], np.float32),
                  0.5 * self.bl["white"][pb])
        # petals from trees upwind join the storm, blowing in from the left edge
        ne = 900
        Te = 41.0 + 11.0 * rng.random(ne)
        x0 = np.r_[x0 + dx, -0.03 * W - 0.05 * W * rng.random(ne)]
        y0 = np.r_[y0 + dy, H * rng.uniform(-0.05, 0.85, ne)]
        T = np.r_[T, Te]
        R = np.r_[R, 0.0145 * H * rng.uniform(0.8, 1.2, ne)]
        col = np.r_[col, shades(rng, PETAL[:2], rng.random(ne), 0.6)]
        n = len(T)
        st = storm(T)
        self.flyer = dict(T=T, x=x0, y=y0, st=st,
                          u=W * (0.02 + 0.12 * st) * rng.uniform(0.6, 1.4, n),
                          v=H * (0.035 + 0.03 * rng.random(n)) - H * 0.05 * st * rng.random(n),
                          ax=W * (0.01 + 0.06 * st) * rng.random(n), wx=rng.uniform(1.2, 3.2, n),
                          ay=H * (0.01 + 0.05 * st) * rng.random(n), ph=rng.uniform(0, 6.28, n),
                          a0=rng.uniform(0, 6.28, n), spin=rng.uniform(-4, 4, n), flip=rng.uniform(2, 6, n),
                          R=R * 1.3, col=col)

    # ---- leaves, bird, weather ---------------------------------------------------------------------
    def build_leaves(self):
        rng, H = self.rng, self.H
        S = self.sites
        idx = np.nonzero(rng.random(len(S)) < 0.75)[0]
        n = len(idx)
        self.lv = dict(x=S[idx, 0], y=S[idx, 1], w=self.weight(S[idx, 2]),
                       ang=S[idx, 3] + rng.choice([-1, 1], n) * rng.uniform(0.5, 1.2, n),
                       L=0.021 * H * rng.uniform(0.8, 1.2, n),
                       T=44.0 + 6.0 * S[idx, 2] / self.maxdist + 2.0 * rng.random(n))
        m = 22
        u = np.tile(np.linspace(0.05, 1, m // 2), 2)
        v = np.r_[np.linspace(-1, 1, m // 2), np.linspace(1, -1, m // 2)] * rng.uniform(0.2, 1.0, m)
        self.leaf_uv = (u, v)
        self.leaf_col = (shades(rng, LEAF_YOUNG, rng.random(n * m)).reshape(n, m, 3),
                         shades(rng, LEAF_GREEN, rng.random(n * m)).reshape(n, m, 3))

    def build_bird(self):
        """A white-eye, drawn in a unit frame facing right; s = body length / 2."""
        rng = self.rng
        pts, cols, part = [], [], []

        def blob(cx, cy, rx, ry, n, pal, tag):
            a, r = rng.uniform(0, 2 * np.pi, n), np.sqrt(rng.random(n))
            pts.append(np.c_[cx + rx * r * np.cos(a), cy + ry * r * np.sin(a)])
            cols.append(shades(rng, pal, rng.random(n), 0.6))
            part.append(np.full(n, tag))

        blob(0, 0, 0.9, 0.55, 110, [(120, 150, 52), (104, 134, 46)], 0)                # back
        blob(0.1, 0.22, 0.7, 0.32, 60, [(228, 224, 176), (214, 206, 156)], 0)          # pale belly
        blob(0.55, -0.15, 0.3, 0.25, 25, [(232, 214, 84), (220, 200, 70)], 0)          # yellow throat
        blob(-0.15, -0.1, 0.55, 0.28, 40, [(84, 104, 40), (72, 92, 36)], 3)             # wing
        blob(0.85, -0.55, 0.42, 0.40, 50, [(126, 156, 54), (112, 142, 48)], 1)         # head
        t = np.linspace(0, 1, 12)
        pts.append(np.c_[-0.8 - 0.8 * t, 0.1 + 0.35 * t])                              # tail
        cols.append(np.tile([[76, 92, 38]], (12, 1)).astype(np.float32))
        part.append(np.full(12, 0))
        a = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        pts.append(np.c_[0.98 + 0.14 * np.cos(a), -0.62 + 0.14 * np.sin(a)])           # the white eye-ring
        cols.append(np.tile([[250, 250, 244]], (10, 1)).astype(np.float32))
        part.append(np.full(10, 1))
        pts.append(np.array([[0.98, -0.62], [1.27, -0.50], [1.36, -0.48], [1.44, -0.47]]))   # eye and beak
        cols.append(np.array([[20, 20, 22], [44, 40, 40], [44, 40, 40], [44, 40, 40]], np.float32))
        part.append(np.full(4, 1))
        self.bird = dict(p=np.concatenate(pts), col=np.concatenate(cols), part=np.concatenate(part))
        # three perches in the middle of the branch, left to right
        S = self.sites
        mid = S[(S[:, 0] > 0.30 * self.W) & (S[:, 0] < 0.78 * self.W) & (S[:, 1] > 0.22 * self.H) & (S[:, 1] < 0.58 * self.H)]
        mid = mid[rng.permutation(len(mid))]
        pick = []
        for s in mid:                                                   # perches well apart from each other
            if all(math.hypot(s[0] - p[0], s[1] - p[1]) > 0.10 * self.W for p in pick):
                pick.append(s)
            if len(pick) == 3:
                break
        pick = np.array(pick if len(pick) == 3 else S[:3])
        self.perch = pick[np.argsort(pick[:, 0])]
        self.plan = [(27.0, 28.6, "fly", "in", 0), (28.6, 31.6, "sit", 0, 0), (31.6, 32.1, "hop", 0, 1),
                     (32.1, 35.2, "sit", 1, 1), (35.2, 35.7, "hop", 1, 2), (35.7, 38.6, "sit", 2, 2),
                     (38.6, 39.1, "hop", 2, 1), (39.1, 41.2, "sit", 1, 1), (41.2, 43.0, "fly", 1, "out")]

    def build_fx(self):
        rng, W, H = self.rng, self.W, self.H
        n = 650
        near = rng.random(n) < 0.15
        self.snow = dict(x=rng.uniform(0, W, n), ph=rng.random(n), v=rng.uniform(0.05, 0.11, n), near=near,
                         rad=np.where(near, 2.5, rng.choice([1.0, 1.5], n)), w=rng.uniform(0, 6.28, n))
        n = 40
        self.streaks = dict(y=H * (0.1 + 0.75 * rng.random(n)), sp=rng.uniform(0.7, 1.3, n), ph=rng.random(n),
                            w=rng.uniform(0, 6.28, n))

    # ---- drawing ---------------------------------------------------------------------------------
    def render(self, t):
        img = self.backdrop.at(t)
        light = (keyed(t, self.amb_keys), keyed(t, self.key_keys), unit(keyed(t, self.dir_keys)))
        dots = []
        self.draw_branch(dots, t, light)
        self.draw_blossoms(dots, t, light)
        self.draw_leaves(dots, t, light)
        self.draw_bird(dots, t, light)
        self.draw_flyers(dots, t, light)
        self.draw_snow(dots, t, light)
        paint(img, dots)
        self.draw_wind(img, t)
        return np.clip(img, 0, 255).astype(np.uint8)

    def draw_branch(self, dots, t, light):
        B = self.moving
        dx, dy = self.sway(t, B["w"], B["cx"], B["cy"])
        X, Y = B["x"] + dx, B["y"] + dy
        col = lit(B["col"], X, Y, B["cx"] + dx, B["cy"] + dy, B["half"], 0.45, light)
        dots.append((X, Y, col, B["rad"], 0.0))
        cover = float(ss(t, 0.5, 8.0) * (1 - ss(t, 11.0, 18.0)))       # snow settles, then melts
        if cover > 0.01:
            S = self.snowcap
            m = S["thr"] < cover
            dx, dy = self.sway(t, S["w"][m], S["x"][m], S["y"][m])
            X, Y = S["x"][m] + dx, S["y"][m] + dy
            dots.append((X, Y, S["col"][m] * light[0] + light[1] * 0.4, 1.5, 0.5))

    def draw_blossoms(self, dots, t, light):
        Bl, D = self.bl, self.bdots
        bud = float(ss(t, 11.0, 19.0))
        if bud < 0.01:
            return
        o_b = ss(t, Bl["T"], Bl["T"] + 2.4)
        b = D["b"]
        o = o_b[b]
        petal = D["kind"] == 0
        gone = np.zeros(len(b), bool)
        gone[petal] = t >= self.pfall[b[petal], D["k"][petal]]
        keep = ~gone & (petal | (o > 0.3))
        b, o = b[keep], o[keep]
        s = 0.35 * bud * (1 - o) + o                                     # a tight bud, then wide open
        R = Bl["R"][b] * s
        lx, ly, r = D["lx"][keep], D["ly"][keep], D["r"][keep]
        P = R[:, None] * (lx[:, None] * Bl["e1"][b] + ly[:, None] * Bl["e2"][b]) + (Bl["R"][b] * 0.12 * r ** 2 * o)[:, None] * Bl["n"][b]
        cx, cy = Bl["x"][b], Bl["y"][b]
        dx, dy = self.sway(t, Bl["w"][b], cx, cy)
        cx, cy = cx + dx, cy + dy
        X, Y = cx + P[:, 0], cy - P[:, 1]
        col = mix(D["bud"][keep], D["col"][keep], o)
        col = lit(col, X, Y, cx, cy, Bl["R"][b] + 1, 0.35, light, wash=0.15)
        dots.append((X, Y, col, D["rad"][keep] * (0.6 + 0.4 * o), 2.0 + Bl["z"][b] + 0.1 * P[:, 2] / (Bl["R"][b] + 1)))

    def draw_leaves(self, dots, t, light):
        V = self.lv
        g = ss(t, V["T"], V["T"] + 4.0)
        m = g > 0.01
        if not m.any():
            return
        u, v = self.leaf_uv
        L = (V["L"][m] * g[m])[:, None]
        a = V["ang"][m][:, None]
        wid = 0.32 * L * np.sin(np.pi * u ** 0.8)[None, :] * v[None, :]
        lx, ly = u[None, :] * L, wid
        dx, dy = self.sway(t, V["w"][m], V["x"][m], V["y"][m])
        X = V["x"][m][:, None] + dx[:, None] + np.cos(a) * lx - np.sin(a) * ly
        Y = V["y"][m][:, None] + dy[:, None] + np.sin(a) * lx + np.cos(a) * ly
        mature = ss(t, V["T"][m] + 2.0, V["T"][m] + 10.0)
        col = self.leaf_col[0][m] * (1 - mature)[:, None, None] + self.leaf_col[1][m] * mature[:, None, None]
        X, Y, col = X.ravel(), Y.ravel(), col.reshape(-1, 3)
        dots.append((X, Y, lit(col, X, Y, X, Y, 1.0, 0.3, light), 1.5, 1.5))

    def bird_at(self, t):
        """Where the bird is, which way it faces, and what it is doing."""
        W, H = self.W, self.H

        def spot(p, tt):
            if p == "in":
                return np.array([1.12 * W, 0.12 * H])
            if p == "out":
                return np.array([1.15 * W, -0.10 * H])
            x, y = self.perch[p, 0], self.perch[p, 1]
            dx, dy = self.sway(tt, self.weight(self.perch[p, 2]), x, y)
            return np.array([x + dx, y + dy - 0.012 * H])

        for t0, t1, kind, a, b in self.plan:
            if t0 <= t < t1:
                u = (t - t0) / (t1 - t0)
                pa, pb = spot(a, t), spot(b, t)
                face = 1.0 if pb[0] >= pa[0] else -1.0
                if kind == "sit":
                    return pa, self._face, "sit"
                e = float(ss(u, 0.0, 1.0))
                p = pa + (pb - pa) * e
                p[1] -= (0.08 if kind == "fly" else 0.035) * H * math.sin(math.pi * u)
                self._face = face
                return p, face, kind
        return None, 1.0, None

    def draw_bird(self, dots, t, light):
        if not 27.0 <= t < 43.0:
            return
        if not hasattr(self, "_face"):
            self._face = -1.0
        p, face, mode = self.bird_at(t)
        if p is None:
            return
        B = self.bird
        s = 0.020 * self.H
        q = B["p"].copy()
        if mode == "sit":                                               # peck at the blossoms now and then
            peck = max(0.0, math.sin(9 * t)) ** 4 * (math.sin(1.3 * t) > 0.3)
            head = B["part"] == 1
            q[head, 0] += 0.12 * peck
            q[head, 1] += 0.18 * peck
        else:                                                           # flap
            wing = B["part"] == 3
            q[wing, 1] -= 0.7 * math.sin(26 * t) * (0.4 + q[wing, 0] * 0)
        X = p[0] + face * q[:, 0] * s
        Y = p[1] + q[:, 1] * s
        dots.append((X, Y, lit(B["col"], X, Y, p[0], p[1], s, 0.3, light), 1.6, 50.0))

    def draw_flyers(self, dots, t, light):
        F = self.flyer
        f = t - F["T"]
        m = (f > 0) & (f < 14.0)
        if not m.any():
            return
        f = f[m]
        x = F["x"][m] + F["u"][m] * f + F["ax"][m] * np.sin(F["wx"][m] * f + F["ph"][m])
        y = F["y"][m] + F["v"][m] * f + F["ay"][m] * np.cos(F["wx"][m] * f + F["ph"][m])
        a = F["a0"][m] + F["spin"][m] * f
        flat = 0.25 + 0.75 * np.abs(np.cos(F["flip"][m] * f))          # tumbling, edge-on now and then
        R = F["R"][m]
        ou = np.array([-0.5, 0.0, 0.5, -0.25, -0.25, 0.25, 0.25])
        ov = np.array([0.0, 0.0, 0.0, -0.32, 0.32, -0.32, 0.32])
        lu = R[:, None] * ou[None, :]
        lv = (R * flat)[:, None] * ov[None, :]
        X = x[:, None] + np.cos(a)[:, None] * lu - np.sin(a)[:, None] * lv
        Y = y[:, None] + np.sin(a)[:, None] * lu + np.cos(a)[:, None] * lv
        col = np.repeat(F["col"][m], 7, axis=0)
        X, Y = X.ravel(), Y.ravel()
        dots.append((X, Y, lit(col, X, Y, np.repeat(x, 7), np.repeat(y, 7), np.repeat(R, 7), 0.4, light, wash=0.2),
                     1.5, 100.0))

    def draw_snow(self, dots, t, light):
        fall = float(1 - ss(t, 8.0, 13.0))
        if fall < 0.01:
            return
        S = self.snow
        n = int(len(S["x"]) * fall)
        x = (S["x"][:n] + 0.02 * self.W * np.sin(0.6 * t + S["w"][:n]) + 0.01 * self.W * t) % self.W
        y = ((S["ph"][:n] + t * S["v"][:n]) % 1.1 - 0.05) * self.H
        col = np.array([232, 238, 250]) * light[0] * np.where(S["near"][:n], 0.7, 1.0)[:, None]
        dots.append((x, y, col, S["rad"][:n], np.where(S["near"][:n], 1e5, np.where(np.arange(n) % 2, -1e5, 1e5))))

    def draw_wind(self, img, t):
        st = float(storm(t))
        if st < 0.01:
            return
        W, H = self.W, self.H
        S = self.streaks
        head = ((t * 0.3 * S["sp"] + S["ph"]) % 1.4 - 0.2) * W
        j = np.arange(16)
        xs = (head[:, None] - j[None, :] * 4.0 * W / 1000).ravel()
        ys = (S["y"][:, None] + 0.006 * H * np.sin(xs.reshape(-1, 16) / W * 10 + S["w"][:, None])).ravel()
        b = (st * 0.14 * (1 - j / 16))[None, :].repeat(len(head), 0).ravel()
        glow(img, xs, ys, np.array([255, 240, 230]) * b[:, None], 1.0)


if __name__ == "__main__":
    run(Sakura, "Sakura - from snow to a storm of petals", "sakura", T_END, STAGES)
