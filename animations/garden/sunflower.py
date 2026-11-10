"""
sunflower.py - one summer day in the life of a sunflower, in dots.

Dawn: a seedling opens its first two leaves as the sun comes up behind the hills.
Morning: the stalk shoots up leaf by leaf and a spiky green bud turns to face the sun.
Midday: golden petals unfurl and the disc blooms in rings from the rim inward, florets
packed on the golden-angle spiral, while bees work it. All afternoon the head follows
the sun across the sky. At sunset the petals wither and drop, the disc ripens to black
seed, and the heavy head bows. Night falls over the field; a star shoots past.

    python sunflower.py            interactive window (Space pause, R restart, S save frame, Esc quit)
    python sunflower.py --png      save five stages
    python sunflower.py --gif      save the whole day as a GIF
"""
import math

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))   # the shared engine: engine/pointillism.py

from pointillism import (Backdrop, Sun, bezier, cloud_puffs, glow, hill_layers, join, keyed, lights, lit, mix,
                         moon, paint, run, sample_u, sample_v, shades, sky, ss, stamp, stars, unit, warm_glow)

T_END = 60.0
STAGES = [3.5, 15.0, 30.0, 45.0, 57.5]
GA = math.pi * (3 - math.sqrt(5))          # the golden angle: how a sunflower packs its florets

PETAL = [(236, 146, 22), (250, 182, 30), (255, 204, 44), (255, 222, 80)]
PETAL_DEAD = [(196, 124, 44), (164, 104, 46), (132, 90, 52)]
LEAF = [(40, 92, 34), (56, 118, 40), (78, 140, 48), (104, 160, 60)]
LEAF_VEIN = [(130, 176, 84), (150, 190, 96)]
LEAF_DEAD = [(156, 134, 52), (130, 102, 44), (104, 84, 44)]
STEM = [(54, 96, 34), (72, 120, 42), (96, 142, 54)]
BRACT = [(46, 92, 34), (62, 116, 42), (88, 140, 54)]
SOIL = [(34, 28, 22), (48, 38, 28), (62, 50, 36), (40, 40, 30)]
GRASS = [(34, 70, 30), (50, 96, 38), (76, 124, 50)]
CLOUDS = [(0.15, 0.16, 0.15, 7000), (0.62, 0.10, 0.19, 9000), (0.88, 0.30, 0.12, 5000), (0.40, 0.34, 0.10, 4000)]
MOON_XY = (0.20, 0.17)

LEAVES = [  # height on the stalk, side, size, azimuth (towards / away from the viewer), roll
    (0.10, -1, 1.00, -0.35, 0.55), (0.18, 1, 0.97, 0.40, 0.60), (0.27, -1, 0.92, 0.30, 0.55),
    (0.36, 1, 0.85, -0.30, 0.60), (0.46, -1, 0.78, -0.45, 0.55), (0.56, 1, 0.70, 0.35, 0.60),
    (0.66, -1, 0.60, 0.20, 0.55), (0.76, 1, 0.50, -0.25, 0.60),
]

# the day, as six looks the backdrop cross-fades between
LOOKS = [
    dict(t=0.0, sky=[(0.0, (20, 24, 62)), (0.32, (58, 60, 110)), (0.55, (160, 116, 148)), (0.70, (246, 168, 128))],
         glow=((120, 60, 30), 0.06, 0.70, 0.30), lit=[(255, 176, 150), (246, 150, 140)],
         shade=[(64, 60, 108), (84, 76, 124)], far=(86, 74, 118), near=(60, 64, 92),
         field=(0.45, 0.42, 0.55), ground=(0.40, 0.40, 0.55), stars=0.35, moon=0.0,
         amb=(0.52, 0.52, 0.72), key=(150, 100, 70)),
    dict(t=9.0, sky=[(0.0, (84, 140, 210)), (0.40, (148, 190, 232)), (0.70, (248, 222, 186))],
         glow=((90, 60, 20), 0.20, 0.62, 0.30), lit=[(255, 246, 232), (255, 236, 214)],
         shade=[(176, 186, 214), (196, 204, 226)], far=(120, 150, 170), near=(96, 130, 96),
         field=(0.85, 0.85, 0.85), ground=(0.75, 0.80, 0.75), stars=0.0, moon=0.0,
         amb=(0.86, 0.88, 0.95), key=(130, 110, 80)),
    dict(t=24.0, sky=[(0.0, (36, 104, 200)), (0.40, (88, 158, 230)), (0.70, (186, 214, 240))],
         glow=None, lit=[(255, 255, 255), (246, 250, 255)],
         shade=[(176, 192, 220), (200, 212, 232)], far=(110, 150, 180), near=(90, 140, 80),
         field=(1.0, 1.0, 1.0), ground=(1.0, 1.0, 1.0), stars=0.0, moon=0.0,
         amb=(1.0, 1.0, 1.0), key=(90, 85, 70)),
    dict(t=37.0, sky=[(0.0, (64, 112, 190)), (0.40, (150, 172, 210)), (0.70, (255, 208, 140))],
         glow=((110, 70, 20), 0.80, 0.62, 0.30), lit=[(255, 224, 170), (255, 210, 150)],
         shade=[(160, 156, 190), (180, 170, 196)], far=(130, 130, 150), near=(110, 130, 70),
         field=(1.05, 0.95, 0.80), ground=(0.95, 0.88, 0.75), stars=0.0, moon=0.0,
         amb=(0.92, 0.85, 0.76), key=(170, 120, 60)),
    dict(t=48.0, sky=[(0.0, (38, 30, 88)), (0.32, (140, 66, 118)), (0.55, (238, 116, 70)), (0.70, (255, 186, 106))],
         glow=((150, 70, 20), 0.93, 0.68, 0.32), lit=[(255, 150, 100), (255, 124, 108)],
         shade=[(98, 54, 104), (124, 62, 110)], far=(112, 60, 96), near=(70, 40, 64),
         field=(0.70, 0.50, 0.50), ground=(0.60, 0.45, 0.50), stars=0.1, moon=0.0,
         amb=(0.62, 0.48, 0.56), key=(220, 120, 60)),
    dict(t=56.0, sky=[(0.0, (6, 8, 24)), (0.45, (14, 18, 44)), (0.70, (34, 36, 72))],
         glow=None, lit=[(150, 156, 190), (130, 136, 170)],
         shade=[(30, 32, 56), (40, 42, 66)], far=(26, 30, 54), near=(18, 22, 36),
         field=(0.22, 0.24, 0.36), ground=(0.20, 0.22, 0.34), stars=1.0, moon=1.0,
         amb=(0.30, 0.33, 0.50), key=(70, 80, 120)),
]


def phases(t):
    """The story as 0..1 dials over time."""
    return dict(
        sprout=ss(t, 0.3, 3.0),
        grow=0.07 * ss(t, 0.3, 3.0) + 0.93 * ss(t, 2.5, 24.0),
        bud=ss(t, 9.0, 18.0),
        open=ss(t, 19.0, 27.0),
        disc=ss(t, 24.0, 38.0),
        seed=ss(t, 38.0, 50.0),
        wilt=ss(t, 40.0, 48.0),
        bow=ss(t, 44.0, 55.0),
        night=ss(t, 49.0, 55.0),
    )


def leaf_width(u):
    u = np.clip(u, 0, 1)
    heart = np.sin(np.pi * np.clip(0.06 + 0.94 * u, 0, 1) ** 0.85) ** 0.7 + 0.22 * np.exp(-((u - 0.04) / 0.07) ** 2)
    return np.clip(heart * (1 + 0.04 * np.sin(u * 70)) / 1.2, 0, 1)


class Sunflower:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.rng = np.random.default_rng(21)
        self.horizon = 0.70 * H
        self.y0 = 0.89 * H
        self.cx = 0.46 * W
        self.Ls = 0.56 * H
        self.Rd = 0.060 * H
        self.Lp = 0.066 * H
        self.sun_low = self.horizon + 0.03 * H
        self.sun = Sun(np.random.default_rng(5), 0.042 * H, 5.0)
        self.amb_keys = [(L["t"], L["amb"]) for L in LOOKS]
        self.key_keys = [(L["t"], L["key"]) for L in LOOKS]
        self.build_backdrop()
        self.build_stem()
        self.build_leaves()
        self.build_disc()
        self.build_petals()
        self.build_bracts()
        self.build_back()
        self.build_bees()
        self.build_falls()

    # ---- sky and the field ------------------------------------------------------------
    def sun_pos(self, t):
        p = float(np.clip((t - 1.0) / 48.0, 0.0, 1.0))            # rises at t=1, sets at t=49
        alt = math.sin(math.pi * p)
        return self.W * (0.06 + 0.88 * p), self.sun_low - (self.sun_low - 0.10 * self.H) * alt, alt

    def build_backdrop(self):
        imgs = [self.background(L) for L in LOOKS]
        self.backdrop = Backdrop([L["t"] for L in LOOKS], imgs)
        yy = np.arange(self.H, dtype=np.float32)[:, None]
        self.sky_mask = (yy < self.far_top[None, :]).astype(np.float32)
        self.glow_mask = (yy < 0.845 * self.H).astype(np.float32) * np.ones((1, self.W), np.float32)

    def background(self, L):
        W, H = self.W, self.H
        rng = np.random.default_rng(1234)                         # same dots in every look
        img = sky(W, H, rng, L["sky"], self.horizon + 0.03 * H)
        if L["glow"]:
            col, gx, gy, gr = L["glow"]
            warm_glow(img, gx * W, gy * H, gr * H, col, 0.6)
        stars(img, rng, 500, 0.62 * H, L["stars"])
        moon(img, rng, MOON_XY[0] * W, MOON_XY[1] * H, 0.036 * H, L["moon"])
        light_xy = self.sun_pos(L["t"])[:2] if L["t"] < 50 else (MOON_XY[0] * W, MOON_XY[1] * H)
        cloud_puffs(img, rng, CLOUDS, L["lit"], L["shade"], light_xy)
        self.far_top, self.near_top = hill_layers(
            img, rng, [(0.690, 0.022, L["far"], 1.1, 3.3, 0.0), (0.735, 0.012, L["near"], 1.7, 4.1, 1.7)], 0.86 * H)
        self.field(img, rng, L, light_xy)
        self.ground(img, rng, L)
        return img

    def field(self, img, rng, L, light_xy):
        """Rows of distant sunflowers on the near hill, all turned the same way."""
        W, H = self.W, self.H
        xs_all = np.arange(W, dtype=np.float32)
        bottom = 0.85 * H
        mult = np.asarray(L["field"], np.float32)
        haze_col = np.asarray(L["near"], np.float32)
        face = 0.0 if L["t"] >= 50 else float(np.clip((light_xy[0] - W / 2) / (0.4 * W), -1, 1))
        rows = 28
        for r in range(rows):
            f = (r + 0.5) / rows
            size = (1.0 + 4.4 * f ** 1.4) * W / 1000
            sp = size * 2.8
            x = np.arange(rng.uniform(0, sp), W, sp)
            n = len(x)
            x = x + rng.normal(0, size * 0.4, n)
            yt = np.interp(x, xs_all, self.near_top)
            y = yt + 0.01 * H + (bottom - yt - 0.01 * H) * f ** 1.6 + rng.normal(0, size * 0.3, n)
            haze = (1 - f) * 0.55
            pet = shades(rng, [(255, 214, 60), (250, 190, 40), (236, 166, 30)], rng.random(n * 6), 0.6)
            pet = pet * mult * (1 - haze) + haze_col * haze
            ang = np.tile(np.linspace(0, 2 * np.pi, 6, endpoint=False), n) + np.repeat(rng.uniform(0, 1, n), 6)
            px = np.repeat(x - face * size * 0.15, 6) + np.cos(ang) * size * 0.8
            py = np.repeat(y, 6) + np.sin(ang) * size * 0.6
            stamp(img, np.repeat(x, 2), np.repeat(y, 2) + np.tile([size * 1.2, size * 2.2], n),
                  np.array([66, 104, 40]) * mult * (1 - haze) + haze_col * haze, max(1.0, size * 0.35))
            stamp(img, px, py, pet, max(1.0, size * 0.42))
            stamp(img, x + face * size * 0.3, y, np.array([86, 52, 28]) * mult * (1 - haze) + haze_col * haze,
                  max(1.0, size * 0.5))

    def ground(self, img, rng, L):
        W, H = self.W, self.H
        g = np.asarray(L["ground"], np.float32)
        xs = np.arange(W, dtype=np.float32)
        top = 0.845 * H + 0.006 * H * np.sin(xs / W * 2 * np.pi * 2.2)
        img[np.arange(H)[:, None] >= top[None, :]] = np.array([30, 26, 20], np.float32) * g
        n = int(26000 * W * H / 1e6)
        x = rng.uniform(0, W, n)
        d = rng.random(n) ** 1.3
        yt = np.interp(x, xs, top)
        y = yt + d * (H - yt)
        stamp(img, x, y, shades(rng, SOIL, rng.random(n), 0.9) * g * (1 - 0.6 * d)[:, None],
              rng.choice([1.0, 1.5, 2.0], n, p=[0.4, 0.4, 0.2]))
        nb, ns = 300, 12
        bx = rng.uniform(0, W, nb)
        bd = rng.random(nb)
        by = np.interp(bx, xs, top) + bd * 0.10 * H
        h = (0.02 + 0.05 * rng.random(nb)) * H * (0.6 + 0.8 * bd)
        lean = rng.normal(0, 0.3, nb)
        s = np.tile(np.linspace(0, 1, ns), nb)
        b = np.repeat(np.arange(nb), ns)
        a = lean[b] * s ** 1.3
        col = shades(rng, GRASS, s, 0.4) * g * (0.5 + 0.5 * bd[b])[:, None]
        stamp(img, bx[b] + h[b] * s * np.sin(a), by[b] - h[b] * s * np.cos(a), col, 1.0 + 0.5 * (s < 0.5))

    # ---- the plant: build once ----------------------------------------------------------
    def build_stem(self):
        rng, n = self.rng, 3600
        s = rng.random(n)
        lat = sample_v(rng, n, 0.2)
        hair = rng.random(n) < 0.12
        lat[hair] = np.sign(lat[hair]) * rng.uniform(1.15, 1.45, int(hair.sum()))
        col = shades(rng, STEM, rng.random(n), 0.6)
        col[hair] = col[hair] * 0.6 + np.array([150, 170, 130]) * 0.4
        col[lat < -0.5] *= 1.18                                   # light down one side
        self.stem = dict(s=s, lat=lat, col=col, hair=hair,
                         rad=np.where(hair, 1.0, rng.choice([1.0, 1.5, 2.0], n, p=[0.3, 0.45, 0.25])))

    def build_leaves(self):
        rng = self.rng
        self.leaf = []
        for _ in LEAVES:
            n, npet = 650, 40
            u = np.r_[sample_u(rng, n, leaf_width), np.linspace(0, 1, npet)]
            v = np.r_[sample_v(rng, n, 0.15), rng.uniform(-0.3, 0.3, npet)]
            pet = np.r_[np.zeros(n, bool), np.ones(npet, bool)]
            av = np.abs(v)
            vein = ~pet & ((av < 0.05) | ((np.mod(u * 6 - av * 1.4, 1.0) < 0.10) & (av < 0.85)))
            live = shades(rng, LEAF, u * 0.7 + rng.normal(0, 0.12, n + npet), 0.5)
            live[vein] = shades(rng, LEAF_VEIN, rng.random(int(vein.sum())))
            live[pet] = shades(rng, STEM, rng.random(npet))
            self.leaf.append(dict(u=u, v=v, pet=pet, live=live, dead=shades(rng, LEAF_DEAD, rng.random(n + npet), 0.6),
                                  rad=rng.choice([1.0, 1.5, 2.0], n + npet, p=[0.3, 0.45, 0.25])))
        n = 240                                                   # the two seed leaves
        u, v = rng.random(n), sample_v(rng, n)
        self.cotyl = dict(u=u, v=v, side=np.where(np.arange(n) < n // 2, -1.0, 1.0),
                          col=shades(rng, LEAF[1:], rng.random(n)), rad=rng.choice([1.0, 1.5], n))

    def build_disc(self):
        rng = self.rng
        N = 1500
        i = np.arange(N)
        dark = ((i % 21) < 10) ^ ((i % 34) < 17)                 # the two families of spirals show in the seeds
        seed = np.where(dark[:, None], [[38, 34, 36]], [[118, 110, 100]]) * (1 + rng.normal(0, 0.08, (N, 1)))
        self.disc = dict(r=np.sqrt((i + 0.5) / N), th=i * GA, rn=rng.random(N),
                         bud=shades(rng, [(110, 120, 46), (130, 130, 50), (96, 110, 40)], rng.random(N)),
                         pollen=shades(rng, [(255, 206, 64), (250, 170, 40), (255, 226, 110)], rng.random(N)),
                         ripe=shades(rng, [(84, 44, 26), (104, 56, 34), (70, 36, 44)], rng.random(N)),
                         seed=seed.astype(np.float32), rad=rng.choice([1.5, 2.0, 2.5], N, p=[0.3, 0.5, 0.2]))

    def build_petals(self):
        rng, K, d = self.rng, 34, 110
        width = lambda u: np.sin(np.pi * np.clip(u, 0, 1) ** 0.55) ** 0.8
        parts = []
        self.petal_phi = 2 * np.pi * np.arange(K) / K + rng.normal(0, 0.03, K)
        for k in range(K):
            u, v = sample_u(rng, d, width), sample_v(rng, d, 0.15)
            live = shades(rng, PETAL, u * 0.85 + rng.normal(0, 0.1, d), 0.45)
            live[np.abs(np.mod(v * 3.5, 1.0) - 0.5) < 0.07] *= 0.9     # faint grooves
            parts.append(dict(k=np.full(d, k), u=u, v=v, inner=np.full(d, k % 2),
                              ln=np.full(d, 1 - 0.08 * (k % 2) + rng.normal(0, 0.04)), live=live,
                              dead=shades(rng, PETAL_DEAD, rng.random(d), 0.6),
                              rad=rng.choice([1.0, 1.5, 2.0], d, p=[0.3, 0.45, 0.25])))
        self.petals = join(parts)
        self.petal_fall = 44.0 + 8.0 * rng.permutation(K) / K + rng.uniform(0, 0.6, K)

    def build_bracts(self):
        rng, K, d = self.rng, 26, 55
        width = lambda u: np.sin(np.pi * np.clip(u, 0, 1) ** 0.65) ** 0.9 * (1 - 0.5 * np.clip(u, 0, 1) ** 3)
        parts = []
        for k in range(K):
            row = k % 2
            u, v = sample_u(rng, d, width), sample_v(rng, d)
            parts.append(dict(phi=np.full(d, 2 * np.pi * k / K + 0.12 * row), row=np.full(d, row),
                              ln=np.full(d, (0.42 if row == 0 else 0.32) * rng.uniform(0.9, 1.1)), u=u, v=v,
                              col=shades(rng, BRACT, u, 0.5), rad=rng.choice([1.0, 1.5], d)))
        self.bracts = join(parts)

    def build_back(self):
        rng, n = self.rng, 700
        self.back = dict(r=np.sqrt(rng.random(n)) * 1.03, th=rng.uniform(0, 2 * np.pi, n),
                         col=shades(rng, [(54, 90, 36), (70, 110, 44), (46, 78, 32)], rng.random(n)),
                         rad=rng.choice([1.5, 2.0], n))

    def build_bees(self):
        rng = self.rng
        self.bee = dict(om=rng.uniform(0.9, 1.5, 3) * rng.choice([-1, 1], 3), ph=rng.uniform(0, 6.28, 3),
                        p2=rng.uniform(0, 6.28, 3), p3=rng.uniform(0, 6.28, 3))

    def build_falls(self):
        """Where each petal and seed is when it lets go."""
        rng = self.rng
        K = len(self.petal_fall)
        sx, sy = np.zeros(K), np.zeros(K)
        for k in range(K):
            T = self.petal_fall[k]
            ph = phases(T)
            P, _ = self.stem_curve(T, ph)
            hp = self.head_pose(T, ph, P)
            rr = self.Rd + 0.5 * self.Lp
            sx[k], sy[k], _ = self.proj(hp, rr * math.cos(self.petal_phi[k]), rr * math.sin(self.petal_phi[k]), 0.0)
        self.pfall = dict(T=self.petal_fall, sx=sx, sy=sy, vx=self.W * rng.uniform(-0.03, 0.05, K),
                          ph=rng.uniform(0, 6.28, K), spin=rng.uniform(1.5, 3.5, K) * rng.choice([-1, 1], K),
                          land=self.y0 + rng.uniform(-0.02, 0.05, K) * self.H,
                          col=shades(rng, PETAL_DEAD[:2], rng.random(K * 30)).reshape(K, 30, 3))
        n = 160
        T = 50.0 + 9.0 * rng.random(n)
        sx, sy = np.zeros(n), np.zeros(n)
        for i in range(n):
            ph = phases(T[i])
            P, _ = self.stem_curve(T[i], ph)
            hp = self.head_pose(T[i], ph, P)
            r, a = self.Rd * math.sqrt(rng.random()), rng.uniform(0, 2 * np.pi)
            sx[i], sy[i], _ = self.proj(hp, r * math.cos(a), r * math.sin(a), 0.0)
        self.sfall = dict(T=T, sx=sx, sy=sy, land=self.y0 + rng.uniform(-0.03, 0.06, n) * self.H)

    # ---- the plant: pose ---------------------------------------------------------------------
    def stem_curve(self, t, ph):
        L = self.Ls * float(ph["grow"])
        sx, _, _ = self.sun_pos(min(t, 49.0))
        lean = 0.05 * L * float(np.clip((sx - self.cx) / (0.4 * self.W), -1, 1))
        bow = float(ph["bow"])
        sway = self.H * 0.004 * math.sin(0.8 * t)
        P = ((self.cx, self.y0 + 3),
             (self.cx - 0.02 * L, self.y0 - 0.35 * L),
             (self.cx + 0.5 * lean + 0.02 * L, self.y0 - 0.75 * L),
             (self.cx + lean + sway + 0.10 * L * bow, self.y0 - L * (1 - 0.06 * bow)))
        return P, L

    def head_pose(self, t, ph, P):
        nx, ny = P[3]
        sx, _, alt = self.sun_pos(min(t, 49.5))
        bow = float(ph["bow"])
        yaw = 1.05 * float(np.clip((sx - nx) / (0.38 * self.W), -1, 1)) * (1 - 0.5 * bow)
        pitch = 0.08 + 0.30 * alt * (1 - bow) - 1.30 * bow
        n = np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch), math.cos(yaw) * math.cos(pitch)])
        e1 = unit(np.cross([0.0, 1.0, 0.0], n))
        e2 = np.cross(n, e1)
        hs = 0.12 + 0.88 * float(ph["bud"])
        off = 0.35 * self.Rd * hs
        return dict(c=(nx + n[0] * off, ny - n[1] * off), n=n, e1=e1, e2=e2, hs=hs)

    @staticmethod
    def proj(hp, a, b, c):
        """Head-local (a along e1, b along e2, c along the facing direction) -> screen x, y, depth."""
        e1, e2, n = hp["e1"], hp["e2"], hp["n"]
        vx = a * e1[0] + b * e2[0] + c * n[0]
        vy = a * e1[1] + b * e2[1] + c * n[1]
        vz = a * e1[2] + b * e2[2] + c * n[2]
        return hp["c"][0] + vx, hp["c"][1] - vy, vz

    # ---- drawing --------------------------------------------------------------------------------
    def render(self, t):
        W, H = self.W, self.H
        ph = phases(t)
        img = self.backdrop.at(t)

        sx, sy, alt = self.sun_pos(t)
        vis = float(ss(t, 0.5, 2.0) * (1 - ss(t, 49.0, 51.0)))
        if vis > 0.01:
            warm = (1 - alt) ** 1.5
            core = np.array([255, 250, 236]) * (1 - warm) + np.array([255, 150, 80]) * warm
            gcol = (np.array([120, 112, 96]) * (1 - warm) + np.array([170, 80, 30]) * warm) * vis
            self.sun.draw(img, sx, sy, core, gcol, self.sky_mask, self.glow_mask)

        P, L = self.stem_curve(t, ph)
        hp = self.head_pose(t, ph, P)
        hx, hy = hp["c"]
        nt = float(ph["night"])
        ldir = unit(unit((sx - hx, sy - hy)) * (1 - nt) + unit((MOON_XY[0] * W - hx, MOON_XY[1] * H - hy)) * nt)
        light = (keyed(t, self.amb_keys), keyed(t, self.key_keys), ldir)

        dots, glows = [], []
        if L > 1:
            self.draw_stem(dots, t, P, light)
        self.draw_cotyledons(dots, t, light)
        self.draw_leaves(dots, t, ph, P, light)
        if ph["bud"] > 0.01:
            self.draw_head(dots, t, ph, hp, light)
        self.draw_bees(dots, glows, t, hp)
        self.draw_falls(dots, t, light)
        paint(img, dots)
        lights(img, glows)
        self.draw_meteor(img, t)
        return np.clip(img, 0, 255).astype(np.uint8)

    def draw_stem(self, dots, t, P, light):
        S = self.stem
        grow = min(1.0, (self.y0 - P[3][1]) / self.Ls)
        x, y, dx, dy = bezier(P, S["s"])
        n = np.hypot(dx, dy) + 1e-6
        w = 0.0105 * self.H * (1 - 0.35 * S["s"]) * (0.35 + 0.65 * min(1.0, grow * 1.6))
        lat = S["lat"] * w
        X, Y = x - dy / n * lat, y + dx / n * lat
        col = mix(S["col"], np.array([110, 96, 54], np.float32), 0.45 * float(ss(t, 46, 58)) * (1 - S["s"]))
        dots.append((X, Y, lit(col, X, Y, x, y, w, 0.5, light), S["rad"], np.where(S["hair"], 0.5, 0.0)))

    def draw_cotyledons(self, dots, t, light):
        sz = float(ss(t, 0.3, 3.0) * (1 - ss(t, 28.0, 34.0)))
        if sz < 0.02:
            return
        C = self.cotyl
        u, v = C["u"], C["v"]
        L = 0.035 * self.H * sz
        X = self.cx + C["side"] * (0.004 * self.H + u * L)
        Y = self.y0 - 0.02 * self.H * min(1.0, sz * 2) - u * 0.25 * L + v * 0.16 * L * np.sin(np.pi * u)
        col = mix(C["col"], np.array([170, 150, 60], np.float32), float(ss(t, 18.0, 30.0)))
        dots.append((X, Y, lit(col, X, Y, self.cx, self.y0, L + 1, 0.4, light), C["rad"], 1.0))

    def draw_leaves(self, dots, t, ph, P, light):
        grow = float(ph["grow"])
        up = np.array([0.0, 1.0, 0.0])
        for j, (s, side, size, az, roll) in enumerate(LEAVES):
            lf = float(ss(grow, s + 0.03, s + 0.20))
            if lf <= 0.01:
                continue
            lw = float(ss(t, 40 + 14 * s, 47 + 14 * s))           # the lowest leaves give up first
            ax, ay, _, _ = bezier(P, np.array([min(1.0, s / max(grow, 1e-3))]))
            ax, ay = float(ax[0]), float(ay[0])
            Lf = 0.15 * self.H * size * lf
            e0 = 0.55 + 0.45 * (1 - lf) - 0.55 * lw
            h = np.array([side * math.cos(az), 0.0, math.sin(az)])
            perp = np.array([-h[2], 0.0, h[0]])
            latv = math.cos(roll) * perp + math.sin(roll) * up
            dir0 = math.cos(e0) * h + math.sin(e0) * up
            Lv = self.leaf[j]
            u, v = Lv["u"], Lv["v"]
            base = 0.32 * Lf * dir0
            am = e0 - 0.35 - (0.9 + 0.8 * lw) * u * 0.5
            along = (Lf * u)[:, None] * (np.cos(am)[:, None] * h + np.sin(am)[:, None] * up)
            wid = 0.45 * Lf * leaf_width(u) * (1 - 0.35 * lw)
            pos = base + along + (v * wid)[:, None] * latv
            pos = np.where(Lv["pet"][:, None], (0.32 * Lf * u)[:, None] * dir0, pos)
            X, Y, Z = ax + pos[:, 0], ay - pos[:, 1], pos[:, 2] + 0.2
            mid = base + 0.5 * Lf * (math.cos(e0 - 0.6) * h + math.sin(e0 - 0.6) * up)
            col = lit(mix(Lv["live"], Lv["dead"], lw), X, Y, ax + mid[0], ay - mid[1], Lf * 0.6 + 1, 0.4, light)
            dots.append((X, Y, col, Lv["rad"], Z))

    def draw_head(self, dots, t, ph, hp, light):
        hs = hp["hs"]
        R = self.Rd * hs
        cx, cy = hp["c"]
        o, wilt, seed = float(ph["open"]), float(ph["wilt"]), float(ph["seed"])

        B = self.back                                              # the green back of the head
        r = B["r"] * R
        X, Y, Z = self.proj(hp, r * np.cos(B["th"]), r * np.sin(B["th"]), -0.12 * R - 0.06 * R * (1 - B["r"] ** 2))
        dots.append((X, Y, lit(B["col"], X, Y, cx, cy, R, 0.35, light), B["rad"], Z))

        D = self.disc                                              # the disc, blooming from the rim inward
        r = D["r"] * R
        X, Y, Z = self.proj(hp, r * np.cos(D["th"]), r * np.sin(D["th"]), 0.12 * R * (1 - D["r"] ** 2) + 0.02 * R)
        front = float(ph["disc"]) * 1.05 - (1 - D["r"])
        col = np.where((front > 0)[:, None], np.where((front < 0.16)[:, None], D["pollen"], D["ripe"]), D["bud"])
        col = mix(col, D["seed"], np.clip(seed * 1.4 - 0.4 * D["rn"], 0, 1))
        col = col * (0.72 + 0.28 * (1 - D["r"] ** 3))[:, None]
        dots.append((X, Y, lit(col, X, Y, cx, cy, R, 0.3, light), D["rad"], Z + 0.5))

        Bt = self.bracts                                           # green bracts: the bud, then a collar
        beta = 1.45 * (1 - o) - 0.55 * o - 0.35 * seed
        curl = 0.9 * (1 - o)
        u, v = Bt["u"], Bt["v"]
        Lb = self.Lp * hs * Bt["ln"] * (1.6 - 0.6 * o)
        bu = beta + curl * u
        rad_ = 0.93 * R + Lb * u * np.cos(bu)
        lat = v * 0.34 * Lb * np.sin(np.pi * np.clip(u, 0, 1) ** 0.65) ** 0.9
        cp, sp = np.cos(Bt["phi"]), np.sin(Bt["phi"])
        X, Y, Z = self.proj(hp, cp * rad_ - sp * lat, sp * rad_ + cp * lat, Lb * u * np.sin(bu) - 0.05 * R * Bt["row"] - 0.02 * R)
        col = mix(Bt["col"], np.array([128, 124, 64], np.float32), 0.5 * seed)
        dots.append((X, Y, lit(col, X, Y, cx, cy, R * 1.4, 0.35, light), Bt["rad"], Z))

        Pt = self.petals                                           # the rays, hidden in the bud until it opens
        keep = (t < self.petal_fall[Pt["k"]]) & (o > 0.01)
        if keep.any():
            g = {k: val[keep] for k, val in Pt.items()}
            u, v = g["u"], g["v"]
            beta = 1.35 * (1 - o) - 0.08 * o - 0.85 * wilt
            curl = 0.25 * (1 - o) - 0.7 * wilt
            ext = (0.15 + 0.85 * o) * (1 - 0.3 * wilt) * g["ln"]
            Lp = self.Lp * hs * ext
            bu = beta + curl * u
            rad_ = 0.97 * R + Lp * u * np.cos(bu)
            lat = v * 0.24 * self.Lp * hs * ext * np.sin(np.pi * np.clip(u, 0, 1) ** 0.55) ** 0.8
            phi = self.petal_phi[g["k"]]
            cp, sp = np.cos(phi), np.sin(phi)
            C = Lp * u * np.sin(bu) + 0.04 * R * g["inner"] + 0.03 * R * v ** 2
            X, Y, Z = self.proj(hp, cp * rad_ - sp * lat, sp * rad_ + cp * lat, C)
            col = mix(g["live"], g["dead"], np.clip(wilt * 1.3 - 0.3 * g["inner"], 0, 1))
            col = mix(col, np.array([150, 160, 60], np.float32), 0.6 * (1 - o) ** 3)    # green-gold as they emerge
            col = col * (0.8 + 0.2 * np.clip(Z / R * 0.5 + 0.5, 0, 1))[:, None]
            dots.append((X, Y, lit(col, X, Y, cx, cy, R * 1.6, 0.45, light), g["rad"], Z))

    def bee_xy(self, t, hp, i):
        B = self.bee
        R = self.Rd * hp["hs"]
        cx, cy = hp["c"]
        a = B["om"][i] * t + B["ph"][i]
        rb = R * (1.6 + 0.5 * math.sin(0.6 * t + B["ph"][i]))
        ox, oy = cx + rb * math.cos(a), cy + rb * 0.8 * math.sin(1.3 * a + B["p2"][i])
        land = float(ss(math.sin(0.33 * t + B["p3"][i]), 0.55, 0.85))
        da = 0.4 * t + B["ph"][i]
        dx, dy, _ = self.proj(hp, 0.55 * R * math.cos(da), 0.55 * R * math.sin(da), 0.15 * R)
        return ox + (dx - ox) * land, oy + (dy - oy) * land

    def draw_bees(self, dots, glows, t, hp):
        act = float(ss(t, 22.0, 25.0) * (1 - ss(t, 40.0, 43.0)))
        if act < 0.01:
            return
        s = 0.0042 * self.H
        cols = np.array([[34, 26, 20], [240, 190, 40], [34, 26, 20], [240, 196, 50], [30, 24, 20]], np.float32)
        for i in range(3):
            x0, y0 = self.bee_xy(t, hp, i)
            x1, y1 = self.bee_xy(t - 0.05, hp, i)
            fx, fy = self.W * (1.1 if i % 2 else -0.1), hp["c"][1] - 0.2 * self.H * (i - 1)
            x, y = fx + (x0 - fx) * act, fy + (y0 - fy) * act
            hd = unit((x0 - x1, y0 - y1 + 1e-3))
            k = np.arange(-2, 3) * s
            dots.append((x + hd[0] * k, y + hd[1] * k, cols, np.full(5, 0.0022 * self.H + 0.5), 1e5))
            pv = np.array([-hd[1], hd[0]])
            flick = 0.6 + 0.4 * math.sin(70 * t + i)
            wx = x + np.array([1, -1]) * pv[0] * 2.4 * s - hd[0] * 0.4 * s
            wy = y + np.array([1, -1]) * pv[1] * 2.4 * s - hd[1] * 0.4 * s - 0.8 * s
            glows.append((wx, wy, np.array([210, 220, 240]) * 0.45 * flick, 2.6))

    def draw_falls(self, dots, t, light):
        F = self.pfall
        f = t - F["T"]
        m = f > 0
        if m.any():
            f, idx = f[m], np.nonzero(m)[0]
            g = 0.035 * self.H
            tl = np.sqrt(2 * np.maximum(F["land"][idx] - F["sy"][idx], 1) / g)
            fe = np.minimum(f, tl)
            air = f < tl
            x = F["sx"][idx] + F["vx"][idx] * fe + 0.02 * self.W * np.sin(2 * fe + F["ph"][idx]) * air
            y = np.minimum(F["sy"][idx] + 0.5 * g * fe ** 2, F["land"][idx])
            a = F["ph"][idx] + F["spin"][idx] * fe
            wid = 0.16 * self.Lp * np.where(air, 0.25 + 0.75 * np.abs(np.cos(1.7 * fe + F["ph"][idx])), 1.0)
            uu = np.tile(np.linspace(-0.45, 0.45, 10), 3) * self.Lp * 0.8
            vv = np.repeat([-0.45, 0.0, 0.45], 10) * np.cos(np.tile(np.linspace(-1.2, 1.2, 10), 3))
            X = x[:, None] + np.cos(a)[:, None] * uu - np.sin(a)[:, None] * vv * wid[:, None]
            Y = y[:, None] + np.sin(a)[:, None] * uu * np.where(air, 1.0, 0.35)[:, None] + np.cos(a)[:, None] * vv * wid[:, None]
            col = F["col"][idx].reshape(-1, 3)
            z = np.repeat(np.where(air, 9e3, -5.0), 30)
            dots.append((X.ravel(), Y.ravel(), lit(col, X.ravel(), Y.ravel(), X.ravel(), Y.ravel(), 1.0, 0.3, light), 1.6, z))
        S = self.sfall
        f = t - S["T"]
        m = f > 0
        if m.any():
            g = 0.09 * self.H
            y = np.minimum(S["sy"][m] + 0.5 * g * f[m] ** 2, S["land"][m])
            dots.append((S["sx"][m], y, np.array([30, 28, 30], np.float32), 1.5, 9e3))

    def draw_meteor(self, img, t):
        a = (t - 56.4) / 0.9
        if not 0 <= a <= 1:
            return
        W, H = self.W, self.H
        hx, hy = W * (0.72 - 0.30 * a), H * (0.07 + 0.12 * a)
        tail = np.arange(40) / 40
        b = (1 - tail) ** 1.5 * math.sin(math.pi * a) * 0.9
        glow(img, hx + tail * 0.13 * W, hy - tail * 0.052 * H, np.array([255, 250, 235]) * b[:, None], 1.2)


if __name__ == "__main__":
    run(Sunflower, "Sunflower - one summer day", "sunflower", T_END, STAGES)
