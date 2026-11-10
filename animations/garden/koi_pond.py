"""
koi_pond.py - a koi pond seen from above, in dots. A seamless one-minute loop.

Sunlight wobbles in a net of light over the pebbles on the bottom while nine koi -
kohaku, sanke, showa, ogon, platinum, tancho, asagi, chagoi - cruise in slow loops,
their shadows gliding beneath them. Water lilies open among the pads. A handful of food
lands on the water and the koi swarm in, gulping at the surface, then drift apart again.
A maple leaf spins down and floats, a dragonfly skims across and taps the surface,
clouds slide over the reflection, and the lilies close as the minute comes round.

    python koi_pond.py            the pond, looping forever (Space pause, R restart, S save frame, Esc quit)
    python koi_pond.py --png      save five moments
    python koi_pond.py --gif      save one seamless loop as a GIF
"""
import math

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))   # the shared engine: engine/pointillism.py

from pointillism import glow, mix, paint, run, sample_u, sample_v, shades, ss, stamp, unit

T_END = 60.0
TAU = 2 * math.pi
STAGES = [6.0, 14.0, 23.0, 41.5, 50.0]
WATER = np.array([30, 82, 88], np.float32)
SHADOW = np.array(unit((0.72, 0.62)))          # the sun is up and to the left; shadows fall this way

WHITE, RED, BLACK = (246, 242, 234), (224, 64, 34), (26, 24, 28)
FIN = dict(kohaku=WHITE, sanke=WHITE, showa=(60, 56, 60), tancho=WHITE, ogon=(240, 196, 90),
           platinum=(236, 236, 228), asagi=(226, 118, 60), chagoi=(176, 128, 78))
KOI = [  # variety, length (fraction of the frame), laps a minute (sign = direction), depth
    ("kohaku", 0.135, 2, 0.35), ("sanke", 0.120, -2, 0.55), ("showa", 0.125, 1, 0.45),
    ("ogon", 0.118, 2, 0.30), ("tancho", 0.105, -3, 0.40), ("asagi", 0.115, 2, 0.60),
    ("chagoi", 0.150, -1, 0.65), ("kohaku", 0.098, 3, 0.25), ("platinum", 0.108, -2, 0.50),
]
SHY = {5, 8}                                    # these two never come to the food

PAD_G = [(46, 96, 44), (62, 118, 50), (84, 138, 58), (108, 152, 68)]
PAD_B = [(110, 64, 48), (132, 84, 56), (150, 104, 64)]
PADS = [  # x, y, radius, notch angle, bronze?, flower
    (0.24, 0.30, 0.056, 0.6, 0, "pink"), (0.33, 0.24, 0.044, 2.2, 0, None), (0.19, 0.44, 0.040, 4.0, 1, None),
    (0.76, 0.73, 0.060, 3.5, 0, "white"), (0.82, 0.60, 0.046, 5.2, 0, None), (0.66, 0.79, 0.042, 1.2, 1, None),
    (0.42, 0.80, 0.038, 0.2, 0, None), (0.76, 0.30, 0.042, 2.8, 0, None), (0.22, 0.67, 0.050, 5.8, 0, "pink"),
]
LILY_RINGS = [(12, 1.00, 0.0), (10, 0.82, 0.3), (8, 0.62, 0.15)]
STONE = [(118, 116, 110), (138, 134, 126), (100, 98, 94), (150, 140, 124), (126, 118, 104)]
PEBBLE = [(132, 130, 124), (164, 146, 116), (152, 104, 72), (72, 72, 70), (192, 186, 176), (110, 118, 96)]
GRASS = [(46, 86, 38), (66, 116, 48), (96, 138, 58), (38, 66, 30)]


def body_width(s):
    """Half-width of a koi seen from above (fraction of its length); s runs from nose (0) to tail root."""
    b = np.clip(s / 0.86, 0, 1) * 0.93
    return 0.125 * np.sin(np.pi * np.clip(b, 1e-3, 1) ** 0.62) ** 0.85 * (1 - 0.3 * b ** 2)


def koi_colors(kind, s, v, rng):
    n = len(s)
    ph = rng.uniform(0, TAU, 6)
    blob = np.sin(7.5 * s + 2.2 * v + ph[0]) + 0.7 * np.sin(13 * s - 3.1 * v + ph[1]) + 0.5 * np.sin(21 * s + 1.3 * v + ph[2])
    spot = np.sin(31 * s + 9 * v + ph[3]) * np.sin(23 * s - 11 * v + ph[4])
    net = np.sin(s * 75) * np.sin(v * 13 + 0.5) > 0.45                # the scale net of asagi and chagoi
    c = np.empty((n, 3), np.float32)
    every = np.ones(n, bool)
    if kind in ("kohaku", "sanke"):
        c[every] = WHITE
        c[(blob > 0.2) & (s > 0.04)] = RED
        if kind == "sanke":
            c[(spot > 0.62) & (s > 0.18) & (np.abs(v) < 0.9)] = BLACK
    elif kind == "showa":
        c[every] = BLACK
        c[blob > 0.15] = RED
        c[(blob < -0.8) & (s > 0.12)] = WHITE
    elif kind == "tancho":
        c[every] = WHITE
        c[((s - 0.12) / 0.07) ** 2 + (v / 0.55) ** 2 < 1] = RED
    elif kind == "ogon":
        c[every] = (240, 188, 66)
    elif kind == "platinum":
        c[every] = (232, 232, 222)
    elif kind == "asagi":
        c[every] = (98, 120, 146)
        c[net] = (150, 170, 190)
        c[(np.abs(v) > 0.62) | (s < 0.05)] = (226, 110, 52)
    elif kind == "chagoi":
        c[every] = (170, 120, 70)
        c[net] = (196, 150, 96)
    metal = 1.0 if kind in ("ogon", "platinum") else 0.0
    sheen = 1 + 0.06 * np.sin(s * 90) * np.cos(v * 9) + 0.12 * metal * (1 - np.abs(v))
    return c * sheen[:, None] * (1 + rng.normal(0, 0.03, (n, 1)))


class KoiPond:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.S = min(W, H)
        self.rng = np.random.default_rng(8)
        self.pc = np.array([0.5 * W, 0.52 * H])
        self.R0 = 0.44 * self.S
        self.food_c = np.array([0.62 * W, 0.40 * H])
        self.build_base()
        self.build_caustics()
        self.build_koi()
        self.build_pads()
        self.build_food()
        self.build_leaves()
        self.build_events()
        self.build_fx()

    # ---- the pond itself ---------------------------------------------------------------------
    def rim(self, ang):
        return self.R0 * (1 + 0.08 * np.sin(2 * ang + 1.0) + 0.05 * np.sin(3 * ang + 2.0) + 0.03 * np.sin(5 * ang + 0.4))

    def polar(self, x, y):
        dx, dy = x - self.pc[0], y - self.pc[1]
        return np.hypot(dx, dy), np.arctan2(dy, dx)

    def depth(self, x, y):
        r, a = self.polar(x, y)
        return np.clip(1 - (r / self.rim(a)) ** 2, 0, 1)

    def build_base(self):
        W, H, S = self.W, self.H, self.S
        rng = np.random.default_rng(12)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r, a = self.polar(xx, yy)
        rr = self.rim(a)
        inside = r < rr
        d = np.clip(1 - (r / rr) ** 2, 0, 1)
        self.inside = inside
        img = np.empty((H, W, 3), np.float32)
        img[:] = np.array([40, 56, 32], np.float32)

        # grass and clover around the pond
        n = int(70000 * W * H / 1e6)
        gx, gy = rng.uniform(0, W, n), rng.uniform(0, H, n)
        ang = rng.uniform(0, TAU, n)
        ln = rng.uniform(2, 6, n) * S / 1000
        col = shades(rng, GRASS, rng.random(n), 0.8)
        for k in range(3):
            stamp(img, gx + np.cos(ang) * ln * k / 2, gy + np.sin(ang) * ln * k / 2, col * (0.85 + 0.1 * k), 1.0)
        m = rng.random(n) < 0.012
        stamp(img, gx[m], gy[m], np.where(rng.random(int(m.sum()))[:, None] < 0.7, [[236, 236, 226]], [[240, 210, 90]]), 1.5)

        # the floor: sand at the edges, darker and greener in the deep middle, seen through tinted water
        floor = np.array([150, 138, 108], np.float32) * (1 - d)[..., None] + np.array([84, 92, 78], np.float32) * d[..., None]
        tint = (0.42 + 0.38 * d)[..., None]
        img[inside] = (floor * (1 - tint) + WATER * tint)[inside]
        ys, xs = np.nonzero(inside)
        n = int(len(xs) * 0.012)
        pick = rng.choice(len(xs), n, replace=False)
        px, py = xs[pick] + rng.random(n), ys[pick] + rng.random(n)
        pd = self.depth(px, py)[:, None]
        pt = 0.42 + 0.38 * pd
        pc = shades(rng, PEBBLE, rng.random(n), 0.9)
        pr = rng.choice([1.5, 2.0, 2.5, 3.0], n, p=[0.35, 0.35, 0.2, 0.1])
        stamp(img, px + 0.8, py + 0.8, (pc * 0.55) * (1 - pt) + WATER * 0.7 * pt, pr)
        stamp(img, px, py, pc * (1 - pt) + WATER * pt, pr)
        stamp(img, px - pr * 0.35, py - pr * 0.35, (pc * 1.25 + 20) * (1 - pt) + WATER * pt, 1.0)
        # weeds in the shallows
        for _ in range(34):
            while True:
                ax, ay = rng.uniform(0, W), rng.uniform(0, H)
                if 0.05 < self.depth(ax, ay) < 0.4:
                    break
            for _ in range(rng.integers(4, 8)):
                a0, cu = rng.uniform(0, TAU), rng.normal(0, 1.2)
                s = np.linspace(0, 1, 30)
                L = rng.uniform(0.03, 0.07) * S
                ang = a0 + cu * s
                wx = ax + np.cumsum(np.cos(ang)) * L / 30
                wy = ay + np.cumsum(np.sin(ang)) * L / 30
                wd = self.depth(wx, wy)[:, None]
                wc = shades(rng, [(40, 80, 40), (56, 100, 48), (70, 112, 52)], rng.random(30)) * (1 - 0.5 * wd) + WATER * 0.5 * wd
                stamp(img, wx, wy, wc, 1.5)
        # the bank shades the water just inside it on the sunny side
        edge = inside & (r > rr - 0.035 * S)
        facing = ((xx - self.pc[0]) * -SHADOW[0] + (yy - self.pc[1]) * -SHADOW[1]) / np.maximum(r, 1)
        shade = np.clip(1 - (rr - r) / (0.035 * S), 0, 1) * np.clip(facing, 0, 1) * 0.45
        img[edge] *= (1 - shade[edge])[:, None]

        # stones around the rim, with their shadows
        nr = 56
        ra = np.sort(rng.uniform(0, TAU, nr))
        rpos = self.rim(ra) * rng.uniform(1.0, 1.08, nr)
        rocks = []
        for k in range(nr):
            cx, cy = self.pc[0] + np.cos(ra[k]) * rpos[k], self.pc[1] + np.sin(ra[k]) * rpos[k]
            rx = rng.uniform(0.028, 0.062) * S
            rocks.append((cx, cy, rx, rx * rng.uniform(0.65, 1.0), rng.uniform(0, np.pi), rng.integers(len(STONE)),
                          rng.random() < 0.35))
        for cx, cy, rx, ry, rot, *_ in rocks:                        # shadows first
            n = int(rx * ry * 1.2)
            a, rr_ = rng.uniform(0, TAU, n), np.sqrt(rng.random(n))
            lx, ly = np.cos(a) * rr_ * rx, np.sin(a) * rr_ * ry
            sx = cx + lx * math.cos(rot) - ly * math.sin(rot) + SHADOW[0] * 0.016 * S
            sy = cy + lx * math.sin(rot) + ly * math.cos(rot) + SHADOW[1] * 0.016 * S
            ix, iy = np.clip(sx.astype(int), 0, W - 1), np.clip(sy.astype(int), 0, H - 1)
            stamp(img, sx, sy, img[iy, ix] * 0.55, 2.0)
        L3 = unit((-0.6, -0.5, 0.62))
        for cx, cy, rx, ry, rot, ci, mossy in rocks:
            n = int(rx * ry * 1.6)
            a, rr_ = rng.uniform(0, TAU, n), np.sqrt(rng.random(n))
            ex, ey = np.cos(a) * rr_, np.sin(a) * rr_
            hz = np.sqrt(np.clip(1 - rr_ ** 2, 0, 1))                   # a domed top
            nx, ny = ex * math.cos(rot) - ey * math.sin(rot), ex * math.sin(rot) + ey * math.cos(rot)
            lum = np.clip(nx * L3[0] + ny * L3[1] + hz * L3[2], 0, 1) * 0.85 + 0.25
            col = np.asarray(STONE[ci], np.float32) * (1 + rng.normal(0, 0.07, (n, 1)))
            if mossy:
                moss = (ny < -0.2) & (rng.random(n) < 0.6)
                col[moss] = shades(rng, [(84, 116, 52), (104, 132, 60)], rng.random(int(moss.sum())))
            lx, ly = ex * rx, ey * ry
            stamp(img, cx + lx * math.cos(rot) - ly * math.sin(rot), cy + lx * math.sin(rot) + ly * math.cos(rot),
                  col * lum[:, None], rng.choice([1.0, 1.5, 2.0], n, p=[0.3, 0.45, 0.25]))
        self.base = img

    def build_caustics(self):
        rng = self.rng
        ys, xs = np.nonzero(self.inside)
        n = int(len(xs) * 0.05)
        pick = rng.choice(len(xs), n, replace=False)
        x, y = xs[pick] + rng.random(n), ys[pick] + rng.random(n)
        k = TAU / (0.085 * self.S)
        self.ca = dict(x=x, y=y, u=x * k, v=y * k, d=self.depth(x, y))

    # ---- koi --------------------------------------------------------------------------------
    def build_koi(self):
        rng, S = self.rng, self.S
        self.koi = []
        for i, (kind, ln, laps, d0) in enumerate(KOI):
            nb, nt, npf, npl, nd = 900, 200, 60, 26, 40
            sb = 0.86 * sample_u(rng, nb, lambda u: body_width(u * 0.86) / 0.125)
            vb = sample_v(rng, nb, 0.1)
            st = 0.84 + 0.24 * sample_u(rng, nt, lambda u: 0.25 + 0.75 * u)
            vt = rng.uniform(-1, 1, nt)
            st[(st > 1.0) & (np.abs(vt) < 0.45 * (st - 1.0) / 0.08)] = 0.98   # the forked tail
            fin = np.asarray(FIN[kind], np.float32)
            tail = fin * (1 + rng.normal(0, 0.05, (nt, 1)))
            tail[np.abs(np.mod(vt * 4, 1.0) - 0.5) < 0.12] *= 0.82           # fin rays
            K = dict(kind=kind, L=ln * S, laps=laps, d0=d0, joins=i not in SHY,
                     sb=sb, vb=vb, body=koi_colors(kind, sb, vb, rng),
                     st=st, vt=vt, tail=tail,
                     uf=rng.random(2 * npf), cf=rng.uniform(-1, 1, 2 * npf), sf=np.repeat([-1.0, 1.0], npf),
                     pect=fin * (1 + rng.normal(0, 0.05, (2 * npf, 1))),
                     ul=rng.random(2 * npl), cl=rng.uniform(-1, 1, 2 * npl), sl=np.repeat([-1.0, 1.0], npl),
                     pelv=fin * (1 + rng.normal(0, 0.05, (2 * npl, 1))),
                     sd=rng.uniform(0.30, 0.72, nd), vd=rng.uniform(-0.12, 0.12, nd),
                     # a closed path, travelled a whole number of times a minute, so the loop is seamless
                     cx=self.pc[0] + rng.uniform(-0.04, 0.04) * S, cy=self.pc[1] + rng.uniform(-0.04, 0.04) * S,
                     A=rng.uniform(0.13, 0.21) * S, C=rng.uniform(0.13, 0.21) * S,
                     B=rng.uniform(0.02, 0.07) * S, D=rng.uniform(0.02, 0.07) * S,
                     n2=int(rng.choice([2, 3])), n3=int(rng.choice([2, 3])), p=rng.uniform(0, TAU, 3),
                     fr=rng.uniform(0.06, 0.11) * S, fv=rng.uniform(0.09, 0.13) * S, fdir=rng.choice([-1.0, 1.0]),
                     fph=rng.uniform(0, TAU), fdelay=rng.uniform(0, 1.6),
                     dm=int(rng.integers(1, 3)), dph=rng.uniform(0, TAU),
                     om=TAU * int(rng.integers(40, 61)) / T_END)
            K["dors"] = koi_colors(kind, K["sd"], K["vd"], rng) * 0.8
            self.koi.append(K)
        self.koi.sort(key=lambda K: -K["d0"])                            # deepest drawn first

    def feed_w(self, K, t):
        t = np.asarray(t, np.float64)
        if not K["joins"]:
            return np.zeros_like(t)
        dl = K["fdelay"]
        return ss(t, 19.0 + dl, 22.0 + dl) * (1 - ss(t, 29.5 + dl, 33.0 + dl))

    def koi_xy(self, K, t):
        """Where a koi's nose is at time(s) t."""
        t = np.asarray(t, np.float64)
        th = TAU * K["laps"] * t / T_END + K["p"][2]
        x = K["cx"] + K["A"] * np.cos(th) + K["B"] * np.cos(K["n2"] * th + K["p"][0])
        y = K["cy"] + K["C"] * np.sin(th) + K["D"] * np.sin(K["n3"] * th + K["p"][1])
        f = self.feed_w(K, t)
        if np.any(f > 0):                                               # circling the food
            ang = K["fdir"] * K["fv"] / K["fr"] * t + K["fph"]
            fx = self.food_c[0] + K["fr"] * np.cos(ang)
            fy = self.food_c[1] + K["fr"] * np.sin(ang)
            x, y = x + (fx - x) * f, y + (fy - y) * f
        return x, y

    def koi_depth(self, K, t):
        d = np.clip(K["d0"] + 0.2 * math.sin(TAU * K["dm"] * t / T_END + K["dph"]), 0.08, 1.0)
        return d * (1 - 0.85 * float(self.feed_w(K, t)))                # up at the surface to feed

    def spine(self, K, t):
        """The body follows the path the nose has just swum: sample it back by arc length."""
        L = K["L"]
        tau = np.linspace(0.0, 6.0, 120)
        x, y = self.koi_xy(K, t - tau)
        seg = np.hypot(np.diff(x), np.diff(y)) + 1e-6
        cl = np.r_[0.0, np.cumsum(seg)]
        need = 1.12 * L
        if cl[-1] < need:                                               # nearly still: carry on straight back
            dx, dy = (x[-1] - x[-2]) / seg[-1], (y[-1] - y[-2]) / seg[-1]
            extra = need - cl[-1] + 1
            x, y, cl = np.r_[x, x[-1] + dx * extra], np.r_[y, y[-1] + dy * extra], np.r_[cl, cl[-1] + extra]
        sj = np.linspace(0, 1.12, 48) * L
        xs, ys = np.interp(sj, cl, x), np.interp(sj, cl, y)
        tx, ty = np.gradient(xs), np.gradient(ys)
        nrm = np.hypot(tx, ty) + 1e-9
        return xs, ys, tx / nrm, ty / nrm

    @staticmethod
    def on_spine(sp, s):
        xs, ys, tx, ty = sp
        k = np.clip(s, 0, 1.12) / 1.12 * (len(xs) - 1)
        i = np.arange(len(xs))
        x, y, ux, uy = np.interp(k, i, xs), np.interp(k, i, ys), np.interp(k, i, tx), np.interp(k, i, ty)
        return x, y, -uy, ux, ux, uy                                     # point, normal, tangent (towards the tail)

    def draw_koi(self, dots, t):
        S, base = self.S, self.base
        for i, K in enumerate(self.koi):
            L = K["L"]
            d = self.koi_depth(K, t)
            sp = self.spine(K, t)
            wave = lambda s: L * 0.05 * np.clip(s, 0, None) ** 1.7 * np.sin(TAU * 0.85 * s - K["om"] * t)
            tint = lambda c, extra=0.0: c * (0.95 - 0.42 * d - extra) + WATER * (0.05 + 0.5 * d + extra)

            px, py, nx, ny, _, _ = self.on_spine(sp, K["sb"])            # body
            off = K["vb"] * body_width(K["sb"]) * L + wave(K["sb"])
            bx, by = px + nx * off, py + ny * off
            px, py, nx, ny, _, _ = self.on_spine(sp, K["st"])            # tail fin
            hw = L * (0.015 + 0.14 * np.clip((K["st"] - 0.84) / 0.24, 0, 1) ** 0.9)
            off = K["vt"] * hw + wave(K["st"])
            tx_, ty_ = px + nx * off, py + ny * off
            fins_x, fins_y, fins_c = [], [], []
            for s0, Lf, wf, beta0, us, cs, sides, cols in ((0.22, 0.13, 0.05, 0.75, K["uf"], K["cf"], K["sf"], K["pect"]),
                                                          (0.52, 0.065, 0.025, 1.0, K["ul"], K["cl"], K["sl"], K["pelv"])):
                bx0, by0, bnx, bny, btx, bty = self.on_spine(sp, np.array([s0]))
                beta = beta0 + 0.3 * np.sin(TAU * 1.5 * t + sides)          # paddling
                ox, oy = sides * bnx, sides * bny
                dx, dy = np.cos(beta) * ox + np.sin(beta) * btx, np.cos(beta) * oy + np.sin(beta) * bty
                root = body_width(s0) * L * 0.9
                fx = bx0 + ox * root + dx * us * Lf * L - dy * cs * wf * L * np.sin(np.pi * us ** 0.7)
                fy = by0 + oy * root + dy * us * Lf * L + dx * cs * wf * L * np.sin(np.pi * us ** 0.7)
                fins_x.append(fx)
                fins_y.append(fy)
                fins_c.append(cols)
            fx, fy, fc = np.concatenate(fins_x), np.concatenate(fins_y), np.concatenate(fins_c)
            px, py, nx, ny, _, _ = self.on_spine(sp, K["sd"])            # the ridge of the back
            dx_, dy_ = px + nx * K["vd"] * L * 0.03, py + ny * K["vd"] * L * 0.03
            ex, ey, enx, eny, _, _ = self.on_spine(sp, np.array([0.07, 0.07]))   # eyes
            eo = np.array([-1.0, 1.0]) * body_width(0.07) * L * 0.62
            z = -100 * d + i * 0.01

            # its shadow on the bottom, further off the shallower it swims
            sh = SHADOW * (0.010 + 0.03 * (1 - d)) * S
            sx, sy = np.r_[bx[::2], tx_[::2]] + sh[0], np.r_[by[::2], ty_[::2]] + sh[1]
            ix, iy = np.clip(sx.astype(int), 0, self.W - 1), np.clip(sy.astype(int), 0, self.H - 1)
            dots.append((sx, sy, base[iy, ix] * 0.58, 2.0, -1e6))

            dots.append((fx, fy, tint(mix(fc, WATER, 0.35)), 1.3, z - 0.5))
            dots.append((tx_, ty_, tint(mix(K["tail"], WATER, 0.3)), 1.3, z - 0.4))
            dots.append((bx, by, tint(K["body"]), 1.6, z))
            dots.append((dx_, dy_, tint(K["dors"]), 1.2, z + 0.2))
            dots.append((ex + enx * eo, ey + eny * eo, tint(np.array([18, 18, 20], np.float32)), 1.4, z + 0.3))

    # ---- pads, lilies, food, leaves -----------------------------------------------------------
    def build_pads(self):
        rng, S = self.rng, self.S
        self.pads = []
        for x, y, R, notch, bronze, flower in PADS:
            R = R * S
            n = int(1500 * (R / (0.05 * S)) ** 2)
            r = np.sqrt(rng.random(n))
            th = rng.uniform(0, TAU, n)
            dth = np.angle(np.exp(1j * (th - notch)))
            cut = (np.abs(dth) < 0.2) & (r > 0.08)                      # the V-shaped notch
            th[cut] = notch + np.sign(dth[cut] + 1e-9) * rng.uniform(0.2, 0.3, int(cut.sum()))
            col = shades(rng, PAD_B if bronze else PAD_G, 0.25 + 0.6 * r + rng.normal(0, 0.1, n), 0.5)
            vein = np.abs(np.sin((th - notch) * 8)) < 0.07
            col[vein] = col[vein] * 1.22 + 10
            col[r > 0.93] = col[r > 0.93] * np.array([1.15, 1.1, 0.8])
            P = dict(x=x * self.W, y=y * self.H, R=R, r=r, th=th, col=col, ph=rng.uniform(0, TAU, 3),
                     drop=(rng.uniform(0.2, 0.7, 2), rng.uniform(0, TAU, 2), rng.integers(20, 60, 2), rng.uniform(0, TAU, 2)))
            if flower:
                rings = []
                for ring, (cnt, ln, off) in enumerate(LILY_RINGS):
                    m = 26
                    u = sample_u(rng, cnt * m, lambda u: np.sin(np.pi * np.clip(u, 0, 1) ** 0.75) ** 0.8)
                    v = rng.uniform(-1, 1, cnt * m)
                    k = np.repeat(np.arange(cnt), m)
                    phi = TAU * k / cnt + off + rng.normal(0, 0.04, cnt * m)
                    if flower == "pink":
                        c = shades(rng, [(252, 226, 234), (246, 182, 208), (234, 118, 166)], u * 0.9 + 0.1 * ring, 0.3)
                    else:
                        c = shades(rng, [(250, 240, 200), (252, 250, 240), (255, 255, 252)], u, 0.3)
                    rings.append(dict(u=u, v=v, phi=phi, ln=ln, col=c, ring=ring))
                P["lily"] = rings
                P["lily_o"] = rng.uniform(0, 1.5)
                ns = 90
                P["stamen"] = (np.sqrt(rng.random(ns)), rng.uniform(0, TAU, ns),
                               shades(rng, [(255, 206, 70), (250, 170, 40), (255, 226, 120)], rng.random(ns)))
            self.pads.append(P)

    def pad_pose(self, P, t):
        a = TAU * t / T_END
        cx = P["x"] + 0.006 * self.S * math.sin(a + P["ph"][0])
        cy = P["y"] + 0.006 * self.S * math.cos(a + P["ph"][1])
        return cx, cy, 0.08 * math.sin(a + P["ph"][2])

    def draw_pads(self, dots, glows, t):
        S, base = self.S, self.base
        for P in self.pads:
            cx, cy, rot = self.pad_pose(P, t)
            th = P["th"] + rot
            x, y = cx + np.cos(th) * P["r"] * P["R"], cy + np.sin(th) * P["r"] * P["R"]
            sx, sy = x[::2] + SHADOW[0] * 0.022 * S, y[::2] + SHADOW[1] * 0.022 * S
            ix, iy = np.clip(sx.astype(int), 0, self.W - 1), np.clip(sy.astype(int), 0, self.H - 1)
            dots.append((sx, sy, base[iy, ix] * 0.6, 2.0, -1e6))
            dots.append((x, y, P["col"], 1.6, 1e4))
            rr, ra, w, ph = P["drop"]                                    # beads of water that catch the sun
            dx, dy = cx + np.cos(ra + rot) * rr * P["R"], cy + np.sin(ra + rot) * rr * P["R"]
            dots.append((dx, dy, np.array([200, 222, 220], np.float32), 1.8, 1e4 + 0.5))
            tw = np.clip(np.sin(TAU * w * t / T_END + ph), 0, 1) ** 10
            glows.append((dx - 0.7, dy - 0.7, np.array([255, 255, 255]) * (0.3 + 0.8 * tw)[:, None], 1.6))
            if "lily" in P:
                o = float(ss(t, 3.0 + P["lily_o"], 11.0 + P["lily_o"]) * (1 - ss(t, 50.0 + P["lily_o"], 57.0 + P["lily_o"])))
                fx, fy = cx + 0.25 * P["R"] * math.cos(rot + 1), cy + 0.25 * P["R"] * math.sin(rot + 1)
                Lf = 0.036 * S
                for ring in P["lily"]:
                    vis = Lf * ring["ln"] * (0.22 + 0.78 * o) * (1.0 if ring["ring"] == 0 else 0.6 + 0.4 * o)
                    ph_ = ring["phi"] + rot
                    along, across = ring["u"] * vis, ring["v"] * 0.30 * vis * np.sin(np.pi * np.clip(ring["u"], 0, 1) ** 0.75) ** 0.8
                    X = fx + np.cos(ph_) * along - np.sin(ph_) * across
                    Y = fy + np.sin(ph_) * along + np.cos(ph_) * across
                    col = ring["col"]
                    if ring["ring"] == 0:                                # green-backed sepals show while closed
                        col = mix(col, np.array([110, 140, 80], np.float32), 0.6 * (1 - o) ** 2)
                    dots.append((X, Y, col, 1.4, 2e4 + ring["ring"]))
                if o > 0.2:
                    sr, sa, sc = P["stamen"]
                    rs = sr * 0.24 * Lf * o
                    dots.append((fx + np.cos(sa) * rs, fy + np.sin(sa) * rs, sc, 1.3, 2e4 + 5))

    def build_food(self):
        rng, S = self.rng, self.S
        n = 56
        hand = rng.random(n) < 0.5                                       # two handfuls
        land = np.where(hand, 18.0 + 0.5 * rng.random(n), 19.2 + 0.5 * rng.random(n))
        pos = self.food_c + rng.normal(0, 0.032 * S, (n, 2)) + np.where(hand[:, None], [[-0.015 * S, 0.0]], [[0.02 * S, 0.01 * S]])
        self.food = dict(land=land, x=pos[:, 0], y=pos[:, 1], eaten=21.0 + 8.5 * rng.random(n) ** 0.8,
                         ph=rng.uniform(0, TAU, n))

    def draw_food(self, dots, t):
        F, S = self.food, self.S
        m = (t >= F["land"] - 0.5) & (t < F["eaten"])
        if not m.any():
            return
        e = np.clip((t - F["land"][m] + 0.5) / 0.5, 0, 1)               # dropping in from above, shrinking
        x = F["x"][m] + 0.004 * S * np.sin(0.7 * t + F["ph"][m])
        y = F["y"][m] + 0.004 * S * np.cos(0.6 * t + F["ph"][m])
        rad = 1.8 * (1 + 2.0 * (1 - e))
        dots.append((x, y, np.array([138, 92, 46], np.float32), rad, 3e4))
        dots.append((x - 0.6, y - 0.6, np.array([196, 156, 104], np.float32), 1.0, 3e4 + 1))

    def build_leaves(self):
        rng, S = self.rng, self.S
        self.leaves = []
        for T0, (x0, y0), (vx, vy), pal in ((33.0, (0.30, 0.56), (0.004, 0.002), [(206, 56, 34), (226, 82, 38)]),
                                            (39.5, (0.58, 0.68), (-0.003, 0.003), [(232, 120, 40), (242, 150, 50)]),
                                            (46.0, (0.46, 0.28), (0.003, -0.002), [(236, 178, 56), (246, 200, 80)])):
            n = 320
            phi = rng.uniform(0, TAU, n)
            rr = np.sqrt(rng.random(n)) * (0.45 + 0.55 * np.abs(np.cos(2.5 * phi)) ** 1.6)   # five lobes
            col = shades(rng, pal, rng.random(n), 0.6)
            col[(np.abs(np.sin(2.5 * phi)) < 0.08) & (rr < 0.9)] *= 0.72                 # veins
            stem = np.linspace(0.2, 1.25, 12)
            self.leaves.append(dict(T0=T0, x=x0 * self.W, y=y0 * self.H, vx=vx * S, vy=vy * S, phi=phi, r=rr, col=col,
                                    stem=stem, rot0=rng.uniform(0, TAU), spin=rng.choice([-1.0, 1.0]) * rng.uniform(2, 4),
                                    sx=rng.uniform(-0.15, 0.15) * S, sy=rng.uniform(-0.15, 0.05) * S))

    def leaf_state(self, Lf, t):
        """Falling (towards the water, so shrinking), then floating, then sinking away before the loop ends."""
        fall = 2.2
        if t < Lf["T0"] or t > 57.5:
            return None
        e = min((t - Lf["T0"]) / fall, 1.0)
        if e < 1:
            k = float(ss(e, 0, 1))
            x, y = Lf["x"] + Lf["sx"] * (1 - k), Lf["y"] + Lf["sy"] * (1 - k)
            rot = Lf["rot0"] + Lf["spin"] * (t - Lf["T0"])
            return x, y, rot, 1 + 1.6 * (1 - k) ** 1.5, (1 - k), 1.0
        f = t - Lf["T0"] - fall
        rot = Lf["rot0"] + Lf["spin"] * fall + 0.06 * f
        return Lf["x"] + Lf["vx"] * f, Lf["y"] + Lf["vy"] * f, rot, 1.0, 0.0, float(1 - ss(t, 53.0, 57.5))

    def draw_leaves(self, dots, t):
        S, base = self.S, self.base
        Rl = 0.026 * S
        for Lf in self.leaves:
            st = self.leaf_state(Lf, t)
            if st is None:
                continue
            x, y, rot, scale, height, keep = st
            r = Lf["r"] * Rl * scale
            X, Y = x + np.cos(Lf["phi"] + rot) * r, y + np.sin(Lf["phi"] + rot) * r
            sa = rot + np.pi
            X = np.r_[X, x + np.cos(sa) * Lf["stem"] * Rl * scale]
            Y = np.r_[Y, y + np.sin(sa) * Lf["stem"] * Rl * scale]
            col = np.r_[Lf["col"], np.tile([[120, 70, 40]], (len(Lf["stem"]), 1))].astype(np.float32)
            col = mix(col, WATER * 0.9, 1 - keep)
            sh = SHADOW * (0.02 + 0.2 * height) * S                       # its shadow closes in as it lands
            sx, sy = X[::2] + sh[0], Y[::2] + sh[1]
            ix, iy = np.clip(sx.astype(int), 0, self.W - 1), np.clip(sy.astype(int), 0, self.H - 1)
            dots.append((sx, sy, base[iy, ix] * (0.6 + 0.3 * height), 2.0, -1e6))
            dots.append((X, Y, col, 1.6 * min(scale, 1.8), 1e6 if height > 0 else 3e4))

    # ---- the dragonfly ----------------------------------------------------------------------------
    FLY = [(40.0, (1.10, 0.18)), (41.2, (0.62, 0.30)), (42.4, (0.62, 0.30)), (43.0, (0.47, 0.52)),
           (44.2, (0.47, 0.52)), (44.8, (0.33, 0.62)), (46.0, (0.33, 0.62)), (47.2, (-0.12, 0.74))]

    def fly_at(self, t):
        ts = [w[0] for w in self.FLY]
        if not ts[0] <= t <= ts[-1]:
            return None
        k = min(int(np.searchsorted(ts, t, side="right")) - 1, len(ts) - 2)
        a, b = np.array(self.FLY[k][1]), np.array(self.FLY[k + 1][1])
        e = float(ss(t, ts[k], ts[k + 1]))
        p = (a + (b - a) * e) * np.array([self.W, self.H])
        p += 0.003 * self.S * np.array([math.sin(13 * t), math.cos(11 * t)])
        j = k
        while j > 0 and np.allclose(self.FLY[j][1], self.FLY[j + 1][1]):
            j -= 1                                                       # hovering: keep the last heading
        hd = unit(np.array(self.FLY[j + 1][1]) - np.array(self.FLY[j][1]) + 1e-6)
        return p, hd

    def draw_fly(self, dots, glows, t):
        st = self.fly_at(t)
        if st is None:
            return
        S, base = self.S, self.base
        p, hd = st
        pv = np.array([-hd[1], hd[0]])
        seg = 0.0058 * S
        k = np.arange(-1, 11)
        bx, by = p[0] - hd[0] * k * seg, p[1] - hd[1] * k * seg
        col = np.where((k <= 0)[:, None], [[40, 92, 112]], np.where((k % 3 == 0)[:, None], [[24, 56, 76]], [[62, 156, 206]]))
        rad = np.where(k <= 0, 2.4, np.clip(1.8 - 0.06 * k, 1.1, 1.8))
        ex, ey = p[0] + hd[0] * seg * 1.6 + pv[0] * np.array([-1, 1]) * seg * 0.9, p[1] + hd[1] * seg * 1.6 + pv[1] * np.array([-1, 1]) * seg * 0.9
        wing_x, wing_y = [], []
        flick = 0.7 + 0.3 * math.sin(80 * t)
        for side in (-1, 1):
            for fwd, ln in ((0.25, 0.044), (-0.2, 0.040)):
                wd = unit(pv * side + hd * fwd)
                u = np.linspace(0.1, 1.0, 12)
                base_pt = p - hd * seg * (0.4 if fwd > 0 else 1.2)
                for w_off in (-0.18, 0.18):
                    wing_x.append(base_pt[0] + wd[0] * u * ln * S + hd[0] * w_off * seg * np.sin(np.pi * u))
                    wing_y.append(base_pt[1] + wd[1] * u * ln * S + hd[1] * w_off * seg * np.sin(np.pi * u))
        wx, wy = np.concatenate(wing_x), np.concatenate(wing_y)
        glows.append((wx, wy, np.array([215, 230, 240]) * 0.5 * flick, 1.6))
        sh = SHADOW * 0.06 * S                                          # its shadow on the bottom
        sx, sy = np.r_[bx, wx[::2]] + sh[0], np.r_[by, wy[::2]] + sh[1]
        ix, iy = np.clip(sx.astype(int), 0, self.W - 1), np.clip(sy.astype(int), 0, self.H - 1)
        dots.append((sx, sy, base[iy, ix] * 0.65, 1.6, -1e6))
        dots.append((np.r_[bx, ex], np.r_[by, ey], np.r_[col, [[70, 170, 160], [70, 170, 160]]].astype(np.float32),
                     np.r_[rad, 2.6, 2.6], 1e7))

    # ---- the surface ------------------------------------------------------------------------------
    def build_events(self):
        """Every ring on the water: food landing, gulps, leaves, the dragonfly, koi kissing the surface."""
        rng, F = self.rng, self.food
        ev = [(T, x, y, 0.35, 1.0) for T, x, y in zip(F["land"], F["x"], F["y"])]
        ev += [(T, x, y, 1.0, 1.8) for T, x, y in zip(F["eaten"], F["x"], F["y"])]
        ev += [(Lf["T0"] + 2.2, Lf["x"], Lf["y"], 0.7, 2.0) for Lf in self.leaves]
        ev.append((44.2, 0.47 * self.W, 0.52 * self.H, 0.9, 2.0))
        for k, T in enumerate((3.5, 9.0, 13.5, 36.5, 49.0, 54.0)):
            x, y = self.koi_xy(self.koi[(3 * k + 1) % len(self.koi)], T)
            ev.append((T, float(x), float(y), 0.8, 2.0))
        for T in (7.2, 25.5, 51.5):                                     # drips from the trees
            r, a = np.sqrt(rng.random()) * self.R0 * 0.8, rng.uniform(0, TAU)
            ev.append((T, self.pc[0] + r * math.cos(a), self.pc[1] + r * math.sin(a), 0.5, 1.6))
        ev = np.array(ev, np.float64)
        self.events = dict(t=ev[:, 0], x=ev[:, 1], y=ev[:, 2], s=ev[:, 3], life=ev[:, 4])

    def build_fx(self):
        rng, S = self.rng, self.S
        ys, xs = np.nonzero(self.inside)
        pick = rng.choice(len(xs), 300, replace=False)
        self.glints = dict(x=xs[pick] + 0.5, y=ys[pick] + 0.5, k=rng.integers(20, 61, 300), ph=rng.uniform(0, TAU, 300))
        self.clouds = []
        for x0, y0 in ((0.1, 0.30), (0.6, 0.62), (1.05, 0.45)):
            n = 1800
            k = rng.integers(0, 5, n)
            bx, by = rng.normal(0, 0.06, 5) * S, rng.normal(0, 0.025, 5) * S
            a, r = rng.uniform(0, TAU, n), np.sqrt(rng.random(n)) * 0.05 * S
            self.clouds.append(dict(x0=x0 * S, y=y0 * self.H + by[k] + np.sin(a) * r * 0.6, dx=bx[k] + np.cos(a) * r * 1.4))

    def draw_surface(self, img, glows, t):
        S, W, H = self.S, self.W, self.H
        E = self.events                                                  # rings: light, with a dark trough outside
        age = t - E["t"]
        m = (age >= 0) & (age < E["life"])
        if m.any():
            a = np.linspace(0, TAU, 40, endpoint=False)
            ag, s = age[m], E["s"][m]
            b = s * (1 - ag / E["life"][m]) ** 1.6
            for scale, amp in ((1.0, 0.38), (0.62, 0.22), (1.12, -0.16)):
                r = (2 + 0.07 * S * ag * (0.6 + 0.4 * s)) * scale
                x = E["x"][m][:, None] + r[:, None] * np.cos(a)
                y = E["y"][m][:, None] + r[:, None] * np.sin(a)
                glow(img, x.ravel(), y.ravel(), np.array([220, 236, 240]) * amp * np.repeat(b, 40)[:, None], 1.3)
        for C in self.clouds:                                            # clouds drifting over the reflection
            x = (C["x0"] + 1.5 * S * t / T_END + C["dx"]) % (1.5 * S) - 0.25 * S
            ix, iy = np.clip(x.astype(int), 0, W - 1), np.clip(C["y"].astype(int), 0, H - 1)
            ok = self.inside[iy, ix] & (x >= 0) & (x < W)
            glow(img, x[ok], C["y"][ok], np.array([255, 255, 255]) * 0.05, 3.0)
        G = self.glints
        tw = np.clip(np.sin(TAU * G["k"] * t / T_END + G["ph"]), 0, 1) ** 14
        on = tw > 0.02
        glow(img, G["x"][on], G["y"][on], np.array([255, 255, 240]) * 0.9 * tw[on][:, None], 1.5)
        for x, y, c, r in glows:
            glow(img, x, y, c, r)

    # ---- one frame ----------------------------------------------------------------------------------
    def render(self, t):
        t = t % T_END
        img = self.base.copy()
        C = self.ca                                                     # the net of light on the bottom
        a = TAU * t / T_END
        s1 = np.sin(C["u"] + 6 * a + 1.4 * np.sin(0.8 * C["v"] - 4 * a))
        s2 = np.sin(1.1 * C["v"] - 5 * a + 1.4 * np.sin(0.9 * C["u"] + 3 * a))
        c = np.clip(1 - np.abs(s1 + s2) * 0.9, 0, 1) ** 3
        on = c > 0.03
        glow(img, C["x"][on], C["y"][on], np.array([255, 250, 214]) * (0.48 * c[on] * (1 - 0.4 * C["d"][on]))[:, None], 1.0)

        dots, glows = [], []
        self.draw_koi(dots, t)
        self.draw_pads(dots, glows, t)
        self.draw_food(dots, t)
        self.draw_leaves(dots, t)
        self.draw_fly(dots, glows, t)
        paint(img, dots)
        self.draw_surface(img, glows, t)
        return np.clip(img, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    run(KoiPond, "Koi pond", "koi_pond", T_END, STAGES, wrap=True)
