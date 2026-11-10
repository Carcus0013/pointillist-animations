"""
diwali.py - Diwali night in a courtyard, seen from straight above, in dots.

Dusk falls on a stone courtyard. In the middle a rangoli pours itself out of coloured powder,
ring by ring from the centre - a lotus, a band of kolam dots, a wreath of leaves, twelve great
petals, a scalloped border - while Lakshmi's footprints appear at the door. Then the lamps:
seventy-odd clay diyas catch one after another in chasing waves, round the rangoli, along the
ledges, in the brass urli where they float among marigolds; strings of fairy lights blink on
overhead and a paper star lantern sways in the corner. Two children run out with sparklers and
write loops of light in the air, chakris spin and spit spiral sparks, anar fountains roar up at
us, and a handful of marigold petals is flung high and comes down everywhere. Then the rangoli
itself rises: its powder lifts off as glowing motes and spirals up past us - and bursts overhead
as fireworks in its own colours, each flash washing the courtyard pink, green, gold. Embers rain
down onto the pattern and it glows, ring after ring, as the smoke drifts and the lamps burn on.

    python diwali.py            interactive window (Space pause, R restart, S save frame, Esc quit)
    python diwali.py --png      save five stages
    python diwali.py --gif      save the whole night as a GIF
"""
import math

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))   # the shared engine: engine/pointillism.py

from pointillism import Haze, glow, keyed, run, ss, stamp

T_END = 60.0
STAGES = [9.0, 22.5, 33.0, 44.8, 51.3]
TAU = 2 * math.pi
FIELD = 5.0                      # metres across the frame
HC = 7.0                         # camera height above the floor
CX = CY = 2.5
RC = (2.5, 2.45)                 # the rangoli's centre
RR = 1.58                        # and its radius

C = dict(magenta=(230, 30, 130), pink=(255, 96, 176), saffron=(255, 128, 12), yellow=(255, 202, 34), turq=(0, 186, 196),
         green=(40, 172, 72), lime=(150, 230, 110), blue=(42, 72, 210), purple=(116, 42, 176), deep=(78, 24, 128),
         white=(250, 246, 236), red=(216, 30, 42), gold=(255, 196, 90))
AMB = [(0.0, (0.46, 0.46, 0.64)), (8.0, (0.30, 0.30, 0.47)), (16.0, (0.14, 0.14, 0.26)), (24.0, (0.05, 0.05, 0.11)),
       (60.0, (0.05, 0.05, 0.11))]
WARM = np.array([1.05, 0.66, 0.34], np.float32)
RINGS = [  # inner, outer radius (m), when the powder is poured
    (0.00, 0.12, 2.0, 3.0), (0.12, 0.42, 3.0, 5.5), (0.42, 0.50, 5.5, 6.5), (0.50, 0.90, 6.5, 9.5), (0.90, 0.97, 9.5, 10.2),
    (0.97, 1.42, 10.2, 14.0), (1.42, 1.52, 14.0, 15.5), (1.52, 1.58, 15.5, 16.5),
]
BURSTS = [  # time, where (m), height (m), two colours, size
    (46.0, (2.5, 2.45), 3.6, "magenta", "pink", 1.0), (47.2, (1.5, 1.5), 3.2, "turq", "lime", 0.9),
    (48.1, (3.6, 3.4), 3.4, "saffron", "yellow", 1.0), (49.0, (1.3, 3.7), 3.0, "blue", "white", 0.85),
    (49.8, (3.7, 1.3), 3.5, "red", "gold", 1.0), (50.7, (2.5, 2.45), 3.9, "purple", "pink", 1.25),
    (51.6, (1.8, 2.9), 3.3, "green", "gold", 0.95), (52.5, (3.2, 2.0), 3.6, "pink", "white", 1.0),
    (53.6, (2.5, 2.45), 4.1, "gold", "white", 1.5),
]


def hash01(a, b):
    return (np.sin(a * 12.9898 + b * 78.233) * 43758.5453) % 1.0


class Diwali:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.S = S = min(W, H)
        self.P = S / FIELD
        self.haze = Haze(W, H, max(2, round(4 * S / 1000)))
        v = np.arange(4096, dtype=np.float32) * (1.25 / 255.0)
        self.lut = np.clip(255.0 * v * (1 + v / 6.0) / (1 + v), 0, 255).astype(np.uint8)
        self.rng = np.random.default_rng(10)
        self.build_floor()
        self.build_rangoli()
        self.build_diyas()
        self.build_lights()
        self.build_motion()

    def proj(self, X, Y, Z=0.0):
        """Straight down from HC metres: things higher up look bigger and lie further from the centre."""
        k = HC / (HC - np.minimum(np.asarray(Z, np.float64), HC - 0.3))
        return (CX + (np.asarray(X) - CX) * k) * self.P, (CY + (np.asarray(Y) - CY) * k) * self.P

    # ---- the courtyard ----------------------------------------------------------------------------
    def build_floor(self):
        W, H, P = self.W, self.H, self.P
        rng = np.random.default_rng(3)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        X, Y = (xx + 0.5) / P, (yy + 0.5) / P
        ti, tj = np.floor(X / 0.5), np.floor(Y / 0.5)
        hs = hash01(ti, tj)
        pal = np.array([(196, 150, 112), (186, 138, 104), (206, 162, 122), (176, 130, 98)], np.float32)
        img = pal[(hs * 4).astype(int) % 4] * (0.92 + 0.12 * rng.random((H, W, 1)).astype(np.float32))
        grout = ((X % 0.5) < 0.012) | ((Y % 0.5) < 0.012)
        img[grout] *= 0.62
        ledge = (Y < 0.34) | (X < 0.32) | (X > 4.68) | ((Y > 4.74) & ((X < 1.8) | (X > 3.2)))
        img[ledge] = np.array([150, 112, 92], np.float32) * (0.85 + 0.25 * rng.random((int(ledge.sum()), 1)))
        lip = ((np.abs(Y - 0.35) < 0.018) & (X > 0.32) & (X < 4.68)) | ((np.abs(X - 0.33) < 0.018) & (Y > 0.34)) | \
            ((np.abs(X - 4.67) < 0.018) & (Y > 0.34)) | ((np.abs(Y - 4.73) < 0.018) & ((X < 1.8) | (X > 3.2)) & (X > 0.32) & (X < 4.68))
        img[lip] = np.array([96, 70, 58], np.float32)
        door = (Y > 4.74) & (X >= 1.8) & (X <= 3.2)                            # the carved wooden threshold
        img[door] = np.array([112, 64, 40], np.float32) * (0.8 + 0.3 * (np.sin(X[door] * 90) > 0.2))[:, None]
        self.floor = img
        self.objects(img, rng)

    def put(self, img, x, y, col, rad):
        stamp(img, np.asarray(x) * self.P, np.asarray(y) * self.P, col, rad)

    def objects(self, img, rng):
        P = self.P
        r1 = max(1.0, 1.4 * self.S / 1000)
        # the tulsi planter, top left: whitewashed, painted, with the holy basil growing in it
        cx, cy = 0.80, 0.82
        n = 9000
        u, v = rng.uniform(-1, 1, (2, n))
        x, y = cx + u * 0.26, cy + v * 0.26
        inner = (np.abs(u) < 0.6) & (np.abs(v) < 0.6)
        col = np.where(inner[:, None], np.array([86, 58, 40], np.float32), np.array([230, 220, 198], np.float32))
        band = (np.maximum(np.abs(u), np.abs(v)) > 0.78) & (np.maximum(np.abs(u), np.abs(v)) < 0.9)
        motif = band & (np.sin((u + v) * 30) > 0)
        col[band] = np.array(C["red"], np.float32)
        col[motif] = np.array(C["yellow"], np.float32)
        self.put(img, x, y, col, r1)
        m = 900
        a, rr = rng.uniform(0, TAU, m), 0.17 * np.sqrt(rng.random(m))
        leaf = np.array([(40, 110, 50), (60, 140, 60), (80, 110, 80), (90, 60, 100)], np.float32)[rng.integers(0, 4, m)]
        self.put(img, cx + np.cos(a) * rr, cy + np.sin(a) * rr, leaf * (0.8 + 0.4 * rng.random((m, 1))), r1 * 1.3)
        # a brass thali with sweets, top right
        cx, cy = 4.22, 0.86
        n = 5000
        a, rr = rng.uniform(0, TAU, n), 0.24 * np.sqrt(rng.random(n))
        brass = np.array([214, 172, 82], np.float32) * (0.75 + 0.35 * (rr / 0.24) ** 4 + 0.08 * rng.random(n))[:, None]
        self.put(img, cx + np.cos(a) * rr, cy + np.sin(a) * rr, brass, r1)
        for k in range(6):                                                    # laddoos and barfi
            ang = k * TAU / 6 + 0.3
            bx, by = cx + 0.13 * math.cos(ang), cy + 0.13 * math.sin(ang)
            if k % 2:
                q = rng.random((2, 120)) - 0.5
                self.put(img, bx + q[0] * 0.06, by + q[1] * 0.06, np.array([(236, 236, 220), (150, 200, 120), (240, 170, 190)][k // 2], np.float32),
                         r1)
            else:
                a2, r2 = rng.uniform(0, TAU, 160), 0.035 * np.sqrt(rng.random(160))
                c2 = np.array([246, 150, 40], np.float32) * (0.85 + 0.3 * rng.random((160, 1)))
                self.put(img, bx + np.cos(a2) * r2, by + np.sin(a2) * r2, c2, r1)
        a2, r2 = rng.uniform(0, TAU, 80), 0.022 * np.sqrt(rng.random(80))
        self.put(img, cx + np.cos(a2) * r2, cy + 0.02 + np.sin(a2) * r2, np.array(C["red"], np.float32), r1)
        # the brass urli, bottom right: water, floating marigolds and rose petals
        cx, cy = 4.30, 4.28
        n = 14000
        a, rr = rng.uniform(0, TAU, n), 0.40 * np.sqrt(rng.random(n))
        rim = rr > 0.35
        col = np.where(rim[:, None], np.array([220, 176, 84], np.float32), np.array([18, 52, 64], np.float32))
        col *= (0.8 + 0.3 * rng.random((n, 1)))
        self.put(img, cx + np.cos(a) * rr, cy + np.sin(a) * rr, col, r1)
        for k in range(7):
            ang, d = rng.uniform(0, TAU), rng.uniform(0.05, 0.27)
            fx, fy = cx + d * math.cos(ang), cy + d * math.sin(ang)
            a2, r2 = rng.uniform(0, TAU, 200), 0.05 * np.sqrt(rng.random(200))
            c2 = np.array([C["saffron"], C["yellow"]][k % 2], np.float32) * (0.75 + 0.4 * rng.random((200, 1)))
            self.put(img, fx + np.cos(a2) * r2, fy + np.sin(a2) * r2, c2, r1)
        q = rng.random((2, 70))
        self.put(img, cx + (q[0] - 0.5) * 0.5, cy + (q[1] - 0.5) * 0.5, np.array([200, 20, 40], np.float32), r1)
        # garlands: marigold swags along the top ledge, a toran of marigolds and mango leaves over the door
        xs = np.arange(0.40, 4.62, 0.035)
        sag = 0.06 * np.abs(np.sin((xs - 0.40) / 0.52 * math.pi))
        balls = np.array([C["saffron"], C["yellow"]], np.float32)[(np.arange(len(xs)) % 2)]
        for j in range(3):
            self.put(img, xs + rng.normal(0, 0.004, len(xs)), 0.385 + sag + rng.normal(0, 0.006, len(xs)), balls * (0.8 + 0.1 * j), r1 * 2.2)
        xs = np.arange(1.82, 3.19, 0.03)
        balls = np.array([C["saffron"], C["yellow"]], np.float32)[(np.arange(len(xs)) % 2)]
        self.put(img, xs, np.full(len(xs), 4.76), balls, r1 * 2.4)
        for x0 in np.arange(1.86, 3.18, 0.11):
            q = np.linspace(0, 1, 18)
            for w in np.linspace(-1, 1, 5):
                self.put(img, x0 + w * 0.02 * (1 - q), 4.78 + 0.09 * q, np.array([50, 130, 50], np.float32), r1)

    # ---- the rangoli -------------------------------------------------------------------------------------
    def build_rangoli(self):
        P, W, H = self.P, self.W, self.H
        rng = np.random.default_rng(5)
        x0, x1 = int((RC[0] - RR - 0.03) * P), int((RC[0] + RR + 0.03) * P) + 1
        y0, y1 = int((RC[1] - RR - 0.03) * P), int((RC[1] + RR + 0.03) * P) + 1
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx, dy = (xx + 0.5) / P - RC[0], (yy + 0.5) / P - RC[1]
        r = np.hypot(dx, dy)
        th = np.arctan2(dy, dx) % TAU
        col = np.zeros(r.shape + (3,), np.float32)
        tdr = np.full(r.shape, np.inf, np.float32)
        frac = ((th + math.pi / 2) % TAU) / TAU                              # poured clockwise from the top
        c = lambda k: np.array(C[k], np.float32)
        rs = np.maximum(r, 1e-3)

        def ang(n):
            per = TAU / n
            return ((th + per / 2) % per) - per / 2, per

        def ring(k, m, value):
            r0, r1_, t0, t1 = RINGS[k]
            col[m] = value[m] if value.ndim == 3 else value
            tdr[m] = (t0 + (t1 - t0) * frac)[m]

        w = np.ones(r.shape + (3,), np.float32)
        # the centre
        m = r < 0.12
        ring(0, m, np.where((r < 0.05)[..., None], c("red"), c("yellow")) * w)
        # an eight-petalled lotus
        m = (r >= 0.12) & (r < 0.42)
        s = (r - 0.12) / 0.30
        d, per = ang(8)
        pw = (per / 2) * 0.95 * np.maximum(np.sin(np.pi * np.clip(s * 0.9 + 0.05, 0, 1)), 0) ** 0.65
        pet = np.abs(d) < pw
        edge = pet & (np.abs(d) > pw - 0.013 / rs)
        v = c("magenta") + (c("pink") - c("magenta")) * s[..., None]
        v = np.where(pet[..., None], v, c("saffron") * w)
        v[edge] = c("white")
        ring(1, m, v)
        # a band of kolam dots
        m = (r >= 0.42) & (r < 0.50)
        d, per = ang(24)
        dot = np.hypot(d * rs, r - 0.46) < 0.019
        ring(2, m, np.where(dot[..., None], c("white"), c("turq")) * w)
        # a wreath of sixteen leaves
        m = (r >= 0.50) & (r < 0.90)
        s = (r - 0.50) / 0.40
        d, per = ang(16)
        lw = (per / 2) * 0.9 * np.maximum(np.sin(np.pi * np.clip(s, 0, 1)), 0) ** 0.8
        leaf = np.abs(d) < lw
        v = np.where(leaf[..., None], c("green") * (0.85 + 0.3 * s[..., None]), c("blue") * w)
        v[leaf & (np.abs(d) * rs < 0.005)] = c("lime")
        v[leaf & (np.abs(d) > lw - 0.012 / rs)] = c("white")
        ring(3, m, v)
        m = (r >= 0.90) & (r < 0.97)
        ring(4, m, c("white") * w)
        # twelve great petals, saffron at the heart, hot pink at the tips, with a yellow petal inside each
        m = (r >= 0.97) & (r < 1.42)
        s = (r - 0.97) / 0.45
        d, per = ang(12)
        pw = (per / 2) * 0.98 * np.maximum(np.sin(np.pi * np.clip(s * 0.92 + 0.04, 0, 1)), 0) ** 0.55
        pet = np.abs(d) < pw
        inner = pet & (np.abs(d) < pw * 0.5) & (s < 0.72)
        v = c("saffron") + (c("pink") - c("saffron")) * np.clip(s * 1.3, 0, 1)[..., None]
        v = np.where(pet[..., None], v, c("deep") * w)
        v[inner] = c("yellow")
        v[pet & (np.abs(d) > pw - 0.013 / rs)] = c("white")
        v[inner & (np.abs(d) > pw * 0.5 - 0.009 / rs)] = c("red")
        d2, per2 = ang(12)
        dmid = ((th) % per2) - per2 / 2
        v[(~pet) & (np.hypot(dmid * rs, r - 1.30) < 0.022)] = c("white")
        ring(5, m, v)
        # a scalloped border, yellow on red
        m = (r >= 1.42) & (r < 1.52)
        d, per = ang(36)
        sc = np.hypot(d * rs, r - 1.42) < 0.075
        ring(6, m, np.where(sc[..., None], c("yellow"), c("red")) * w)
        # a ring of white kolam dots
        d, per = ang(60)
        m = (r >= 1.52) & (r < 1.58) & (np.hypot(d * rs, r - 1.55) < 0.017)
        ring(7, m, c("white") * w)
        # powder: grainy, slightly uneven, a few bright grains
        g = 0.86 + 0.22 * rng.random(r.shape).astype(np.float32)
        col *= g[..., None]
        sparkle = rng.random(r.shape) < 0.012
        col[sparkle] = np.minimum(col[sparkle] * 1.5 + 30, 255)
        self.rg = dict(box=(y0, y1, x0, x1), col=col, t=tdr, r=r, th=th)
        # Lakshmi's footprints, walking in from the door
        feet = []
        for k, (fx, fy, side) in enumerate(((2.43, 4.60, -1), (2.57, 4.47, 1), (2.43, 4.34, -1))):
            m = 260
            u, v = rng.random(m), rng.uniform(-1, 1, m)
            wdt = 0.022 * (0.7 + 0.3 * np.sin(np.pi * u)) * (1 - 0.3 * (u < 0.25))
            x = fx + v * wdt + side * 0.004 * u
            y = fy - 0.09 * u
            toes = rng.integers(0, 5, 80)
            tx = fx + (toes - 2) * 0.009 * (-side) + rng.normal(0, 0.002, 80)
            ty = fy - 0.103 - np.abs(toes - 2) * 0.004 * 0.5 + rng.normal(0, 0.002, 80)
            col = np.vstack([np.where((np.abs(v) > 0.8)[:, None], c("white"), c("red")) * np.ones((m, 1)),
                             np.broadcast_to(c("red"), (80, 3))])
            feet.append((np.concatenate([x, tx]), np.concatenate([y, ty]), col, 12.5 + 1.1 * k))
        self.feet = feet

    # ---- diyas -------------------------------------------------------------------------------------
    def build_diyas(self):
        rng = np.random.default_rng(8)
        D = []                                                                # x, y, direction, size, when lit, group
        for k in range(16):
            a = k * TAU / 16 - math.pi / 2
            D.append((RC[0] + 1.72 * math.cos(a), RC[1] + 1.72 * math.sin(a), a, 1.0, 16.6 + 2.4 * k / 16, 0))
        for i in range(16):
            D.append((0.45 + 0.27 * i, 0.17, -math.pi / 2, 0.95, 18.0 + 2.2 * i / 15, 1))
        for i in range(13):
            y = 0.65 + 0.30 * i
            D.append((0.16, y, math.pi, 0.95, 18.6 + 2.6 * i / 12, 2))
            D.append((4.84, y, 0.0, 0.95, 18.6 + 2.6 * i / 12, 2))
        for j, x in enumerate((0.42, 0.72, 1.02, 1.32, 1.62, 3.38, 3.68, 3.98, 4.28, 4.58)):
            D.append((x, 4.87, math.pi / 2, 0.95, 20.2 + 1.8 * j / 9, 3))
        D += [(1.68, 4.58, math.pi / 2, 1.4, 20.0, 3), (3.32, 4.58, math.pi / 2, 1.4, 20.0, 3),
              (0.80, 1.20, math.pi / 2, 1.0, 19.0, 4), (4.22, 0.70, -math.pi / 2, 0.9, 19.4, 4),
              (4.20, 4.20, 0.6, 0.8, 21.0, 4), (4.42, 4.30, 2.0, 0.8, 21.3, 4), (4.30, 4.45, -1.4, 0.8, 21.6, 4)]
        self.diya = np.array(D, np.float64)
        self.d_ph = rng.uniform(0, TAU, (2, len(D)))
        # their clay bodies, painted on the floor
        img = self.floor
        r1 = max(1.0, 1.3 * self.S / 1000)
        for x, y, a, sz, _, _ in D:
            n = int(260 * sz ** 2)
            u = rng.random(n)
            ang_ = rng.uniform(0, TAU, n)
            rr = 0.055 * sz * np.sqrt(u)
            px, py = x + np.cos(ang_) * rr, y + np.sin(ang_) * rr
            col = np.where((rr > 0.040 * sz)[:, None], np.array([176, 82, 44], np.float32), np.array([66, 38, 22], np.float32))
            if rng.random() < 0.6:                                            # painted rims
                paint = (rr > 0.044 * sz) & (np.sin(ang_ * 8) > 0.4)
                col[paint] = np.array([C["yellow"], C["turq"], C["pink"], C["white"]][int(rng.integers(0, 4))], np.float32)
            m = 50                                                            # the pinched spout
            q = rng.random(m)
            sx = x + np.cos(a) * (0.045 + 0.035 * q) * sz - np.sin(a) * rng.uniform(-1, 1, m) * 0.016 * (1 - q) * sz
            sy = y + np.sin(a) * (0.045 + 0.035 * q) * sz + np.cos(a) * rng.uniform(-1, 1, m) * 0.016 * (1 - q) * sz
            self.put(img, np.concatenate([px, sx]), np.concatenate([py, sy]),
                     np.vstack([col, np.broadcast_to(np.array([160, 76, 40], np.float32), (m, 3))]), r1)
        # the light each group of lamps throws on the floor (worked out at half size)
        H2, W2 = self.H // 2 + 1, self.W // 2 + 1
        yy, xx = np.mgrid[0:H2, 0:W2].astype(np.float32)
        X, Y = (xx * 2 + 1) / self.P, (yy * 2 + 1) / self.P
        self.gmap = []
        for g in range(5):
            m = np.zeros((H2, W2), np.float32)
            for x, y, a, sz, _, gg in D:
                if int(gg) != g:
                    continue
                d2 = (X - x) ** 2 + (Y - y) ** 2
                m += sz * (1.0 / (1 + d2 / 0.05) + 0.022 / (1 + d2 / 0.5))
            full = np.repeat(np.repeat(m, 2, 0), 2, 1)[:self.H, :self.W]
            self.gmap.append(full.astype(np.float32))
        self.gall = sum(self.gmap)
        self.gwin = [(16.6, 19.4), (18.0, 20.6), (18.6, 21.6), (20.0, 22.4), (19.0, 22.0)]

    # ---- strings of lights and the star lantern ---------------------------------------------------
    def build_lights(self):
        rng = np.random.default_rng(12)
        self.strings = []
        for a, b, zt, zm in (((0.2, 0.25), (4.8, 4.75), 2.9, 2.2), ((0.2, 4.75), (4.8, 0.25), 2.9, 2.3), ((0.0, 1.55), (5.0, 1.75), 3.0, 2.5),
                             ((0.0, 3.55), (5.0, 3.35), 3.0, 2.4)):
            q = np.linspace(0, 1, 140)
            X = a[0] + (b[0] - a[0]) * q
            Y = a[1] + (b[1] - a[1]) * q
            Z = zt - (zt - zm) * 4 * q * (1 - q)
            self.strings.append((X, Y, Z, rng.integers(0, 4, 140), rng.uniform(0, TAU, 140)))
        self.bulb_cols = np.array([C["pink"], C["turq"], C["yellow"], (255, 70, 60)], np.float32)
        a = np.linspace(0, TAU, 11)[:-1]                                      # the star lantern's outline
        rad = np.where(np.arange(10) % 2 == 0, 1.0, 0.45)
        pts = np.stack([np.cos(a - math.pi / 2) * rad, np.sin(a - math.pi / 2) * rad], 1)
        edge = []
        for k in range(10):
            p0, p1 = pts[k], pts[(k + 1) % 10]
            q = np.linspace(0, 1, 30)[:, None]
            edge.append(p0 + (p1 - p0) * q)
        self.star_edge = np.vstack(edge)
        u = rng.random((2, 1200)) * 2 - 1                                     # its paper, inside the outline
        ang = np.arctan2(u[1], u[0]) + math.pi / 2
        lim = 0.45 + 0.55 * np.abs(np.cos(ang * 2.5)) ** 2
        keep = np.hypot(u[0], u[1]) < lim
        self.star_fill = u[:, keep].T

    # ---- the celebration --------------------------------------------------------------------------
    def build_motion(self):
        rng = self.rng
        # the rangoli's grains that will lift into the sky
        R = self.rg
        y0, y1, x0, x1 = R["box"]
        ok = np.isfinite(R["t"])
        iy, ix = np.nonzero(ok)
        pick = rng.choice(len(iy), 3200, replace=False)
        iy, ix = iy[pick], ix[pick]
        self.motes = dict(x=(ix + x0 + 0.5) / self.P, y=(iy + y0 + 0.5) / self.P, col=R["col"][iy, ix], r=R["r"][iy, ix],
                          th=R["th"][iy, ix], j=rng.random(3200))
        # firework shells
        self.bursts = []
        for tb, (bx, by), z, c0, c1, size in BURSTS:
            n = int(300 * size)
            v = rng.normal(0, 1, (3, n))
            v /= np.linalg.norm(v, axis=0)
            self.bursts.append(dict(t=tb, x=bx, y=by, z=z, c0=np.array(C[c0], np.float32), c1=np.array(C[c1], np.float32),
                                    size=size, v=v * 2.2 * size, id=rng.uniform(0, 100, n)))
        # anar fountains: each spark relaunches every 1.1 s on its own schedule
        n = 520
        self.anar = dict(o=rng.random(n) * 1.1, vz=rng.uniform(3.0, 5.4, n), vh=rng.uniform(0.1, 1.7, n), a=rng.uniform(0, TAU, n),
                         green=rng.random(n) < 0.15)
        # marigold petals, flung up and left where they land
        n = 220
        src = rng.integers(0, 2, n)
        self.petals = dict(src=src, t0=34.0 + rng.random(n) * 0.7, vz=rng.uniform(2.6, 4.2, n), vx=rng.normal(0, 0.8, n),
                           vy=rng.normal(0, 0.8, n), ph=rng.uniform(0, TAU, n), col=np.array([C["saffron"], C["yellow"], (255, 160, 20)],
                                                                                              np.float32)[rng.integers(0, 3, n)])
        n = 400                                                               # embers raining onto the rangoli
        ang, rr = rng.uniform(0, TAU, n), RR * np.sqrt(rng.random(n))
        self.embers = dict(x=RC[0] + np.cos(ang) * rr, y=RC[1] + np.sin(ang) * rr, t0=52.0 + rng.random(n) * 3.5, z0=rng.uniform(2.0, 3.5, n))
        n = 500                                                               # smoke
        self.smoke = dict(x=rng.uniform(0.3, 4.7, n), y=rng.uniform(0.3, 4.7, n), ph=rng.uniform(0, TAU, n), r=rng.random(n))

    # kids ----------------------------------------------------------------------------------------------
    KIDS = [  # where they stand, which way they face, clothes, sparkler on/off, sparkler pattern
        dict(x=0.86, y=3.50, face=-0.55, skirt=(240, 70, 150), top=(40, 160, 90), on=(24.0, 43.0), pat="eight", girl=True),
        dict(x=4.20, y=1.62, face=2.6, skirt=None, top=(250, 200, 40), on=(25.5, 42.5), pat="spiral", girl=False),
    ]

    def kid_pos(self, k, t):
        K = self.KIDS[k]
        arrive = float(ss(t, 22.5, 24.0))
        bx = K["x"] + 0.08 * math.sin(0.7 * t + k) * arrive
        by = K["y"] + 0.05 * math.cos(0.5 * t + k * 2) * arrive
        door = (2.5, 5.3)
        return door[0] + (bx - door[0]) * arrive, door[1] + (by - door[1]) * arrive, arrive

    def tip(self, k, t):
        """The sparkler's tip (m), out in front of the child, drawing its pattern."""
        K = self.KIDS[k]
        x, y, _ = self.kid_pos(k, t)
        f = K["face"]
        cx, cy = x + 0.42 * math.cos(f), y + 0.42 * math.sin(f)
        if K["pat"] == "eight":
            tau = 3.0 * t
            u, v = 0.22 * math.sin(tau), 0.14 * math.sin(2 * tau)
        else:
            tau = 4.2 * t
            rr = 0.10 + 0.10 * (0.5 + 0.5 * math.sin(0.6 * t))
            u, v = rr * math.cos(tau), rr * math.sin(tau)
        cf, sf = math.cos(f), math.sin(f)
        return cx + u * cf - v * sf, cy + u * sf + v * cf

    def draw_kids(self, img, t, light):
        for k, K in enumerate(self.KIDS):
            x, y, arrive = self.kid_pos(k, t)
            if arrive <= 0:
                continue
            rng = np.random.default_rng(40 + k)
            f = K["face"]
            cf, sf = math.cos(f), math.sin(f)
            lx, ly = int(np.clip(x * self.P, 0, self.W - 1)), int(np.clip(y * self.P, 0, self.H - 1))
            lit = self._amb_m + light[ly, lx] * 0.8 + 0.15
            parts = []
            if K["skirt"] is not None:                                        # a lehenga, flaring as she twirls
                n = 1600
                a, rr = rng.uniform(0, TAU, n), 0.27 * np.sqrt(rng.random(n))
                spin = 2.0 * t
                pat = (np.sin(a * 12 + spin) > 0.6) | (rr > 0.24)
                col = np.where(pat[:, None], np.array(C["gold"], np.float32), np.array(K["skirt"], np.float32))
                parts.append((x + np.cos(a) * rr, y + np.sin(a) * rr, 0.45, col))
            n = 500                                                           # shoulders and back
            a, rr = rng.uniform(0, TAU, n), np.sqrt(rng.random(n))
            parts.append((x + (np.cos(a) * 0.10 * cf - np.sin(a) * 0.18 * sf) * rr, y + (np.cos(a) * 0.10 * sf + np.sin(a) * 0.18 * cf) * rr,
                          1.15, np.broadcast_to(np.array(K["top"], np.float32), (n, 3))))
            tx, ty = self.tip(k, t)                                           # the arm out to the sparkler
            hx, hy = x + (tx - x) * 0.62, y + (ty - y) * 0.62
            q = np.linspace(0, 1, 40)
            parts.append((x + 0.13 * -sf + (hx - x - 0.13 * -sf) * q, y + 0.13 * cf + (hy - y - 0.13 * cf) * q, 1.1,
                          np.broadcast_to(np.array([150, 100, 70], np.float32), (40, 3))))
            n = 420                                                           # the head: dark hair
            a, rr = rng.uniform(0, TAU, n), 0.085 * np.sqrt(rng.random(n))
            hair = np.array([24, 18, 18], np.float32) * (0.8 + 0.5 * rng.random((n, 1)))
            parts.append((x + np.cos(a) * rr, y + np.sin(a) * rr, 1.3, hair))
            if K["girl"]:                                                     # a braid with jasmine
                q = np.linspace(0, 1, 30)
                bx, by = x - cf * (0.08 + 0.16 * q), y - sf * (0.08 + 0.16 * q)
                parts.append((bx, by, 1.2, np.broadcast_to(np.array([24, 18, 18], np.float32), (30, 3))))
                parts.append((bx[:8] + rng.normal(0, 0.012, 8), by[:8] + rng.normal(0, 0.012, 8), 1.31,
                              np.broadcast_to(np.array([250, 250, 240], np.float32), (8, 3))))
            for X, Y, Z, col in parts:
                px, py = self.proj(X, Y, Z * 0.3)
                stamp(img, px, py, col * lit, max(1.0, 1.5 * self.S / 1000))
            # the sparkler: a trail of light written in the air, and its crackle
            a, b = K["on"]
            on = float(ss(t, a, a + 0.4) * (1 - ss(t, b - 0.5, b)))
            if on <= 0.01:
                continue
            ks = np.arange(0, 110)
            pts = np.array([self.tip(k, t - 0.008 * j) for j in ks])
            px, py = self.proj(pts[:, 0], pts[:, 1], 0.3)
            br = (1 - ks / 110) ** 1.6 * on
            glow(img, px, py, np.array([255, 226, 170])[None] * (br * 2.6)[:, None], 1.6)
            self.haze.add(px[::6], py[::6], np.array([255, 200, 120])[None] * (br[::6] * 0.35)[:, None], 16 * self.S / 1000)
            tx, ty = self.proj(np.array([pts[0, 0]]), np.array([pts[0, 1]]), 0.3)
            r = np.random.default_rng(int(t * 60) + k * 7)
            n = 26
            ang = r.uniform(0, TAU, n)
            ln = r.uniform(0.01, 0.06, n) * self.P
            q = np.array([0.3, 0.6, 1.0])[None, :]
            sx = tx[0] + np.cos(ang)[:, None] * ln[:, None] * q
            sy = ty[0] + np.sin(ang)[:, None] * ln[:, None] * q
            glow(img, sx.ravel(), sy.ravel(), np.array([255, 240, 200]) * 1.1 * on, 1.0)
            glow(img, tx, ty, np.array([255, 255, 240]) * 3.0 * on, 2.5)
            self.haze.add(tx, ty, np.array([255, 210, 140]) * 0.9 * on, 40 * self.S / 1000)

    # fireworks of the ground ---------------------------------------------------------------------------------
    def draw_chakri(self, img, t, cx, cy, a, b, seed):
        on = float(ss(t, a, a + 0.3) * (1 - ss(t, b - 0.6, b)))
        if on <= 0.01:
            return
        w = 32.0 * on
        K = 420
        k = np.arange(K)
        age = 0.45 * k / K + ((t * 333) % 1) * 0.45 / K
        e = t - age
        th = w * e + seed
        rim = 0.035
        out = []
        for lag, kk in ((0.0, 1.0), (0.008, 0.75), (0.016, 0.5)):
            ag = np.maximum(age - lag, 0)
            d = 3.6 * 0.15 * (1 - np.exp(-ag / 0.15))
            tx, ty = -np.sin(th), np.cos(th)
            x = cx + np.cos(th) * rim + (tx + 0.3 * np.cos(th)) * d
            y = cy + np.sin(th) * rim + (ty + 0.3 * np.sin(th)) * d
            br = np.exp(-ag / 0.22) * kk * on
            col = np.array([255, 220, 120])[None] * (1 - ag / 0.5)[:, None] + np.array([255, 90, 40])[None] * (ag / 0.5)[:, None]
            out.append((x, y, col * br[:, None] * 2.6))
        x, y, col = (np.concatenate(z) for z in zip(*out))
        px, py = self.proj(x, y, 0.05)
        glow(img, px, py, col, 1.4)
        qx, qy = self.proj(np.array([cx]), np.array([cy]), 0.05)
        glow(img, qx, qy, np.array([255, 250, 220]) * 4.0 * on, 3.0)
        self.haze.add(qx, qy, np.array([255, 190, 90]) * 1.4 * on, 60 * self.S / 1000)

    def draw_anar(self, img, t, cx, cy, a, b):
        on = float(ss(t, a, a + 0.5) * (1 - ss(t, b - 0.8, b)))
        if on <= 0.01:
            return
        A = self.anar
        age = (t - A["o"]) % 1.1
        z = A["vz"] * age - 4.9 * age ** 2
        alive = z > 0
        x = cx + np.cos(A["a"]) * A["vh"] * age
        y = cy + np.sin(A["a"]) * A["vh"] * age
        out = []
        for lag, kk in ((0.0, 1.0), (0.03, 0.5)):
            ag = np.maximum(age - lag, 0)
            zz = np.maximum(A["vz"] * ag - 4.9 * ag ** 2, 0)
            px, py = self.proj(cx + np.cos(A["a"]) * A["vh"] * ag, cy + np.sin(A["a"]) * A["vh"] * ag, zz)
            br = (1 - ag / 1.1) ** 1.3 * kk * on * alive
            col = np.where(A["green"][:, None], np.array([120, 255, 140], np.float32), np.array([255, 226, 150], np.float32))
            out.append((px, py, col * br[:, None] * 2.4, 1.2 + 0.8 * (zz > 2.0)))
        px, py, col, rad = (np.concatenate(z_) for z_ in zip(*out))
        glow(img, px, py, col, rad)
        qx, qy = self.proj(np.array([cx]), np.array([cy]), 0.1)
        glow(img, qx, qy, np.array([255, 250, 230]) * 3.0 * on, 3.0)
        self.haze.add(qx, qy, np.array([255, 200, 110]) * 1.6 * on, 70 * self.S / 1000)

    def draw_petals(self, img, t, light):
        Pt = self.petals
        age = t - Pt["t0"]
        on = age > 0
        if not on.any():
            return
        age = age[on]
        src = Pt["src"][on]
        sx = np.array([self.tip(0, 34.0)[0], self.tip(1, 34.0)[0]])[src]
        sy = np.array([self.tip(0, 34.0)[1], self.tip(1, 34.0)[1]])[src]
        tl = Pt["vz"][on] / 9.8 * 2                                           # time to come back down (a little drag)
        fe = np.minimum(age, tl)
        z = np.maximum(1.0 + Pt["vz"][on] * fe - 4.9 * fe ** 2, 0) * (age < tl)
        x = sx + Pt["vx"][on] * fe + 0.05 * np.sin(5 * fe + Pt["ph"][on])
        y = sy + Pt["vy"][on] * fe + 0.05 * np.cos(4 * fe + Pt["ph"][on])
        px, py = self.proj(x, y, z)
        a = Pt["ph"][on] + 7 * fe
        lx = np.clip(px.astype(int), 0, self.W - 1)
        ly = np.clip(py.astype(int), 0, self.H - 1)
        col = Pt["col"][on] * (self._amb_m + light[ly, lx][:, None] * 0.8 + 0.2)
        s = (1.2 + 1.2 * z / 3) * self.S / 1000 * 2
        for o in (-1, 0, 1):
            stamp(img, px + o * s * np.cos(a), py + o * s * np.sin(a) * 0.6, col, 1.0 + self.S / 1000)

    # the rangoli rising, and the fireworks above ------------------------------------------------------------
    def draw_motes(self, img, t):
        M = self.motes
        ts = 42.0 + 2.2 * (1 - M["r"] / RR) + 0.6 * M["j"]                   # from the rim inward
        age = t - ts
        on = (age > 0) & (age < 5.5)
        if not on.any():
            return
        a = age[on]
        z = 6.4 * (a / 4.5) ** 1.5
        th = M["th"][on] + 1.4 * a
        r = M["r"][on] * (1 + 0.25 * a)
        x, y = RC[0] + np.cos(th) * r, RC[1] + np.sin(th) * r
        px, py = self.proj(x, y, z)
        k = HC / (HC - np.minimum(z, HC - 0.3))
        br = np.clip(a / 0.4, 0, 1) * np.clip(1 - (z - 4.8) / 1.6, 0, 1)
        glow(img, px, py, M["col"][on] * (br * 2.6)[:, None], np.clip(1.0 + 0.4 * k, 1.2, 3.0))

    def draw_bursts(self, img, t):
        for B in self.bursts:
            a = t - B["t"]
            if a < -0.6 or a > 3.6:
                continue
            if a < 0:                                                         # the shell rising past us
                q = (a + 0.6) / 0.6
                px, py = self.proj(np.array([B["x"]]), np.array([B["y"]]), B["z"] * q)
                glow(img, px, py, np.array([255, 220, 160]) * 1.5, 2.0)
                continue
            out = []
            side = np.hypot(B["v"][0], B["v"][1]) / (np.linalg.norm(B["v"], axis=0) + 1e-6)
            for lag, kk in ((0.0, 1.0), (0.025, 0.85), (0.05, 0.7), (0.075, 0.58), (0.10, 0.46), (0.13, 0.36), (0.16, 0.27),
                            (0.20, 0.18), (0.24, 0.1)):
                aa = max(a - lag, 0.0)
                d = 0.55 * (1 - math.exp(-aa / 0.55))
                X = B["x"] + B["v"][0] * d
                Y = B["y"] + B["v"][1] * d
                Z = B["z"] + B["v"][2] * d - 1.2 * aa * aa
                ok = Z < HC - 0.9
                px, py = self.proj(X[ok], Y[ok], Z[ok])
                br = math.exp(-aa / 1.3) * kk * 5.5
                tw = (np.sin(70 * aa + B["id"][ok]) > -0.2).astype(np.float32) if aa > 1.6 else np.ones(int(ok.sum()), np.float32)
                col = B["c0"] + (B["c1"] - B["c0"]) * min(aa / 1.8, 1.0)
                out.append((px, py, (col * br)[None] * (tw * (0.25 + 0.75 * side[ok] ** 2))[:, None]))
            px = np.concatenate([o[0] for o in out])
            py = np.concatenate([o[1] for o in out])
            col = np.concatenate([o[2] for o in out])
            ok = (px > -5) & (px < self.W + 5) & (py > -5) & (py < self.H + 5)
            glow(img, px[ok], py[ok], col[ok], 2.0)
            f = math.exp(-a / 0.35)
            if f > 0.03:
                qx, qy = self.proj(np.array([B["x"]]), np.array([B["y"]]), B["z"])
                self.haze.add(np.repeat(qx, 6), np.repeat(qy, 6), (B["c0"] + B["c1"]) * 0.6 * f, 120 * self.S / 1000)

    def draw_embers(self, img, t):
        E = self.embers
        a = t - E["t0"]
        on = (a > 0) & (a < 2.4)
        if not on.any():
            return
        a = a[on]
        fall = np.minimum(a / 1.2, 1)
        z = E["z0"][on] * (1 - fall ** 2)
        px, py = self.proj(E["x"][on], E["y"][on], z)
        land = a > 1.2
        br = np.where(land, np.exp(-(a - 1.2) / 0.5) * 1.6, 0.7)
        glow(img, px, py, np.array([255, 210, 140])[None] * br[:, None], 1.0 + 0.6 * land)

    # ---- the frame ---------------------------------------------------------------------------------
    def settled(self):
        """Once the rangoli is finished and every lamp is lit, the lit floor never changes: work it out once."""
        if not hasattr(self, "_full"):
            base = self.floor.copy()
            R = self.rg
            y0, y1, x0, x1 = R["box"]
            m = np.isfinite(R["t"])
            base[y0:y1, x0:x1][m] = R["col"][m]
            for x, y, col, tf in self.feet:
                self.put(base, x, y, col, max(1.0, 1.3 * self.S / 1000))
            self._full = base
            self._full_lit = (base * (self.gall * 0.29)[..., None] * WARM).astype(np.float32)
        return self._full, self._full_lit

    def render(self, t):
        W, H = self.W, self.H
        R = self.rg
        y0, y1, x0, x1 = R["box"]
        if getattr(self, "quick", True) and t > max(b for a, b in self.gwin) and t > RINGS[-1][3] + 0.3 and t > self.feet[-1][3]:
            base, lit = self.settled()                                        # the quick way, once all is lit
            m = np.isfinite(R["t"])
            amb = keyed(t, AMB).astype(np.float32)
            flick = 1 + 0.035 * math.sin(13 * t) * math.sin(5.3 * t)
            flash = self.draw_bursts_light(t)
            dim = 1 - 0.5 * float(ss(t, 45.6, 46.3) * (1 - ss(t, 55.5, 58.0)))
            img = lit * np.float32(flick * dim)
            img += base * ((amb + flash) * dim)[None, None, :]
            warm = self.gall * (0.29 * flick)
            return self.finish_frame(img, t, m, amb, warm, flash)
        img = self.floor.copy()
        m = R["t"] <= t
        reg = img[y0:y1, x0:x1]
        reg[m] = R["col"][m]
        fresh = m & (R["t"] > t - 0.25)
        reg[fresh] = np.minimum(R["col"][fresh] * 1.25 + 25, 255)
        for x, y, col, tf in self.feet:
            if t > tf:
                self.put(img, x, y, col, max(1.0, 1.3 * self.S / 1000))
        # light: the dusk, the lamps, the flashes of the fireworks
        amb = keyed(t, AMB).astype(np.float32)
        ws = [float(np.clip((t - a) / (b - a), 0, 1)) for a, b in self.gwin]
        flick = 1 + 0.035 * math.sin(13 * t) * math.sin(5.3 * t)
        if all(w >= 1 for w in ws):
            warm = self.gall * (0.29 * flick)
        elif any(w > 0 for w in ws):
            warm = sum(w * g for w, g in zip(ws, self.gmap) if w > 0) * (0.29 * flick)
        else:
            warm = np.zeros((H, W), np.float32)
        flash = self.draw_bursts_light(t)
        dim = 1 - 0.5 * float(ss(t, 45.6, 46.3) * (1 - ss(t, 55.5, 58.0)))  # eyes on the sky: the courtyard drops back
        img *= ((amb + flash)[None, None, :] + warm[..., None] * WARM) * dim
        return self.finish_frame(img, t, m, amb, warm, flash)

    def finish_frame(self, img, t, m, amb, warm, flash):
        R = self.rg
        y0, y1, x0, x1 = R["box"]
        lamp = float(ss(t, 16.6, 19.5)) * 0.16                                # the powder catching the lamplight
        glowr = lamp + float(ss(t, 42.0, 44.0) * (1 - ss(t, 46.0, 48.0))) * 0.25 + float(ss(t, 52.0, 54.0)) * 0.12
        if glowr > 0 or t > 52.4:                                             # the rangoli glowing, a wave running outward
            k = np.full(R["r"].shape, glowr, np.float32)
            if t > 52.4:
                k += 0.6 * np.exp(-((R["r"] - (t - 52.4) * 0.65) / 0.16) ** 2)
            if "colm" not in R:
                R["colm"] = np.where(np.isfinite(R["t"])[..., None], R["col"], 0).astype(np.float32)
            img[y0:y1, x0:x1] += R["colm"] * k[..., None]
        self._amb_m = float(amb.mean())
        self.draw_petals(img, t, warm)
        self.draw_kids(img, t, warm)
        self.draw_flames(img, t)
        self.draw_chakri(img, t, 1.22, 4.45, 26.0, 38.5, 0.0)
        self.draw_chakri(img, t, 3.78, 0.56, 29.0, 40.0, 1.7)
        self.draw_anar(img, t, 0.50, 2.45, 30.0, 41.5)
        self.draw_anar(img, t, 4.50, 2.45, 31.5, 42.0)
        self.draw_strings(img, t)
        self.draw_motes(img, t)
        self.draw_bursts(img, t)
        self.draw_embers(img, t)
        self.draw_smoke(t, flash)
        self.haze.apply(img, 1.0, 2)
        np.clip(img, 0, 4095, out=img)
        return self.lut[img.astype(np.uint16)]

    def draw_bursts_light(self, t):
        flash = np.zeros(3, np.float32)
        for B in self.bursts:
            a = t - B["t"]
            if 0 <= a < 2.5:
                flash += (B["c0"] * 0.6 + B["c1"] * 0.4) / 255 * 1.1 * math.exp(-a / 0.4) * B["size"]
        return flash

    def draw_flames(self, img, t):
        D = self.diya
        x, y, a, sz, ton = D[:, 0], D[:, 1], D[:, 2], D[:, 3], D[:, 4]
        on = t > ton
        if not on.any():
            return
        ph0, ph1 = self.d_ph
        fl = (0.82 + 0.18 * np.sin(17 * t + ph0) * np.sin(7.3 * t + ph1))[on]
        strike = np.exp(-(t - ton[on]) / 0.25) * 1.5
        lean = 0.012 * np.sin(1.3 * t + ph0[on])
        fx = x[on] + np.cos(a[on]) * 0.082 * sz[on] + lean
        fy = y[on] + np.sin(a[on]) * 0.082 * sz[on]
        px, py = self.proj(fx, fy, 0.04)
        s = self.S / 1000
        glow(img, px, py, np.array([255, 140, 40])[None] * ((fl + strike) * 2.2)[:, None], np.full(int(on.sum()), 3.0))
        glow(img, px, py - 1, np.array([255, 250, 225])[None] * ((fl + strike) * 3.2)[:, None], np.maximum(1.2, 2.0 * s * sz[on]))
        self.haze.add(px, py, np.array([255, 150, 60])[None] * ((fl + strike) * 1.3)[:, None], 34 * s * sz[on])

    def draw_strings(self, img, t):
        on = float(ss(t, 21.5, 23.5))
        if on <= 0:
            return
        for k, (X, Y, Z, ci, ph) in enumerate(self.strings):
            px, py = self.proj(X, Y, Z)
            stamp(img, px, py, np.array([24, 20, 22], np.float32), 1.0)       # the wire, dark against the floor
            chase = (np.arange(len(X)) + int(t * 6) * (1 if k % 2 else -1)) % 4
            cols = self.bulb_cols[(ci + chase) % 4]
            sparkle = np.clip((t - 21.5 - np.arange(len(X)) / len(X) * 2.0) / 0.2, 0, 1)
            b = on * sparkle * (0.7 + 0.3 * np.sin(4 * t + ph))
            glow(img, px, py, cols * (b * 2.4)[:, None], 2.0)
            self.haze.add(px[::2], py[::2], cols[::2] * (b[::2] * 0.45)[:, None], 22 * self.S / 1000)
        # the star lantern, swaying
        sx, sy, sz = 3.95 + 0.04 * math.sin(0.9 * t), 0.95 + 0.03 * math.cos(0.7 * t), 2.2
        rot = 0.15 * math.sin(0.6 * t)
        cr, sr = math.cos(rot), math.sin(rot)
        R = 0.22
        e = self.star_edge
        px, py = self.proj(sx + R * (e[:, 0] * cr - e[:, 1] * sr), sy + R * (e[:, 0] * sr + e[:, 1] * cr), sz)
        glow(img, px, py, np.array([255, 200, 90]) * 1.6 * on, 1.3)
        f = self.star_fill
        px, py = self.proj(sx + R * (f[:, 0] * cr - f[:, 1] * sr), sy + R * (f[:, 0] * sr + f[:, 1] * cr), sz)
        glow(img, px, py, np.array([255, 70, 150]) * 0.5 * on, 1.5)
        qx, qy = self.proj(np.array([sx]), np.array([sy]), sz)
        self.haze.add(np.repeat(qx, 3), np.repeat(qy, 3), np.array([255, 110, 160]) * 0.9 * on, 70 * self.S / 1000)

    def draw_smoke(self, t, flash):
        k = float(ss(t, 30, 38))
        if k <= 0:
            return
        Sm = self.smoke
        x = Sm["x"] + 0.15 * np.sin(0.2 * t + Sm["ph"])
        y = Sm["y"] + 0.10 * np.cos(0.17 * t + Sm["ph"])
        px, py = self.proj(x, y, 1.5 + Sm["r"])
        col = np.array([34, 30, 34], np.float32) + flash * 120
        self.haze.add(px, py, col[None] * (k * (0.3 + 0.7 * Sm["r"]))[:, None] * 0.6, 40 * self.S / 1000)


if __name__ == "__main__":
    run(Diwali, "Diwali night", "diwali", T_END, STAGES)
