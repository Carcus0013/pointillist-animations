"""
iron_streamliner.py - an armoured streamliner crossing a steel trestle by night, in dots.

Drizzle over a mill town. Down in the valley the blast furnaces pour and glow, lighting the
smoke and the underside of a long steel trestle that strides across on lattice legs. A
headlight swells in the dark at the left: an art-deco streamliner, all riveted armour plate
and chrome speed-stripes, thunders onto the bridge, its beam cutting a cone through the rain,
smoke rolling back from its stack, sparks spitting from the wheels. The carriages follow,
windows warm, past us and away across the valley, until only the red lamp on the last car is
left, shrinking into the night. A blimp drifts over the far hills, a ribbon of lights running
along its flank.

    python iron_streamliner.py            interactive window (Space pause, R restart, S save frame, Esc quit)
    python iron_streamliner.py --png      save five stages
    python iron_streamliner.py --gif      save the whole crossing as a GIF
"""
import math

import numpy as np

from neon import TAU, Cam, Lights, finish, paint, run, ss, stamp, tone_lut, unit

T_END = 60.0
STAGES = [4.0, 12.0, 24.0, 40.0, 57.0]
AMBER = np.array([255, 160, 60], np.float32)
WARM = np.array([255, 210, 140], np.float32)
FURNACE = np.array([255, 110, 30], np.float32)
ALPHA = math.radians(21)          # the bridge runs away from us to the right
DECK = 46.0                       # height of the rails above the valley floor
SPEED = 6.0                       # metres a second
CARS = [("loco", 30.0)] + [("car", 24.0)] * 7
GAP = 1.6


class IronStreamliner:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.S = S = min(W, H)
        self.cam = Cam(W, H, 0.5 * W, 0.74 * H, 0.80 * S, 8.0)
        self.lut = tone_lut(1.3, 5.0)
        self.L = Lights(W, H, 4)
        self.dir = np.array([math.cos(ALPHA), 0.0, math.sin(ALPHA)])
        self.origin = np.array([-78.0, DECK, 84.0])
        self.build_world()
        self.build_train()
        self.build_extras()

    def track(self, s):
        """Point on the rails at distance s along the bridge."""
        s = np.asarray(s, np.float64)
        return self.origin[None] + s[..., None] * self.dir[None] if s.ndim else self.origin + s * self.dir

    # ---- the valley, the town, the bridge -------------------------------------------------------------
    def build_world(self):
        """Seen from down in the valley: the trestle overhead, the mill town around its feet, a furnace-lit sky."""
        W, H, c = self.W, self.H, self.cam
        rng = np.random.default_rng(6)
        yy = np.arange(H, dtype=np.float32)[:, None] / H
        hz = c.vy / H
        stops = ((0.0, (8, 10, 18)), (0.3, (20, 20, 30)), (0.55, (60, 40, 40)), (hz - 0.02, (140, 74, 44)), (hz, (170, 90, 50)),
                 (1.0, (30, 20, 20)))
        img = np.stack([np.interp(yy, [s_[0] for s_ in stops], [s_[1][k] for s_ in stops]) for k in range(3)], -1)[..., 0, :]
        img = np.repeat(img[:, None, :], W, 1).astype(np.float32)
        xs = np.arange(W)[None, :] / W
        clouds = (0.5 + 0.5 * np.sin(xs * 5 + yy * 9) * np.sin(yy * 21 - xs * 3)) ** 2 * (yy < hz)
        under = np.clip((yy - 0.25) / (hz - 0.25), 0, 1) ** 2                 # clouds lit from below by the furnaces
        img = img * (1 - 0.35 * clouds[..., None]) + (np.array([30, 26, 34], np.float32) + FURNACE * 0.45 * under[..., None]) * \
            (clouds * 0.55)[..., None]
        depth = np.full((H, W), 1e5, np.float32)
        # far hills with a city glittering on them
        u = np.arange(W) / W
        ridge = c.vy - H * (0.025 + 0.02 * np.sin(u * 7 + 1) + 0.01 * np.sin(u * 19))
        m = (np.arange(H)[:, None] >= ridge[None, :]) & (np.arange(H)[:, None] < c.vy + 2)
        img[m] = np.array([30, 22, 24], np.float32)
        n = 2600
        x = rng.uniform(0, W, n)
        y = np.interp(x, np.arange(W), ridge) + rng.random(n) ** 2 * (c.vy - np.interp(x, np.arange(W), ridge) + 2)
        stamp(img, x, y, AMBER * (0.25 + 0.35 * rng.random((n, 1))), 1.0)
        # the valley floor near us: wet cinders and puddles holding the fire in the sky
        ys = np.arange(H)[:, None].astype(np.float64)
        below = ys > c.vy + 1
        zg = np.where(below, c.f * c.hc / np.maximum(ys - c.vy, 1e-3), 1e5) * np.ones((1, W))
        Xg = (np.arange(W)[None, :] - c.vx) * zg / c.f
        pud = np.clip((np.sin(Xg * 0.25 + np.sin(zg * 0.3) * 2) * np.sin(zg * 0.35 + Xg * 0.1) - 0.2) * 2.5, 0, 1)
        src = np.clip((2 * c.vy - ys).astype(int), 0, H - 1)
        mirror = img[src[:, 0]]
        ground = np.array([26, 20, 20], np.float32) * (0.8 + 0.3 * rng.random((H, W, 1))) * (1 - 0.6 * pud[..., None]) + mirror * (0.65 * pud)[..., None]
        rails = (np.abs((Xg % 9.0) - 4.5) < 0.12) & (zg < 400)               # sidings in the yard
        ground[rails] = np.array([90, 70, 60], np.float32)
        img = np.where(below[..., None], ground, img).astype(np.float32)
        depth = np.where(below, zg, depth).astype(np.float32)
        # the mill town: sheds, blast furnaces, chimneys
        self.furnaces = []
        boxes = []
        for _ in range(26):
            X, z = rng.uniform(-140, 260), rng.uniform(110, 420)
            boxes.append((X, z, rng.uniform(8, 18), rng.uniform(6, 16), (36, 28, 28), "shed"))
        for X, z in ((-30.0, 140.0), (40.0, 200.0), (120.0, 260.0), (-95.0, 300.0), (185.0, 340.0)):
            boxes.append((X, z, 9.0, 30.0, (42, 32, 30), "furnace"))
            boxes.append((X + 6, z + 2, 2.5, 56.0, (34, 26, 26), "stack"))
            self.furnaces.append((X, z))
        boxes.sort(key=lambda b: -b[1])
        for X, z, w_, h_, col, kind in boxes:
            xa, ya = c.proj(X - w_, h_, z)
            xb, yb = c.proj(X + w_, 0, z)
            r0, r1, c0, c1 = int(max(ya, 0)), int(min(yb, H)), int(max(xa, 0)), int(min(xb, W))
            if r0 >= r1 or c0 >= c1:
                continue
            fog = min(1.0, (z - 80) / 500)
            body = np.array(col, np.float32) * (1 - 0.5 * fog) + np.array([110, 66, 44], np.float32) * 0.5 * fog
            img[r0:r1, c0:c1] = body * (0.85 + 0.3 * ((np.arange(c0, c1) - xa) / max(xb - xa, 1) > 0.6))[None, :, None]
            depth[r0:r1, c0:c1] = z
            if kind == "shed":                                                # a row of lit windows
                wy = int(r0 + (r1 - r0) * 0.55)
                cols_ = np.arange(c0 + 2, c1 - 1, 4)
                lit = rng.random(len(cols_)) < 0.6
                stamp(img, cols_[lit], np.full(int(lit.sum()), wy), AMBER * 0.8, 1.2)
        # the trestle: deck girders with lattice, tall lattice legs down to the valley floor
        steel = np.array([42, 36, 40], np.float32)
        P_, C_ = [], []

        def seg(a, b, col):
            xa, ya = c.proj(*a)
            xb, yb = c.proj(*b)
            n = int(min(900, max(4, math.hypot(xb - xa, yb - ya) / 1.2)))
            q = np.linspace(0, 1, n)[:, None]
            P_.append(np.asarray(a)[None] + (np.asarray(b) - np.asarray(a))[None] * q)
            C_.append(np.broadcast_to(col, (n, 3)))

        for k in range(-30, 560, 5):                                          # girder chords and lattice, panel by panel
            a, b = self.track(np.array(k, float)), self.track(np.array(k + 5, float))
            for dy, dz in ((-0.6, -2.6), (-0.6, 2.6), (-4.6, -2.6), (-4.6, 2.6)):
                seg(a + np.array([0, dy, dz]), b + np.array([0, dy, dz]), steel)
            seg(a + np.array([0, -0.6, -2.6]), b + np.array([0, -4.6, -2.6]), steel * 0.9)
            seg(a + np.array([0, -4.6, -2.6]), b + np.array([0, -0.6, -2.6]), steel * 0.9)
        for k in range(-24, 560, 30):                                         # legs, spreading out to their feet
            top = self.track(np.array(k, float))
            feet = [np.array([top[0], 0.0, top[2] + sd * 8.0]) for sd in (-1, 1)]
            heads = [top + np.array([0, -4.6, sd * 2.6]) for sd in (-1, 1)]
            for hd, ft in zip(heads, feet):
                seg(hd, ft, steel)
            for lvl in np.linspace(0.1, 0.92, 7):                             # X-bracing between them
                a = heads[0] + (feet[0] - heads[0]) * lvl
                b = heads[1] + (feet[1] - heads[1]) * min(lvl + 0.13, 1.0)
                seg(a, b, steel * 0.85)
                a2 = heads[1] + (feet[1] - heads[1]) * lvl
                b2 = heads[0] + (feet[0] - heads[0]) * min(lvl + 0.13, 1.0)
                seg(a2, b2, steel * 0.85)
        P = np.vstack(P_)
        C = np.vstack(C_)
        under = np.clip((DECK - P[:, 1]) / DECK, 0, 1)                        # lit orange from the furnaces below
        C = C + FURNACE * (0.20 * under ** 0.8)[:, None]
        x, y = c.proj(P[:, 0], P[:, 1], P[:, 2])
        r = np.clip(c.f * 0.30 / P[:, 2], 1.0, 3.0)
        order = np.argsort(-P[:, 2])
        stamp(img, x[order], y[order], C[order], r[order])
        xi, yi = np.clip(x.astype(int), 0, W - 1), np.clip(y.astype(int), 0, H - 1)
        np.minimum.at(depth, (yi, xi), P[:, 2].astype(np.float32))
        # stipple
        n = int(W * H * 0.35)
        x, y = rng.uniform(0, W, n), rng.uniform(0, H, n)
        stamp(img, x, y, img[y.astype(int), x.astype(int)] * (0.86 + 0.28 * rng.random((n, 1))), rng.choice([1.0, 1.5, 2.0], n))
        self.img, self.depth = img, depth

    # ---- the train -----------------------------------------------------------------------------------
    def build_train(self):
        rng = np.random.default_rng(12)
        self.units = []
        for kind, ln in CARS:
            n = 5200 if kind == "loco" else 3200
            u, v = rng.random(n), rng.uniform(-1, 1, n)                       # u along (0 front .. 1 back), v around
            self.units.append(dict(kind=kind, ln=ln, u=u, v=v, tone=rng.random(n)))

    def unit_points(self, U, front_s):
        """A car or the locomotive as points in the world, with colours and a 'what is it' tag."""
        ln = U["ln"]
        u, v = U["u"], U["v"]
        s = front_s - u * ln
        P = self.track(s)
        side = np.array([self.dir[2], 0, -self.dir[0]])                       # across the track, toward us
        loco = U["kind"] == "loco"
        if loco:                                                              # a bullet nose, then a full body
            prof = np.where(u < 0.22, np.sqrt(np.clip(u / 0.22, 0, 1)) ** 0.9, 1.0)
        else:
            prof = np.where((u < 0.03) | (u > 0.97), 0.9, 1.0)
        hgt, half = 4.6, 1.6
        ang = v * math.pi * 0.62                                              # the visible half of the body, top to skirt
        y = 1.0 + hgt * 0.5 + np.sin(ang) * hgt * 0.5 * prof * np.where(loco & (u < 0.22), 0.9, 1.0)
        w = np.cos(ang) * half * prof
        P = P + side[None] * w[:, None] + np.array([0, 1, 0])[None] * (y[:, None] - 0.0)
        top = np.clip(np.sin(ang), 0, 1)
        base = np.array([70, 70, 76], np.float32) * (0.8 + 0.35 * U["tone"])[:, None]
        col = base * (0.45 + 0.55 * top)[:, None] + AMBER * (0.12 * np.clip(-np.sin(ang), 0, 1))[:, None]
        stripe = (np.abs(v - 0.0) < 0.07) | (np.abs(v + 0.25) < 0.04)          # chrome speed stripes
        col[stripe] = np.array([210, 200, 190], np.float32) * (0.75 + 0.25 * U["tone"][stripe, None])
        rivet = ((u * ln * 1.4) % 1 < 0.08) & (np.abs(v - 0.5) < 0.35)
        col[rivet] *= 0.6
        win = np.zeros(len(u), bool)
        if loco:
            win = (u > 0.24) & (u < 0.34) & (v > 0.45) & (v < 0.75)
        else:
            win = ((u * 9) % 1 < 0.6) & (v > 0.40) & (v < 0.70) & (u > 0.06) & (u < 0.94)
        col[win] = WARM * 0.9
        return P, col, win

    def draw_train(self, dots, t):
        c, L = self.cam, self.L
        front = -40.0 + SPEED * t
        s = front
        heads = []
        for k, U in enumerate(self.units):
            P, col, win = self.unit_points(U, s)
            x, y = c.proj(P[:, 0], P[:, 1], P[:, 2])
            ok = (x > -10) & (x < self.W + 10) & (y > -10) & (y < self.H + 10)
            if ok.any():
                r = np.clip(c.f * 0.33 / P[:, 2], 1.0, 3.0)
                dots.append((x[ok], y[ok], col[ok], r[ok], -P[ok, 2] + 0.5))
                L.soft(x[win & ok][::2], y[win & ok][::2], WARM * 0.5)
            # wheels: sparks now and then
            ws = s - np.array([0.1, 0.45, 0.8]) * U["ln"]
            W3 = self.track(ws) + np.array([0, 0.3, 0])
            wx, wy = c.proj(W3[:, 0], W3[:, 1], W3[:, 2])
            r_ = np.random.default_rng(int(t * 30) + k)
            if r_.random() < 0.35:
                j = int(r_.integers(0, 3))
                n = 18
                L.add(wx[j] + r_.normal(0, 6, n), wy[j] + np.abs(r_.normal(0, 4, n)), np.array([255, 200, 120]) * r_.random((n, 1)) * 1.2, 1.0)
            heads.append(s)
            s -= U["ln"] + GAP
        # the last car's red lamp
        tail = self.track(np.array(s + GAP)) + np.array([0, 2.6, 0])
        tx, ty = c.proj(tail[0], tail[1], tail[2])
        L.tube(np.array([tx]), np.array([ty]), np.array([255, 30, 30]), 1.4 if (t * 2) % 1 < 0.75 else 0.4, core=1, halo=1.2, bloom=1.2)
        return front

    def draw_headlight(self, t, front):
        c, L = self.cam, self.L
        hp = self.track(np.array(front + 0.4)) + np.array([0, 3.0, 0])
        hx, hy = c.proj(hp[0], hp[1], hp[2])
        L.tube(np.array([hx]), np.array([hy]), np.array([255, 245, 220]), 2.2, core=1.2, halo=1.6, bloom=2.0)
        r = np.random.default_rng(int(t * 24)).random((3, 1400))              # the beam through the rain
        s = r[0] ** 0.7 * 160
        spread = 0.3 + 0.065 * s
        ang = r[1] * TAU
        up = np.array([0, 1.0, 0])
        side = np.array([self.dir[2], 0, -self.dir[0]])
        P = hp[None] + self.dir[None] * s[:, None] + (np.cos(ang)[:, None] * side[None] + np.sin(ang)[:, None] * up[None] * 0.7) * \
            (spread * np.sqrt(r[2]))[:, None]
        x, y = c.proj(P[:, 0], P[:, 1], P[:, 2])
        k = np.exp(-s / 90) * 0.5
        L.soft(x, y, np.array([255, 240, 210])[None] * k[:, None] * 2.6)
        sp = r[2] < 0.35                                                      # raindrops sparkling in it
        L.add(x[sp], y[sp], np.array([255, 245, 225])[None] * (k[sp] * 3.0)[:, None], 1.0)
        L.bloom.splat(x[::3], y[::3], np.array([255, 230, 190])[None] * (k[::3] * 0.8)[:, None])
        return hp

    # ---- smoke, furnaces, rain, the blimp ------------------------------------------------------------
    def build_extras(self):
        rng = np.random.default_rng(3)
        n = 1500
        self.smoke = dict(t0=rng.random(n) * 7, sp=rng.normal(0, 1, (3, n)))
        n = 2600
        self.rain = dict(X=rng.uniform(-30, 60, n), Y=rng.uniform(0, 60, n), z=6 + 160 * rng.random(n) ** 1.6, v=rng.uniform(14, 18, n))
        self.fsmoke = dict(src=rng.integers(0, 5, 900), t0=rng.random(900) * 14, sp=rng.normal(0, 1, (3, 900)))

    def draw_smoke(self, t, front):
        c, L = self.cam, self.L
        Sm = self.smoke
        age = (t + Sm["t0"]) % 7.0
        born = front - SPEED * age * 0.0                                       # puffs leave the stack and hang back
        stack = self.track(np.array(front - 6.0)) + np.array([0, 6.0, 0])
        drift = -self.dir[None] * (SPEED * age)[:, None]                                       # left behind as the train runs on
        P = stack[None] + drift + np.stack([Sm["sp"][0] * (0.5 + 2.5 * age), 2.5 * age ** 0.8 + Sm["sp"][1] * age,
                                                         Sm["sp"][2] * (0.5 + 2.5 * age)], 1)
        x, y = c.proj(P[:, 0], P[:, 1], P[:, 2])
        dens = np.exp(-age / 3.0) * (age > 0.05)
        lit = np.clip((DECK + 10 - P[:, 1]) / 30, 0, 1)                       # lit from below by the furnaces
        col = (np.array([60, 56, 58])[None] + FURNACE[None] * (0.3 * lit)[:, None]) * dens[:, None] * 0.45
        L.soft(x, y, col * 2.0)
        L.add(x[::3], y[::3], col[::3] * 0.5, 1.0)
        F = self.fsmoke                                                      # furnace smoke and glow
        age = (t + F["t0"]) % 14.0
        f = age / 14.0
        fx = np.array([q[0] for q in self.furnaces])[F["src"]]
        fz = np.array([q[1] for q in self.furnaces])[F["src"]]
        P = np.stack([fx + 6 + F["sp"][0] * (2 + 30 * f) - 40 * f, 56 + 90 * f ** 0.8 + F["sp"][1] * 10 * f, fz + 2 + F["sp"][2] * 20 * f], 1)
        x, y = c.proj(P[:, 0], P[:, 1], P[:, 2])
        col = (np.array([50, 44, 44])[None] + FURNACE[None] * np.exp(-f * 3)[:, None] * 0.5) * (np.sin(np.pi * f) * (1 - f) * 0.4)[:, None]
        L.soft(x, y, col * 1.6)
        for k, (X, z) in enumerate(self.furnaces):                            # the furnace mouths, flickering
            fl = 0.75 + 0.25 * math.sin(t * 7 + k * 2.1) * math.sin(t * 3.1 + k)
            q = np.random.default_rng(k).random((2, 60))
            px, py = c.proj(X + (q[0] - 0.5) * 14, 4 + q[1] * 8, np.full(60, z - 10))
            L.tube(px, py, FURNACE, 0.5 * fl, core=0.6, halo=1.0, bloom=1.4)
            if k == 1:                                                       # a pour of molten iron
                q2 = np.linspace(0, 1, 40)
                px, py = c.proj(X + 10 + 6 * q2, 18 - 16 * q2 ** 2, np.full(40, z - 10))
                L.tube(px, py, np.array([255, 200, 90]), 0.9 * fl, core=1, halo=1, bloom=1.2)

    def draw_rain(self, img, t):
        R, c, L = self.rain, self.cam, self.L
        Y = (R["Y"] - R["v"] * t) % 60.0
        X = R["X"] + 0.15 * Y
        x0, y0 = c.proj(X, Y, R["z"])
        x1, y1 = c.proj(X + 0.15, Y + 1.0, R["z"])
        xi, yi = np.clip(x0.astype(int), 0, self.W - 1), np.clip(y0.astype(int), 0, self.H - 1)
        vis = (x0 > 0) & (x0 < self.W) & (y0 > 0) & (y0 < self.H) & (self.depth[yi, xi] >= R["z"] - 1)
        col = img[yi[vis], xi[vis]] * 0.3 + 10
        for k in (0.0, 0.5, 1.0):
            L.add(x0[vis] + (x1 - x0)[vis] * k, y0[vis] + (y1 - y0)[vis] * k, col * (0.8 - 0.25 * k), 1.0)

    def draw_blimp(self, dots, t):
        c, L = self.cam, self.L
        cx, cy, cz = 300 - 5.0 * t, 230.0, 900.0
        r = np.random.default_rng(1)
        n = 1500
        u, v = r.random(n), r.uniform(-1, 1, n)
        X = cx + (u - 0.5) * 120
        rad = 22 * np.sqrt(np.clip(1 - (2 * u - 1) ** 2, 0, 1))
        Y = cy + v * rad
        x, y = c.proj(X, Y, np.full(n, cz))
        col = np.array([60, 56, 60], np.float32) * (0.6 + 0.5 * np.clip(v, 0, 1))[:, None] + AMBER * (0.15 * np.clip(-v, 0, 1))[:, None]
        dots.append((x, y, col, 1.5, np.full(n, -cz)))
        q = np.linspace(-0.4, 0.4, 70)                                         # its ribbon of running lights
        lit = (np.sin(q * 60 - t * 6) > 0.2) | (np.sin(q * 23 + t * 2) > 0.85)
        bx, by = c.proj(cx + q * 120, np.full(70, cy - 2), np.full(70, cz - 20))
        L.tube(bx[lit], by[lit], np.array([255, 190, 80]), 0.8, core=0.8, halo=0.8, bloom=0.8)

    # ---- the frame ---------------------------------------------------------------------------------
    def render(self, t):
        img = self.img.copy()
        dots = []
        self.draw_blimp(dots, t)
        front = self.draw_train(dots, t)
        self.draw_headlight(t, front)
        self.draw_smoke(t, front)
        paint(img, dots)
        self.draw_rain(img, t)
        self.L.draw(img, 2)
        return finish(img, self.lut)


if __name__ == "__main__":
    run(IronStreamliner, "Iron streamliner", "iron_streamliner", T_END, STAGES)
