"""
neon_alley.py - a rain-soaked alley in a neon city, in dots.

Night, rain. A narrow alley runs between towering blocks to a gap of glowing sky where a
megatower's giant screen plays. Stacked signs in pink and cyan buzz and stutter on the walls,
red lanterns sway on the cables, steam rolls from a vent and a manhole, and every puddle
holds a broken, trembling copy of it all. A police drone drifts overhead sweeping its
searchlight through the rain while someone walks slowly toward you under a clear umbrella
with a glowing rim. Then the power surges: the signs stutter, and the whole district goes
black - lightning, the drone's beam and the umbrella the only light - until the city comes
back on block by block, rushing up the alley toward you in a cascade of neon.

    python neon_alley.py            interactive window (Space pause, R restart, S save frame, Esc quit)
    python neon_alley.py --png      save five stages
    python neon_alley.py --gif      save the whole minute as a GIF
"""
import math

import numpy as np

from neon import NEON, TAU, Cam, Lights, finish, flicker, glow, paint, run, shades, sign_strokes, ss, stamp, \
    stroke_points, tone_lut

T_END = 60.0
STAGES = [8.0, 25.4, 29.2, 33.6, 52.0]

XL, XR = -2.9, 2.7               # the alley walls
ZEND = 62.0                      # where the alley opens onto the plaza
ZTOWER = 105.0                   # the megatower beyond
SURGE, DARK0, DARK1 = 24.0, 27.0, 31.0
LIGHTNING = [(28.25, 0.9), (28.42, 0.6), (30.05, 1.0), (30.2, 0.5)]

CONCRETE = [(40, 38, 50), (46, 44, 56), (34, 34, 44), (52, 48, 58)]
ASPHALT = [(22, 22, 30), (26, 26, 34), (18, 18, 26)]
WIN = [(255, 170, 90), (120, 220, 255), (255, 110, 200), (200, 210, 255), (255, 200, 140)]


def hash01(*a):
    """A repeatable pseudo-random number in [0, 1) from integers."""
    h = np.zeros(np.broadcast(*a).shape, np.int64) if any(np.ndim(v) for v in a) else 0
    for i, v in enumerate(a):
        h = h * 1000003 + (np.asarray(v, np.int64) + 7919 * (i + 1)) * 2654435761
    return ((h ^ (h >> 13)) % 100003) / 100003.0


class NeonAlley:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.S = S = min(W, H)
        self.rng = np.random.default_rng(77)
        self.cam = Cam(W, H, 0.5 * W, 0.56 * H, 0.62 * S, 1.7)
        self.lut = tone_lut(1.25, 5.0)
        self.L = Lights(W, H, 4)
        self.build_buildings()
        self.build_signs()
        self.raycast()
        self.build_images()
        self.build_motion()

    # ---- the place ---------------------------------------------------------------------------------
    def build_buildings(self):
        rng = self.rng
        self.segs = {}
        for side in (-1, 1):
            z, segs = 0.0, []
            while z < ZEND:
                ln = rng.uniform(5, 11)
                near = z < 16
                h = rng.uniform(28, 45) if near else (rng.uniform(16, 26) if z < 34 else rng.uniform(10, 18))
                segs.append((z, min(z + ln, ZEND), h, rng.integers(0, 4), int(rng.integers(1, 3)) if near else int(rng.integers(0, 3))))
                z += ln
            self.segs[side] = segs

    def seg_of(self, side, z):
        segs = self.segs[side]
        edges = np.array([s[0] for s in segs])
        i = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, len(segs) - 1)
        return i

    def raycast(self):
        """For every pixel: what it sees (0 sky, 1 left wall, 2 right wall, 3 ground, 4 tower) and where."""
        W, H, c = self.W, self.H, self.cam
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        dx, dy = (xx - c.vx) / c.f, (c.vy - yy) / c.f
        big = 1e9
        surf = np.zeros((H, W), np.int8)
        depth = np.full((H, W), 400.0)
        for side, Xw, sid in ((-1, XL, 1), (1, XR, 2)):
            with np.errstate(divide="ignore", invalid="ignore"):
                zw = np.where(dx * side > 0, Xw / dx, big)
            Yw = c.hc + dy * zw
            heights = np.array([s[2] for s in self.segs[side]])
            hgt = heights[self.seg_of(side, np.clip(zw, 0, ZEND - 1e-3))]
            hit = (zw < ZEND) & (Yw >= 0) & (Yw <= hgt) & (zw < depth)
            surf[hit], depth[hit] = sid, zw[hit]
        with np.errstate(divide="ignore", invalid="ignore"):
            zg = np.where(dy < 0, -c.hc / dy, big)
        hit = (zg < depth) & (zg < ZTOWER)
        surf[hit], depth[hit] = 3, zg[hit]
        Xt, Yt = dx * ZTOWER, c.hc + dy * ZTOWER                           # the megatower's face
        hit = (surf == 0) & (Xt > -34) & (Xt < 26) & (Yt > 0) & (Yt < 260 - 0.4 * np.abs(Xt + 4) ** 1.2)
        surf[hit], depth[hit] = 4, ZTOWER
        self.surf, self.depth = surf, depth.astype(np.float32)
        self.P = (dx * depth, c.hc + dy * depth, depth)                     # each pixel's point in the world

    def build_signs(self):
        """Every neon sign as 3D points, a colour, where it hangs, and how it misbehaves."""
        rng = np.random.default_rng(12)
        cols = [NEON["pink"], NEON["cyan"], NEON["magenta"], NEON["blue"], NEON["violet"], NEON["pink"], NEON["cyan"]]
        signs = []

        def add(X, Y, z, col, broken=0.0, kind="sign"):
            signs.append(dict(X=np.asarray(X, np.float64), Y=np.asarray(Y, np.float64), z=np.asarray(z, np.float64),
                              col=np.asarray(col, np.float32), broken=broken, seed=rng.uniform(0, 100), kind=kind))

        # blade signs sticking out from the walls, their faces toward us
        for side, Xw, zs in ((-1, XL, (5.5, 10.5, 17, 25, 34, 45)), (1, XR, (7.5, 13.5, 21, 29, 39, 52))):
            for z in zs:
                n = int(rng.integers(3, 6))
                cell = rng.uniform(0.55, 0.85)
                u, v = stroke_points(sign_strokes(rng, n, True, True), 0.035)
                X = Xw - side * (0.25 + cell * (u + 0.22))
                Y = rng.uniform(3.0, 6.5) + (n * 1.15 - v) * cell
                add(X, Y, np.full_like(u, z), cols[int(rng.integers(0, len(cols)))], broken=0.6 if rng.random() < 0.3 else 0.0)
        # flat signs along the walls (seen edge-on, the alley's rhythm)
        for side, Xw in ((-1, XL), (1, XR)):
            for z0 in (8.0, 19.0, 31.0, 42.0):
                z0 += rng.uniform(-1.5, 1.5)
                n = int(rng.integers(3, 6))
                u, v = stroke_points(sign_strokes(rng, n, False, rng.random() < 0.5), 0.05)
                cell = 0.9
                add(np.full_like(u, Xw - side * 0.05), 4.2 + (1 - v) * cell, z0 + u * cell, cols[int(rng.integers(0, len(cols)))])
        # shapes: an arrow down to a door, a heart, a ring, a lightning bolt
        q = np.linspace(0, 1, 120)
        arrow = [(0, 2, 0, 0.6), (-0.5, 1.0, 0, 0.3), (0.5, 1.0, 0, 0.3)]
        u, v = stroke_points([(a, b, c_, d) for a, b, c_, d in arrow], 0.03)
        add(np.full_like(u, XR - 0.05), 2.4 + v * 0.8 - 0.4, 11.0 + u * 0.8, NEON["amber"], broken=0.3)
        a = q * TAU
        hx, hy = 16 * np.sin(a) ** 3, 13 * np.cos(a) - 5 * np.cos(2 * a) - 2 * np.cos(3 * a) - np.cos(4 * a)
        add(np.full_like(a, XL + 0.05), 6.5 + hy * 0.06, 14.5 + hx * 0.06, NEON["pink"])
        add(np.full_like(a, XR - 0.05), 7.5 + np.sin(a) * 1.1, 26.0 + np.cos(a) * 1.1, NEON["cyan"], broken=0.5)
        bolt = [(0, 2, 0.6, 1.0), (0.6, 1.0, 0.1, 1.0), (0.1, 1.0, 0.8, 0)]
        u, v = stroke_points(bolt, 0.02)
        add(np.full_like(u, XL + 0.05), 4.5 + v * 1.4, 36.0 + u * 1.2, NEON["violet"])
        # light strips along the walls above the doors
        for side, Xw, z0, z1, col in ((-1, XL, 3.0, 13.0, NEON["cyan"]), (1, XR, 15.0, 27.0, NEON["pink"]),
                                      (-1, XL, 27.0, 40.0, NEON["magenta"]), (1, XR, 33.0, 48.0, NEON["blue"])):
            z = np.linspace(z0, z1, 260)
            add(np.full_like(z, Xw - side * 0.08), np.full_like(z, 3.2), z, col)
        self.signs = signs
        # paper lanterns on cables across the alley
        self.lanterns = [(x, 7.6 + 0.3 * math.cos(x), z) for z in (9.0, 23.0) for x in np.linspace(XL + 0.6, XR - 0.6, 6)]
        self.cables = [(9.0, 8.4), (23.0, 8.4), (16.0, 11.0), (35.0, 9.5), (48.0, 8.0)]

    # ---- the still pictures: lights on, lights off, and a lightning flash -------------------------------
    def build_images(self):
        W, H, rng = self.W, self.H, np.random.default_rng(5)
        X3, Y3, Z3 = self.P
        surf = self.surf
        base = np.zeros((H, W, 3), np.float32)
        emit = np.zeros((H, W, 3), np.float32)                               # what glows by itself when powered
        noise = rng.random((H, W)).astype(np.float32)
        # the sky: city glow low down, clouds lit from below
        yy = np.arange(H, dtype=np.float32)[:, None] / H
        skyc = np.stack([np.interp(yy, [0, 0.35, 0.6], c) for c in ((8, 26, 70), (8, 14, 40), (22, 34, 80))], -1)[..., 0, :]
        sky = (surf == 0)
        base[sky] = np.broadcast_to(skyc[:, None, :], (H, W, 3))[sky]
        cloud = (0.5 + 0.5 * np.sin(np.arange(W)[None, :] / W * 9 + yy * 14) * np.sin(yy * 23 - np.arange(W)[None, :] / W * 4)) ** 3
        base[sky] += (np.array([60, 20, 70], np.float32) * (cloud * (0.4 + yy))[..., None])[sky]
        # walls: concrete with seams, grime, windows, shop fronts, pipes, air conditioners
        for side, sid in ((-1, 1), (1, 2)):
            m = surf == sid
            z, Y = Z3[m], Y3[m]
            si = self.seg_of(side, z)
            segs = self.segs[side]
            z0 = np.array([s[0] for s in segs])[si]
            tint = np.array([CONCRETE[s[3]] for s in segs], np.float32)[si]
            kind = np.array([s[4] for s in segs])[si]
            hgt = np.array([s[2] for s in segs])[si]
            grime = 0.65 + 0.35 * np.clip(Y / 12, 0, 1) - 0.12 * (np.sin(z * 9 + side) > 0.7)
            c = tint * (grime * (0.8 + 0.4 * noise[m]))[:, None]
            c *= (1 - 0.5 * ((z - z0) < 0.15))[:, None]                     # seams between buildings
            col_i = np.floor((z - z0) / 1.7).astype(int)
            row_i = np.floor((Y - 4.0) / 3.3).astype(int)
            wz, wy = (z - z0) % 1.7, (Y - 4.0) % 3.3
            inwin = (wz > 0.5) & (wz < 1.2) & (wy > 0.8) & (wy < 2.0) & (Y > 4.0) & (Y < hgt - 1.5)
            frame = inwin & ((wz < 0.56) | (wz > 1.14) | (wy < 0.86) | (wy > 1.94) | (np.abs(wz - 0.85) < 0.03))
            hsh = hash01(si, col_i, row_i, side)
            lit = inwin & (hsh < 0.38) & ~frame
            wc = np.array(WIN, np.float32)[(hash01(col_i, row_i, si) * len(WIN)).astype(int) % len(WIN)]
            c[inwin] = np.array([14, 14, 22], np.float32)
            c[frame] = np.array([30, 30, 40], np.float32)
            blinds = (np.sin(Y * 40) > 0.6)
            e = np.zeros_like(c)
            e[lit] = wc[lit] * (0.22 + 0.30 * hash01(row_i, col_i)[lit, None]) * (1 - 0.45 * blinds[lit, None])
            gf = Y < 3.6                                                    # the ground floor
            shutter = gf & (kind == 0)
            c[shutter] = np.array([70, 74, 86], np.float32) * (0.7 + 0.3 * (np.sin(Y[shutter] * 55) > 0))[:, None]
            sz = (z - z0) % 4.0                                             # shops: framed glass, warm inside, a cloth curtain
            shop = gf & (kind == 1) & (Y > 0.25) & (Y < 2.5) & (sz > 0.3) & (sz < 2.9)
            mull = shop & ((np.abs((sz - 0.3) % 0.65) < 0.05) | (np.abs(Y - 1.9) < 0.04))
            curtain = shop & (Y > 1.95) & ((sz * 6) % 1 < 0.85)
            glass = shop & ~mull & ~curtain
            warm = np.array([255, 150, 80], np.float32)
            e[glass] = warm * (0.42 - 0.12 * Y[glass, None]) * (0.75 + 0.25 * noise[m][glass, None])
            c[mull] = np.array([24, 22, 26], np.float32)
            c[curtain] = np.array([110, 20, 40], np.float32)
            e[curtain] = np.array([120, 20, 50], np.float32) * 0.25
            lamp = gf & (kind == 2) & (np.abs((z - z0) % 3.0 - 0.55) < 0.12) & (np.abs(Y - 2.55) < 0.12)
            e[lamp] = np.array([255, 190, 110], np.float32) * 0.9
            door = gf & (kind == 2) & (Y < 2.3) & ((z - z0) % 3.0 < 1.1)
            c[door] = np.array([16, 16, 24], np.float32)
            pipe = (np.abs((z - z0) - 0.35) < 0.07) | (np.abs((z - z0) - 3.9) < 0.05)
            c[pipe] = np.array([70, 70, 84], np.float32) * (0.6 + 0.5 * noise[m][pipe, None])
            ac_c = np.floor(z / 3.1).astype(int)
            ac_r = np.floor(Y / 4.4).astype(int)
            acx, acy = (z % 3.1), (Y % 4.4)
            ac = (hash01(ac_c, ac_r, side + 5) < 0.28) & (acx > 1.2) & (acx < 2.1) & (acy > 2.4) & (acy < 3.1) & (Y > 4)
            c[ac] = np.array([74, 76, 88], np.float32) * (0.6 + 0.4 * (np.sin(acx * 60) > 0))[ac][:, None]
            for zv in ((4.6, 5.6, 9.2) if side < 0 else (6.1, 7.0, 12.4)):  # vending machines, glowing
                vm = (z > zv) & (z < zv + 0.85) & (Y < 1.85)
                if vm.any():
                    fy, fz = Y[vm], z[vm] - zv
                    panel = (fy > 0.95) & (fy < 1.75) & (fz > 0.08) & (fz < 0.6)
                    cans = panel & ((fy * 9) % 1 < 0.55) & ((fz * 14) % 1 < 0.6)
                    vc = np.where(panel[:, None], np.array([150, 230, 255], np.float32), np.array([40, 50, 70], np.float32))
                    hue = np.array(WIN + [NEON["pink"], NEON["cyan"]], np.float32)[(hash01((fy * 9).astype(int), (fz * 14).astype(int)) * 7).astype(int) % 7]
                    vc = np.where(cans[:, None], hue, vc)
                    c[vm] = vc * 0.35
                    e[vm] = vc * np.where(panel, 0.85, 0.15)[:, None]
            base[m], emit[m] = c, e
        # the ground: wet asphalt, puddles that mirror, a manhole, a drain
        g = surf == 3
        Xg, Zg = X3[g], Z3[g]
        pud = np.sin(Xg * 1.3 + np.sin(Zg * 0.35) * 2) * np.sin(Zg * 0.5 + Xg * 0.4) + 0.35 * np.sin(Zg * 1.7 + 1)
        self.puddle = np.zeros((H, W), np.float32)
        self.puddle[g] = np.clip((pud - 0.15) * 3, 0, 1)
        c = shades(rng, ASPHALT, noise[g], 0.9) * (0.85 + 0.3 * noise[g][:, None])
        c *= (1 - 0.6 * self.puddle[g])[:, None]
        man = np.hypot(Xg - 0.4, (Zg - 12.5)) < 0.42
        c[man] = np.array([46, 44, 52], np.float32) * (0.7 + 0.3 * (np.sin(Xg[man] * 40) > 0))[:, None]
        base[g] = c
        # the megatower: a cliff of small windows
        tw = surf == 4
        Xt, Yt = X3[tw], Y3[tw]
        c = np.full((int(tw.sum()), 3), 12, np.float32) + np.array([6, 6, 16], np.float32)
        w = (np.abs((Xt % 2.2) - 1.1) < 0.45) & (np.abs((Yt % 3.0) - 1.5) < 0.7)
        hs = hash01(np.floor(Xt / 2.2).astype(int), np.floor(Yt / 3.0).astype(int))
        e = np.zeros_like(c)
        e[w & (hs < 0.5)] = np.array(WIN, np.float32)[(hs[w & (hs < 0.5)] * 10).astype(int) % len(WIN)] * 0.35
        base[tw], emit[tw] = c, e
        # neon spill on walls and ground, worked out at half size
        h2, w2 = H // 2, W // 2
        Xs, Ys, Zs = (a[:2 * h2:2, :2 * w2:2] for a in (X3, Y3, Z3))
        spill = np.zeros((h2, w2, 3), np.float32)
        far = self.depth[:2 * h2:2, :2 * w2:2] > ZEND + 5
        for S_ in self.signs:
            cx, cy, cz = float(S_["X"].mean()), float(S_["Y"].mean()), float(S_["z"].mean())
            d2 = (Xs - cx) ** 2 + (Ys - cy) ** 2 + (Zs - cz) ** 2
            spill += S_["col"] * (np.minimum(len(S_["X"]), 600) / 600 * 3.0 / (3.0 + d2))[..., None] * (~far)[..., None]
        for lx, ly, lz in self.lanterns:
            d2 = (Xs - lx) ** 2 + (Ys - ly) ** 2 + (Zs - lz) ** 2
            spill += np.array([255, 90, 50], np.float32) * (0.25 * 4.0 / (4.0 + d2))[..., None]
        spill = np.repeat(np.repeat(spill, 2, 0), 2, 1)
        sp = np.zeros((H, W, 3), np.float32)
        sp[:2 * h2, :2 * w2] = spill
        refl = (1 + 1.5 * self.puddle)[..., None]                           # wet ground picks it up strongly
        on = base * 0.62 + emit + sp * 0.30 * np.where(g[..., None], refl, 1.0)
        off = base * 0.32 + emit * 0.05
        flash = base * np.array([0.9, 0.95, 1.25], np.float32) * 1.6 + (sky[..., None] * np.array([80, 90, 140], np.float32))
        # stipple everything, so it reads as dots
        for img in (on, off, flash):
            r2 = np.random.default_rng(9)
            n = int(W * H * 0.45)
            x, y = r2.uniform(0, W, n), r2.uniform(0, H, n)
            xi, yi = x.astype(int), y.astype(int)
            stamp(img, x, y, img[yi, xi] * (0.82 + 0.36 * r2.random((n, 1))), r2.choice([1.0, 1.5, 2.0], n, p=[0.35, 0.45, 0.2]))
        self.img_on, self.img_off, self.img_flash = on, off, flash - off
        # when each pixel's block comes back on (far first, rushing toward us)
        z = np.minimum(self.depth, ZEND)
        self.t_on = (DARK1 + 5.0 * (1 - z / ZEND)).astype(np.float32)

    # ---- the moving parts ---------------------------------------------------------------------------
    def build_motion(self):
        rng = self.rng
        c = self.cam
        self.sign_scr = []
        for S_ in self.signs:
            x, y = c.proj(S_["X"], S_["Y"], S_["z"])
            rx, ry = c.proj(S_["X"], -S_["Y"], S_["z"])                    # its reflection in the wet ground
            self.sign_scr.append((x, y, rx, ry))
        n = 2600                                                              # rain
        self.rain = dict(X=rng.uniform(XL, XR, n), Y=rng.uniform(0, 16, n), z=2.0 * (60 / 2.0) ** rng.random(n),
                         v=rng.uniform(8, 11, n))
        n = 70                                                                # rings in the puddles
        self.rings = dict(X=rng.uniform(XL + 0.3, XR - 0.3, n), z=2.5 * (40 / 2.5) ** rng.random(n), t0=rng.random(n) * 1.2)
        n = 900                                                               # steam
        self.steam = dict(t0=rng.random(n) * 6, sp=rng.normal(0, 1, (3, n)), src=rng.integers(0, 2, n))
        # the billboard on the megatower: a grid of LEDs
        gx, gy = np.meshgrid(np.linspace(-11, 5, 30), np.linspace(34, 96, 110))
        self.board = (gx.ravel(), gy.ravel())
        self.lamp_seed = rng.uniform(0, 100, len(self.lanterns))
        self.lamp_r, self.lamp_a = rng.random(110), rng.uniform(0, TAU, 110)
        n = 26                                                                # flying traffic over the plaza
        self.cars = dict(lane=rng.integers(0, 3, n), ph=rng.random(n), dirn=np.where(rng.random(n) < 0.5, -1, 1))

    def power(self, t, z, seed, broken):
        """How lit a sign is: steady, stuttering in the surge, dark in the blackout, back on in the cascade."""
        if t < SURGE:
            return flicker(t, seed, 1.0, broken)
        if t < DARK0:
            k = (t - SURGE) / (DARK0 - SURGE)
            on = math.sin(t * (20 + 30 * k) + seed) > -0.2 - 0.9 * (1 - k)
            return (0.25 + 0.75 * on) * (1 - 0.3 * k) * flicker(t, seed, 3.0, 1.0)
        ton = DARK1 + 5.0 * (1 - min(z, ZEND) / ZEND)
        if t < ton:
            return 0.0
        a = t - ton
        if a < 0.5:                                                          # the tube catching
            return float(math.sin(a * 80 + seed) > 0) * a * 2
        return flicker(t, seed, 1.0, broken)

    def draw_city(self, img, t):
        """The far view: the megatower's screen and the traffic crossing the plaza."""
        c, L = self.cam, self.L
        gx, gy = self.board
        u = (gx + 11) / 16
        v = (gy - 34) / 62
        # a koi swimming up the screen, then a wave of glyph-rain behind it
        kx, ky = 0.5 + 0.28 * math.sin(0.6 * t), (0.15 + 0.12 * t) % 1.3 - 0.15
        body = np.exp(-(((u - kx) / 0.22) ** 2 + ((v - ky) / 0.07) ** 2) * 2)
        tail = np.exp(-(((u - kx - 0.05 * math.sin(4 * t)) / 0.14) ** 2 + ((v - ky + 0.11) / 0.05) ** 2) * 2)
        rainc = (np.sin(u * 37 + np.floor(v * 60 + t * 8) * 1.7) > 0.7) * ((v * 9 + t * 1.5 + u * 3) % 1 < 0.35)
        b = np.clip(body + tail, 0, 1)
        col = np.where(b[:, None] > 0.3, np.array(NEON["pink"], np.float32), np.array(NEON["cyan"], np.float32))
        bright = np.maximum(b, rainc * 0.35) * self.city_power(t)
        on = bright > 0.05
        x, y = c.proj(gx[on], gy[on], np.full(int(on.sum()), ZTOWER - 0.5))
        L.tube(x, y, col[on], bright[on] * 0.8, core=0.6, halo=0.9, bloom=1.4)
        C = self.cars                                                        # flying cars
        lanes = [(46.0, 85.0, 0.7), (58.0, 92.0, 1.0), (70.0, 80.0, 1.3)]
        Ys = np.array([l[0] for l in lanes])[C["lane"]]
        zs = np.array([l[1] for l in lanes])[C["lane"]]
        sp = np.array([l[2] for l in lanes])[C["lane"]]
        X = ((C["ph"] + 0.012 * t * sp * C["dirn"]) % 1.0 - 0.5) * 120
        x, y = c.proj(X, Ys, zs)
        xi, yi = np.clip(x.astype(int), 0, self.W - 1), np.clip(y.astype(int), 0, self.H - 1)
        vis = (self.depth[yi, xi] > zs) & (x > 0) & (x < self.W) & (y > 0) & (y < self.H)
        col = np.where(C["dirn"][:, None] > 0, np.array([255, 240, 220], np.float32), np.array([255, 40, 60], np.float32))
        for k in range(6):                                                   # light trails
            L.add(x[vis] - C["dirn"][vis] * k * 2.0, y[vis], col[vis] * (0.9 - 0.13 * k), 1.0)

    def city_power(self, t):
        if t < DARK0:
            return 1.0
        return float(ss(t, DARK1 - 0.5, DARK1 + 1.0))

    def draw_signs(self, img, t):
        L = self.L
        refl_x, refl_y, refl_c = [], [], []
        for S_, (x, y, rx, ry) in zip(self.signs, self.sign_scr):
            p = self.power(t, float(S_["z"].mean()), S_["seed"], S_["broken"])
            if p <= 0.01:
                continue
            L.tube(x, y, S_["col"], p, core=1.0, halo=1.0, bloom=1.0)
            k = slice(None, None, 3)
            refl_x.append(rx[k])
            refl_y.append(ry[k])
            refl_c.append(np.broadcast_to(S_["col"] * p, (len(rx[k]), 3)))
        for (lx, ly, lz), seed in zip(self.lanterns, self.lamp_seed):       # red paper lanterns
            p = self.power(t, lz, seed, 0.0)
            if p <= 0.01:
                continue
            sw = 0.12 * math.sin(1.3 * t + seed)
            r = np.sqrt(self.lamp_r)
            a = self.lamp_a
            X = lx + sw + 0.28 * r * np.cos(a)
            Y = ly - 0.1 + 0.36 * r * np.sin(a)
            x, y = self.cam.proj(X, Y, np.full(len(X), lz))
            rib = 0.75 + 0.25 * (np.abs(np.sin(a * 0.5) * r) > 0.3)
            col = np.array([255, 60, 30], np.float32) * (1.3 - 0.6 * r)[:, None] * rib[:, None]
            L.tube(x, y, col, p * 0.45, core=0.6, halo=1.0, bloom=1.4)
            rx, ry = self.cam.proj(X[::8], -Y[::8], np.full(len(X[::8]), lz))
            refl_x.append(rx)
            refl_y.append(ry)
            refl_c.append(np.broadcast_to(np.array([255, 70, 40], np.float32) * p, (len(rx), 3)))
        return refl_x, refl_y, refl_c

    def draw_reflections(self, t, xs, ys, cs):
        """Light on the wet ground: each source stretched into a trembling vertical streak, strongest in puddles."""
        if not xs:
            return
        x, y, c = np.concatenate(xs), np.concatenate(ys), np.concatenate(cs)
        W, H = self.W, self.H
        xi, yi = np.clip(x.astype(int), 0, W - 1), np.clip(y.astype(int), 0, H - 1)
        ok = (self.surf[yi, xi] == 3) & (y < H) & (x > 0) & (x < W)
        x, y, c = x[ok], y[ok], c[ok]
        pud = self.puddle[yi[ok], xi[ok]]
        dist = (y - self.cam.vy) / (H - self.cam.vy)
        k = (0.10 + 0.55 * pud)[:, None]
        for j in range(5):
            yy = y + j * (3 + 18 * dist) * self.S / 1000
            wob = (2 + 9 * dist) * self.S / 1000 * np.sin(yy * 0.35 + t * 7 + x * 0.05) * (1 - 0.6 * pud)
            self.L.soft(x + wob, yy, c * k * (1.5 - 0.25 * j))

    def draw_rain(self, img, t):
        R, c, L = self.rain, self.cam, self.L
        Y = (R["Y"] - R["v"] * t) % 16.0
        X = R["X"] + 0.08 * Y
        x0, y0 = c.proj(X, Y, R["z"])
        x1, y1 = c.proj(X + 0.05, Y + 0.45, R["z"])
        xi, yi = np.clip(x0.astype(int), 0, self.W - 1), np.clip(y0.astype(int), 0, self.H - 1)
        vis = (x0 > 0) & (x0 < self.W) & (y0 > 0) & (y0 < self.H) & (self.depth[yi, xi] >= R["z"] - 0.2)
        light = img[yi[vis], xi[vis]]                                        # rain lit by whatever is behind it
        col = light * 0.35 + 12
        for k in (0.0, 0.33, 0.66, 1.0):
            L.add(x0[vis] + (x1 - x0)[vis] * k, y0[vis] + (y1 - y0)[vis] * k, col * (0.9 - 0.25 * k), 1.0)
        self.rain_vis = (X[vis], Y[vis], R["z"][vis], x0[vis], y0[vis])
        G = self.rings                                                       # rings in the puddles
        age = (t + G["t0"]) % 1.2
        rad = 0.05 + 0.25 * age
        a = np.linspace(0, TAU, 18)[None, :]
        X = G["X"][:, None] + rad[:, None] * np.cos(a)
        z = G["z"][:, None] + rad[:, None] * np.sin(a)
        x, y = c.proj(X, np.zeros_like(X), z)
        xi, yi = np.clip(x.astype(int), 0, self.W - 1), np.clip(y.astype(int), 0, self.H - 1)
        p = self.puddle[yi, xi] * (1 - age / 1.2)[:, None]
        on = p > 0.05
        L.add(x[on], y[on], (img[yi[on], xi[on]] * 0.25 + 18) * p[on][:, None], 1.0)

    def draw_steam(self, img, t):
        St, c = self.steam, self.cam
        age = (t + St["t0"]) % 6.0
        f = age / 6.0
        src = St["src"]
        X = np.where(src == 0, XR - 0.2 - 1.2 * f, 0.4) + St["sp"][0] * (0.05 + 0.5 * f)
        Y = np.where(src == 0, 0.7, 0.05) + 2.8 * f ** 0.8 + St["sp"][1] * 0.2 * f
        z = np.where(src == 0, 9.0, 12.5) + St["sp"][2] * (0.05 + 0.5 * f)
        x, y = c.proj(X, Y, z)
        xi, yi = np.clip(x.astype(int), 0, self.W - 1), np.clip(y.astype(int), 0, self.H - 1)
        ok = (x > 0) & (x < self.W) & (y > 0) & (y < self.H)
        dens = np.sin(np.pi * f) * (1 - f) * ok
        col = (img[yi, xi] * 0.3 + np.array([40, 40, 50])) * (dens * 0.35)[:, None]
        self.L.bloom.splat(x[ok], y[ok], col[ok] * 1.6)
        self.L.add(x[ok][::3], y[ok][::3], col[ok][::3] * 0.5, 2.0)

    def drone(self, t):
        return np.array([0.6 * math.sin(0.31 * t), 9.5 + 0.5 * math.sin(0.7 * t), 15.0 + 3.0 * math.sin(0.2 * t)])

    def draw_drone(self, img, dots, t, dark):
        c, L = self.cam, self.L
        D = self.drone(t)
        T = np.array([2.0 * math.sin(0.45 * t), 0.0, D[2] - 5.5 + 3.5 * math.sin(0.33 * t + 1)])
        x, y = c.proj(D[0] + np.array([-0.35, 0.35, -0.2, 0.2, 0]), D[1] + np.array([0, 0, 0.1, 0.1, -0.05]), np.full(5, D[2]))
        dots.append((x, y, np.array([30, 30, 36], np.float32), 2.0 * self.S / 1000 + 0.6, np.full(5, -D[2])))
        blink = (t * 1.5) % 1.0 < 0.15
        L.add(x[:2], y[:2] + 1, np.array([[255, 30, 40], [40, 120, 255]]) * (1.2 if blink else 0.3), 1.6)
        # the searchlight: a cone of light through the rain, a pool on the ground
        k = 1.0 + 0.8 * dark
        rng = np.random.default_rng(int(t * 24))
        n = 700
        s = rng.random(n) ** 0.8
        ax = D + (T - D) * s[:, None]
        spread = 0.08 + 0.9 * s
        ang = rng.uniform(0, TAU, n)
        r = spread * np.sqrt(rng.random(n))
        P = ax + np.stack([np.cos(ang) * r, np.zeros(n), np.sin(ang) * r], 1)
        bx, by = c.proj(P[:, 0], P[:, 1], P[:, 2])
        L.add(bx, by, np.array([200, 225, 255]) * (0.10 * k * (1 - 0.6 * s))[:, None], 1.0)
        L.bloom.splat(bx[::2], by[::2], np.array([150, 180, 255]) * (0.16 * k))
        a = np.linspace(0, TAU, 60)
        for rr in (0.4, 0.75, 1.0):
            px, py = c.proj(T[0] + rr * np.cos(a), np.zeros(60), T[2] + rr * np.sin(a))
            L.add(px, py, np.array([210, 230, 255]) * 0.5 * k * (1.2 - rr), 1.6)
        X, Y, Z, rx, ry = self.rain_vis                                      # raindrops sparkle inside the beam
        if len(X):
            v = T - D
            w = np.stack([X - D[0], Y - D[1], Z - D[2]], 1)
            s2 = np.clip((w @ v) / (v @ v), 0, 1)
            d = np.linalg.norm(w - s2[:, None] * v[None], axis=1)
            inb = d < (0.08 + 0.9 * s2)
            L.add(rx[inb], ry[inb], np.array([220, 235, 255]) * 0.8 * k, 1.0)
        return T

    def walker(self, t):
        return 0.55 + 0.25 * math.sin(0.21 * t), 31.0 - 0.44 * t

    def draw_walker(self, dots, t, dark):
        c, L = self.cam, self.L
        X0, z0 = self.walker(t)
        rng = np.random.default_rng(3)
        ph = t * 3.6
        parts = []

        def capsule(a, b, rad, n, shade=1.0):
            a, b = np.asarray(a, float), np.asarray(b, float)
            q = rng.random(n)
            ang, rr = rng.uniform(0, TAU, n), np.sqrt(rng.random(n))
            P = a[None] + (b - a)[None] * q[:, None] + rad * rr[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)
            parts.append(np.hstack([P, np.cos(ang)[:, None] * rr[:, None] ** 6, np.full((n, 1), shade)]))

        for s in (-1, 1):                                                    # legs
            sw = 0.16 * math.sin(ph + (0 if s > 0 else math.pi))
            capsule((0.1 * s, 0.85), (0.1 * s + 0.02, 0.45), 0.07, 90)
            capsule((0.1 * s + 0.02, 0.45), (0.1 * s, 0.05 + 0.05 * max(sw, 0)), 0.06, 80)
        capsule((-0.17, 0.85), (0.17, 0.85), 0.12, 120)                       # a long coat
        for k in range(5):
            q = k / 4
            capsule((-0.20 - 0.02 * q, 0.9 + 0.12 * k), (0.20 + 0.02 * q, 0.9 + 0.12 * k), 0.09, 90)
        capsule((0.16, 1.40), (0.10, 1.62), 0.05, 50)                        # the arm up to the handle
        capsule((0.0, 1.62), (0.0, 1.72), 0.10, 140)                         # head
        P = np.vstack(parts)
        x, y = c.proj(X0 + P[:, 0], P[:, 1], np.full(len(P), z0))
        side = P[:, 2]
        rim = np.clip(side, 0, 1)[:, None] * np.array(NEON["cyan"], np.float32) + np.clip(-side, 0, 1)[:, None] * np.array(NEON["pink"], np.float32)
        col = np.array([18, 16, 24], np.float32) + rim * 0.55 * (1 - 0.8 * dark)
        dots.append((x, y, col, np.clip(c.f * 0.03 / z0, 1.0, 3.0), np.full(len(x), -z0)))
        # the clear umbrella: a faint dome beaded with rain, a ring of light at its rim, the shaft
        a = rng.uniform(0, TAU, 500)
        e = rng.uniform(0, 1, 500) ** 0.6
        U = np.stack([0.62 * e * np.cos(a), 2.05 + 0.32 * np.sqrt(1 - e ** 2), 0.62 * e * np.sin(a) * 0.35], 1)
        x, y = c.proj(X0 + U[:, 0], U[:, 1], z0 + U[:, 2])
        L.add(x, y, np.array([170, 200, 230]) * 0.10, 1.0)
        drop = rng.random(500) < 0.12
        L.add(x[drop], y[drop], np.array([230, 240, 255]) * 0.7, 1.0)
        a = np.linspace(0, TAU, 90)
        rx, ry = c.proj(X0 + 0.62 * np.cos(a), np.full(90, 2.05), z0 + 0.62 * np.sin(a) * 0.35)
        L.tube(rx, ry, NEON["cyan"], 0.9, core=0.8, halo=1.0, bloom=1.2)
        q = np.linspace(0, 1, 30)
        sx, sy = c.proj(np.full(30, X0 + 0.12), 1.55 + 0.82 * q, np.full(30, z0))
        dots.append((sx, sy, np.array([60, 60, 70], np.float32), 1.0, np.full(30, -z0 + 0.01)))
        mx, my = c.proj(X0 + 0.62 * np.cos(a[::3]), np.full(30, -2.05), z0 + 0.62 * np.sin(a[::3]) * 0.35)
        return mx, my

    def flash(self, t):
        return sum(k * math.exp(-((t - t0) / 0.05) ** 2) for t0, k in LIGHTNING)

    # ---- the frame ---------------------------------------------------------------------------------
    def render(self, t):
        if t < SURGE:
            img = self.img_on.copy()
        elif t < DARK0:                                                      # the surge: the whole district dims and stutters
            k = (t - SURGE) / (DARK0 - SURGE)
            w = (0.55 + 0.45 * (math.sin(t * 37) > -0.3 + 0.6 * k)) * (1 - 0.5 * k)
            img = self.img_off + (self.img_on - self.img_off) * w
        elif t < DARK1:
            img = self.img_off.copy()
        elif t < DARK1 + 6:                                                  # blocks coming back, far to near
            w = np.clip((t - self.t_on) / 0.4, 0, 1)
            img = self.img_off + (self.img_on - self.img_off) * w[..., None]
        else:
            img = self.img_on.copy()
        fl = self.flash(t)
        if fl > 0.01:
            img += self.img_flash * fl
        dark = float(ss(t, DARK0 - 0.3, DARK0) * (1 - ss(t, DARK1, DARK1 + 4)))
        dots = []
        self.draw_city(img, t)
        rx, ry, rc = self.draw_signs(img, t)
        self.draw_steam(img, t)
        self.draw_rain(img, t)
        T = self.draw_drone(img, dots, t, dark)
        mx, my = self.draw_walker(dots, t, dark)
        rx.append(mx)
        ry.append(my)
        rc.append(np.broadcast_to(np.array(NEON["cyan"], np.float32) * 0.9, (len(mx), 3)))
        tx, ty = self.cam.proj(T[0] + np.array([0.0, 0.3, -0.3]), np.zeros(3), T[2] + np.array([0.0, 0.2, -0.2]))
        rx.append(tx)
        ry.append(ty - 2)
        rc.append(np.broadcast_to(np.array([200, 225, 255], np.float32) * (0.4 + 0.6 * dark), (3, 3)))
        self.draw_reflections(t, rx, ry, rc)
        paint(img, dots)
        self.L.draw(img, 2)
        return finish(img, self.lut)


if __name__ == "__main__":
    run(NeonAlley, "Neon alley", "neon_alley", T_END, STAGES)
