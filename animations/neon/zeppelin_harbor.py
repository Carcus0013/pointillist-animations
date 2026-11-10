"""
zeppelin_harbor.py - an airship docking above a dieselpunk city, in dots.

Dusk over an art-deco metropolis choked in amber smog. Stepped towers bristle with fins and
spires, their windows burning in long vertical bands; smokestacks pour plumes lit orange by
the furnaces under them; rays of light fan up behind the tallest tower like an old poster.
Out of the haze comes an enormous riveted airship, droning, its engines' propellers a blur,
escorted by circling fighters. The city's searchlights find it and follow it in. It swings
broadside, turns, and creeps nose-first to the mooring mast on top of the great tower -
locks on in a burst of sparks - the mast's beacons turn green, and every searchlight in the
city crosses on it in salute.

    python zeppelin_harbor.py            interactive window (Space pause, R restart, S save frame, Esc quit)
    python zeppelin_harbor.py --png      save five stages
    python zeppelin_harbor.py --gif      save the whole minute as a GIF
"""
import math

import numpy as np

from neon import TAU, Cam, Lights, finish, keyed, paint, run, shades, ss, stamp, tone_lut, unit

T_END = 60.0
STAGES = [6.0, 22.0, 36.0, 48.3, 56.0]
AMBER = np.array([255, 165, 60], np.float32)
CREAM = np.array([255, 220, 160], np.float32)
MAST = np.array([150.0, 470.0, 900.0])        # the mooring mast's tip (X, Y, z)
DOCK = 48.0
HULL_L, HULL_R = 330.0, 40.0
PATH = [  # time, nose position, heading (degrees, 0 = +X), pitch
    (0.0, (-1500.0, 620.0, 2600.0), 10.0), (14.0, (-950.0, 600.0, 1900.0), 25.0), (26.0, (-520.0, 540.0, 1350.0), 5.0),
    (36.0, (-180.0, 500.0, 1020.0), -12.0), (44.0, (90.0, 474.0, 920.0), -2.0), (DOCK, (MAST[0], MAST[1], MAST[2]), 0.0),
    (60.0, (MAST[0], MAST[1], MAST[2]), 0.0),
]


class ZeppelinHarbor:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.S = S = min(W, H)
        self.cam = Cam(W, H, 0.5 * W, 0.72 * H, 0.95 * S, 150.0)
        self.lut = tone_lut(1.25, 5.0)
        self.L = Lights(W, H, 4)
        self.rng = np.random.default_rng(14)
        self.build_city()
        self.build_ship()
        self.build_extras()

    # ---- the city ----------------------------------------------------------------------------------
    def build_city(self):
        W, H, c = self.W, self.H, self.cam
        rng = np.random.default_rng(3)
        yy = np.arange(H, dtype=np.float32)[:, None] / H
        hz = c.vy / H
        stops = ((0.0, (12, 20, 32)), (0.3, (30, 32, 42)), (0.5, (84, 60, 50)), (hz - 0.04, (186, 116, 58)), (hz, (214, 140, 66)))
        img = np.stack([np.interp(yy, [s_[0] for s_ in stops], [s_[1][k] for s_ in stops]) for k in range(3)], -1)[..., 0, :]
        img = np.repeat(img[:, None, :], W, 1).astype(np.float32)
        xs = np.arange(W)[None, :] / W
        smog = (0.5 + 0.5 * np.sin(yy * 46 + np.sin(xs * 6) * 2)) ** 3 * np.exp(-((yy - hz + 0.12) / 0.16) ** 2)
        img = img * (1 - 0.2 * smog[..., None]) + np.array([150, 100, 60], np.float32) * (0.2 * smog)[..., None]
        # faint rays fanning up behind the great tower, like an old poster
        mx, _ = c.proj(MAST[0], 0, MAST[2])
        oy = c.vy
        ang = np.arctan2(oy - np.arange(H)[:, None], np.arange(W)[None, :] - mx)
        wedge = np.exp(-((ang - math.pi / 2) / 0.7) ** 2)
        rays = (np.sin(ang * 40) > 0.55) * wedge * (np.arange(H)[:, None] < oy)
        img += np.array([150, 90, 30], np.float32) * (rays * 0.11 * np.clip((oy - np.arange(H)[:, None]) / H, 0, 1) ** 0.3)[..., None]
        depth = np.full((H, W), 1e5, np.float32)
        # below the skyline: the city spread out, streets strung with sodium lights
        ys = np.arange(H)[:, None].astype(np.float64)
        below = ys > c.vy + 1
        zg = np.where(below, c.f * c.hc / np.maximum(ys - c.vy, 1e-3), 1e5) * np.ones((1, W))
        Xg = (np.arange(W)[None, :] - c.vx) * zg / c.f
        fog = np.clip((zg - 300) / 3500, 0, 1) ** 0.6
        sx_, sz_ = (Xg % 70.0), (zg % 55.0)
        street = (sx_ < 5) | (sz_ < 4)
        lamp = ((sx_ < 5) & ((zg % 18.0) < 2.5)) | ((sz_ < 4) & ((Xg % 18.0) < 2.5))   # strings of street lamps
        block = np.array([30, 26, 28], np.float32)
        hsh = (np.sin(np.floor(Xg / 70) * 12.9898 + np.floor(zg / 55) * 78.233) * 43758.5453) % 1
        roofs = block * (0.6 + 0.8 * hsh[..., None])
        win = ((np.sin(Xg * 1.3) * np.sin(zg * 1.1)) > 0.7) & ~street & (hsh[...] > 0.3)
        sodium = np.array([255, 150, 50], np.float32)
        hazec = np.array([200, 130, 70], np.float32)
        ground = np.where(street[..., None], np.array([46, 36, 30], np.float32) + sodium * 0.12, roofs)             + lamp[..., None] * sodium * 0.75 + win[..., None] * AMBER * 0.30
        ground = ground * (1 - fog[..., None]) + hazec * fog[..., None] * 0.8
        img = np.where(below[..., None], ground, img).astype(np.float32)
        depth = np.where(below, zg, depth).astype(np.float32)
        # towers: art deco - fluted shafts, setbacks with floodlit ledges, crowns and spires
        towers = []
        for z0, z1, n in ((2600, 3400, 30), (1700, 2400, 20), (1100, 1500, 10), (500, 750, 6)):
            for _ in range(n):
                z = rng.uniform(z0, z1)
                X = rng.uniform(-1.1, 1.1) * z * 0.55
                towers.append((z, X, rng.uniform(40, 80), rng.uniform(140, 360), int(rng.integers(2, 5)), rng.random() < 0.6, False))
        towers.append((MAST[2] + 5, MAST[0], 120.0, MAST[1] - 60, 5, False, True))     # the great tower
        towers.sort(key=lambda T: -T[0])
        self.beacons = []
        for z, X, w, h, tiers, spire, hero in towers:
            fog = min(1.0, max(0.0, (z - 400) / 3000)) ** 0.7
            body = np.array([46, 40, 42], np.float32) * (1 - fog) + hazec * fog * 0.8
            tops = []
            for k in range(tiers):
                hk0 = 0.0 if k == 0 else h * (0.55 + 0.45 * (k - 1) / tiers)
                hk1 = h * (0.55 + 0.45 * k / tiers)
                wk = w * (1 - 0.17 * k)
                xa, ya = c.proj(X - wk / 2, hk1, z)
                xb, yb = c.proj(X + wk / 2, hk0 - 20, z)
                r0, r1 = int(max(ya, 0)), int(min(yb, H))
                c0, c1 = int(max(xa, 0)), int(min(xb, W))
                if r0 >= r1 or c0 >= c1:
                    continue
                nfl = 10 if hero else 7
                u = (np.arange(c0, c1) - xa) / max(xb - xa, 1)
                piers = 0.75 + 0.35 * (np.abs(((u * nfl) % 1) - 0.5) < 0.18)                # fluted vertical piers
                lit_side = 0.85 + 0.3 * (u > 0.55)
                v = (np.arange(r0, r1) - ya) / max(yb - ya, 1)
                flood = 1 + (1.6 if hero else 0.7) * np.exp(-v / 0.08)[:, None] * (1 - fog)   # light washing each setback
                img[r0:r1, c0:c1] = body * (piers * lit_side)[None, :, None] * flood[..., None]
                depth[r0:r1, c0:c1] = z
                if fog < 0.85:                                                # windows in long vertical strips
                    strip = np.abs(((u * nfl) % 1) - 0.5) > 0.30
                    rows = (np.sin(np.arange(r0, r1) * (c.f * 0.9 / z)) > 0.1)[:, None]
                    lit = strip[None, :] & rows & (rng.random((r1 - r0, c1 - c0)) < 0.55)
                    block_ = img[r0:r1, c0:c1]
                    block_[lit] += AMBER * (0.45 * (1 - fog))
                tops.append((xa, xb, ya))
            if hero and tops:                                                 # its crown: stacked golden arches
                xa, xb, ya = tops[-1]
                for k in range(5):
                    q = np.linspace(-1, 1, 120)
                    r_ = (xb - xa) / 2 * (1 - 0.18 * k)
                    ax_ = (xa + xb) / 2 + q * r_
                    ay_ = ya - k * c.f * 9 / z - np.sqrt(np.clip(1 - q ** 2, 0, 1)) * r_ * 0.5
                    stamp(img, ax_, ay_, np.array([255, 190, 90], np.float32) * (0.9 - 0.1 * k), 1.5)
                    depth[np.clip(ay_.astype(int), 0, H - 1), np.clip(ax_.astype(int), 0, W - 1)] = z
            if spire and not hero and tops:
                xa, xb, ya = tops[-1]
                q = np.linspace(0, 1, 50)
                sxp = np.full(50, (xa + xb) / 2)
                syp = ya - q * c.f * 45 / z
                stamp(img, sxp, syp, body * 1.2, float(np.clip(c.f * 2.5 / z, 1, 2.2)))
                self.beacons.append((sxp[-1], syp[-1], z))
        # the mooring mast on the great tower: a slim lattice with a docking ring
        q = np.linspace(0, 1, 200)
        for dx in (-3.0, 3.0):
            mx_, my_ = c.proj(np.full(200, MAST[0] + dx * (1 - q)), MAST[1] - 70 + q * 70, np.full(200, MAST[2]))
            stamp(img, mx_, my_, np.array([90, 76, 66], np.float32), 1.3)
            depth[np.clip(my_.astype(int), 0, H - 1), np.clip(mx_.astype(int), 0, W - 1)] = MAST[2]
        for hgt in np.linspace(MAST[1] - 65, MAST[1] - 5, 7):
            lx, ly = c.proj(MAST[0] + np.linspace(-3, 3, 20), np.full(20, hgt), np.full(20, MAST[2]))
            stamp(img, lx, ly, np.array([90, 76, 66], np.float32), 1.0)
        a = np.linspace(0, TAU, 80)
        rx, ry = c.proj(MAST[0] + 5 * np.cos(a), MAST[1] + 2 * np.sin(a), np.full(80, MAST[2]))
        stamp(img, rx, ry, np.array([200, 160, 90], np.float32), 1.3)
        # factory chimneys on the right
        self.stacks = []
        for X, z, h in ((520.0, 650.0, 170.0), (590.0, 700.0, 200.0), (660.0, 640.0, 150.0)):
            xa, ya = c.proj(X - 7, h, z)
            xb, yb = c.proj(X + 7, 0, z)
            sl = (slice(int(max(ya, 0)), int(min(yb, H))), slice(int(max(xa, 0)), int(min(xb, W))))
            img[sl] = np.array([44, 34, 32], np.float32) * (1 + 0.5 * (np.sin(np.arange(sl[0].start, sl[0].stop) * 0.5) > 0.8))[:, None, None]
            depth[sl] = z
            self.stacks.append((X, h, z))
        # stipple it all
        n = int(W * H * 0.4)
        x, y = rng.uniform(0, W, n), rng.uniform(0, H, n)
        stamp(img, x, y, img[y.astype(int), x.astype(int)] * (0.86 + 0.28 * rng.random((n, 1))), rng.choice([1.0, 1.5, 2.0], n))
        self.img, self.depth = img, depth

    # ---- the airship ---------------------------------------------------------------------------------
    def build_ship(self):
        rng = np.random.default_rng(8)
        n_s, n_t = 150, 72
        s = np.repeat(np.linspace(0, 1, n_s), n_t)                           # 0 nose .. 1 tail
        th = np.tile(np.linspace(0, TAU, n_t, endpoint=False), n_s)
        r = HULL_R * np.sin(np.pi * np.clip(0.02 + 0.98 * s, 0, 1) ** 0.85) ** 0.55 * (1 - 0.25 * s ** 3)
        ring = (np.abs(((s * 18) % 1) - 0.5) > 0.46)                          # girder rings and longitudinal panels
        rib = (np.abs(((th / TAU * 24) % 1) - 0.5) > 0.44)
        self.hull = dict(s=s, th=th, r=r, line=(ring | rib), tone=rng.random(len(s)))
        fins = []                                                             # four fins at the tail
        for k in range(4):
            a = k * math.pi / 2 + math.pi / 4
            u, v = rng.random((2, 700))
            fs = 0.80 + 0.2 * u
            span = HULL_R * (0.25 + 1.0 * (1 - v) * np.clip((fs - 0.80) / 0.2, 0, 1) ** 0.7)
            fins.append((fs, np.full(700, a), span + HULL_R * 0.25 * (1 - fs), v))
        self.fins = [np.concatenate(z) for z in zip(*fins)]
        # the gondola, engines with propellers
        u, v = rng.random((2, 1400))
        self.gondola = (0.28 + 0.16 * u, v)
        self.engines = [(0.55, -1), (0.55, 1), (0.70, -1), (0.70, 1)]

    def ship_pose(self, t):
        ts = [p[0] for p in PATH]
        i = int(np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(PATH) - 2))
        t0, t1 = ts[i], ts[i + 1]
        w = float(ss(t, t0, t1))
        p0, p1 = np.array(PATH[i][1]), np.array(PATH[i + 1][1])
        nose = p0 + (p1 - p0) * w
        hd = math.radians(PATH[i][2] + (PATH[i + 1][2] - PATH[i][2]) * w)
        bob = 0.0 if t >= DOCK else 2.5 * math.sin(0.6 * t)
        nose = nose + np.array([0, bob, 0])
        ax = np.array([math.cos(hd), -0.02, math.sin(hd)])                   # nose direction
        return nose, unit(ax)

    def ship_points(self, t):
        """The whole ship as points: positions, base colours, shading weights."""
        nose, ax = self.ship_pose(t)
        up = np.array([0.0, 1.0, 0.0])
        side = unit(np.cross(ax, up))
        up = np.cross(side, ax)
        Hh = self.hull
        cen = nose[None] - ax[None] * (Hh["s"] * HULL_L)[:, None]
        nrm = np.cos(Hh["th"])[:, None] * side[None] + np.sin(Hh["th"])[:, None] * up[None]
        P = cen + nrm * Hh["r"][:, None]
        lightdir = unit(np.array([0.4, -0.6, -0.7]))                       # warm city glow from below and in front
        sky = np.clip(nrm @ np.array([0, 1.0, 0]), 0, 1)
        warm = np.clip(nrm @ -lightdir, 0, 1)
        base = np.array([190, 182, 170], np.float32) * (0.85 + 0.25 * Hh["tone"])[:, None]
        col = base * (0.30 + 0.40 * sky)[:, None] + AMBER * (0.75 * warm ** 1.3)[:, None] + base * 0.08
        col[Hh["line"]] *= 0.55
        fs, fa, fspan, fv = self.fins
        fc = nose[None] - ax[None] * (fs * HULL_L)[:, None]
        fd = np.cos(fa)[:, None] * side[None] + np.sin(fa)[:, None] * up[None]
        FP = fc + fd * fspan[:, None]
        fcol = np.array([110, 96, 86], np.float32)[None] * (0.5 + 0.3 * fv)[:, None] + AMBER * 0.15
        gu, gv = self.gondola                                                 # the gondola under the front third
        gc = nose[None] - ax[None] * (gu * HULL_L)[:, None] - up[None] * (HULL_R * 0.95 + 6 * gv)[:, None]
        GP = gc + side[None] * ((self.rng_g(len(gu)) - 0.5) * 9)[:, None]
        win = (np.abs(gv - 0.35) < 0.15) & (((gu * 60) % 1) < 0.6)
        gcol = np.where(win[:, None], CREAM * 0.9, np.array([60, 50, 46], np.float32))
        pts = np.vstack([P, FP, GP])
        cols = np.vstack([col, fcol, gcol])
        return pts, cols, nose, ax, side, up, GP[win]

    def rng_g(self, n):
        if not hasattr(self, "_g"):
            self._g = np.random.default_rng(2).random(n)
        return self._g

    def draw_ship(self, dots, t):
        c, L = self.cam, self.L
        pts, cols, nose, ax, side, up, wins = self.ship_points(t)
        x, y = c.proj(pts[:, 0], pts[:, 1], pts[:, 2])
        z = pts[:, 2]
        r = np.clip(c.f * 2.4 / z, 1.0, 3.0)
        fade = np.clip(1 - (z - 1200) / 2000, 0.35, 1)[:, None]              # far off, it is lost in the smog
        haze = np.array([170, 110, 60], np.float32)
        cols = cols * fade + haze * (1 - fade) * 0.7
        dots.append((x, y, cols, r, -z))
        wx, wy = c.proj(wins[:, 0], wins[:, 1], wins[:, 2])                   # cabin windows glow
        L.add(wx[::3], wy[::3], CREAM * 0.35, 1.0)
        # engines: nacelles and propeller blur discs
        for es, sgn in self.engines:
            ec = nose - ax * es * HULL_L - up * HULL_R * 0.55 + side * sgn * HULL_R * 1.1
            a = np.linspace(0, TAU, 50)
            disc = ec[None] - ax[None] * 7 + (np.cos(a)[:, None] * side[None] + np.sin(a)[:, None] * up[None]) * 6.5
            px, py = c.proj(disc[:, 0], disc[:, 1], disc[:, 2])
            L.add(px, py, np.array([200, 190, 170]) * 0.12, 1.0)
            bx, by = c.proj(ec[0], ec[1], ec[2])
            L.add(bx, by, AMBER * 0.6, 1.6)
            nx_, ny_ = c.proj(np.array([ec[0]]) + np.array([0, 4, -4]), np.full(3, ec[1]), np.full(3, ec[2]))
            dots.append((nx_, ny_, np.array([50, 44, 40], np.float32), np.clip(c.f * 4 / ec[2], 1, 3), np.full(3, -ec[2] + 1)))
        # navigation lights: red to port, green to starboard, a white blink at the tail
        mid = nose - ax * 0.5 * HULL_L
        for col, sgn in ((np.array([255, 40, 40]), -1), (np.array([40, 255, 90]), 1)):
            p = mid + side * sgn * HULL_R
            px, py = c.proj(p[0], p[1], p[2])
            L.tube(px, py, col, 0.8, core=0.8, halo=1.0, bloom=0.8)
        if (t * 1.2) % 1 < 0.12:
            p = nose - ax * HULL_L
            px, py = c.proj(p[0], p[1], p[2])
            L.tube(px, py, np.array([255, 250, 230]), 1.4, core=1, halo=1, bloom=1)
        return nose, ax

    # ---- searchlights, fighters, smoke ---------------------------------------------------------------
    def build_extras(self):
        rng = self.rng
        self.lights_src = [(-700.0, 0.0, 800.0, 0.0), (-250.0, 0.0, 600.0, 1.3), (300.0, 0.0, 700.0, 2.1),
                           (760.0, 0.0, 900.0, 3.7), (40.0, 0.0, 1300.0, 5.2)]
        n = 1200
        self.smoke = dict(src=rng.integers(0, 3, n), t0=rng.random(n) * 12, sp=rng.normal(0, 1, (3, n)))
        self.planes = [dict(r=90.0, h=40.0, w=0.42, ph=0.0), dict(r=130.0, h=-10.0, w=-0.33, ph=2.0), dict(r=70.0, h=70.0, w=0.55, ph=4.1)]
        n = 1400
        self.motes = dict(x=rng.random(n), y=rng.random(n), z=rng.uniform(300, 900, n), ph=rng.uniform(0, TAU, n))

    def draw_searchlights(self, t, target):
        c, L = self.cam, self.L
        salute = float(ss(t, DOCK + 0.5, DOCK + 3.0))
        follow = float(ss(t, 12, 20))
        for X, Y, z, ph in self.lights_src:
            sweep = np.array([X + 500 * math.sin(0.23 * t + ph), 900 + 250 * math.sin(0.31 * t + ph * 1.7), z + 600])
            aim = sweep + (target - sweep) * (0.55 * follow * (0.5 + 0.5 * math.sin(0.4 * t + ph)) + 0.45 * salute)
            if salute > 0:
                aim = aim + (target - aim) * salute
            src = np.array([X, Y, z])
            d = aim - src
            r = np.random.default_rng(int(t * 20 + ph * 100)).random((2, 600))
            s = r[0] ** 0.6 * 1.3
            spread = 4 + 30 * s
            ang = r[1] * TAU
            perp1 = unit(np.cross(d, [0, 0, 1.0]))
            perp2 = unit(np.cross(d, perp1))
            P = src[None] + d[None] * s[:, None] + (np.cos(ang)[:, None] * perp1[None] + np.sin(ang)[:, None] * perp2[None]) * \
                (spread * np.sqrt(r[0]))[:, None]
            x, y = c.proj(P[:, 0], P[:, 1], P[:, 2])
            k = (1.3 - s) * 0.22
            L.soft(x, y, np.array([255, 230, 180])[None] * k[:, None] * 2.2)
            L.add(x[::2], y[::2], np.array([255, 235, 190])[None] * k[::2, None] * 0.8, 1.0)

    def draw_planes(self, dots, t, center):
        c, L = self.cam, self.L
        if t > DOCK + 6:
            return
        for k, Pl in enumerate(self.planes):
            a = Pl["w"] * t + Pl["ph"]
            pos = center + np.array([Pl["r"] * math.cos(a), Pl["h"] + 8 * math.sin(2 * a), Pl["r"] * 0.6 * math.sin(a)])
            vel = np.array([-math.sin(a) * Pl["r"] * Pl["w"], 0, math.cos(a) * Pl["r"] * 0.6 * Pl["w"]])
            fwd = unit(vel)
            side = unit(np.cross(fwd, [0, 1.0, 0]))
            q = np.linspace(-1, 1, 14)
            body = pos[None] + fwd[None] * (q * 6)[:, None]
            wing = pos[None] + side[None] * (q * 8)[:, None] + fwd[None] * 1.0
            tail = pos[None] - fwd[None] * 5 + side[None] * (q[::2] * 3)[:, None]
            P = np.vstack([body, wing, tail])
            x, y = c.proj(P[:, 0], P[:, 1], P[:, 2])
            dots.append((x, y, np.array([44, 38, 36], np.float32), np.clip(c.f * 1.6 / pos[2], 1, 2.4), np.full(len(x), -pos[2] + 2)))
            ex, ey = c.proj(*(pos - fwd * 6))
            L.add(ex, ey, AMBER * (0.6 + 0.4 * math.sin(t * 30 + k)), 1.4)
            for col, sgn in ((np.array([255, 40, 40]), -1), (np.array([40, 255, 90]), 1)):
                wx, wy = c.proj(*(pos + side * 8 * sgn))
                L.add(wx, wy, col * 0.8, 1.2)

    def draw_smoke(self, t):
        c, L = self.cam, self.L
        Sm = self.smoke
        age = (t + Sm["t0"]) % 12.0
        f = age / 12.0
        X0 = np.array([s[0] for s in self.stacks])[Sm["src"]]
        h0 = np.array([s[1] for s in self.stacks])[Sm["src"]]
        z0 = np.array([s[2] for s in self.stacks])[Sm["src"]]
        X = X0 - 240 * f ** 1.2 + Sm["sp"][0] * (3 + 50 * f)
        Y = h0 + 160 * f ** 0.7 + Sm["sp"][1] * (2 + 30 * f)
        z = z0 + Sm["sp"][2] * 30 * f
        x, y = c.proj(X, Y, z)
        dens = np.sin(np.pi * f) ** 0.8 * (1 - f)
        lit = np.exp(-f * 4)                                                  # lit orange near the stack mouths
        col = (np.array([70, 60, 56])[None] * dens[:, None] + AMBER[None] * (0.6 * lit * dens)[:, None]) * 0.35
        L.soft(x, y, col * 2.0)
        L.add(x[::4], y[::4], col[::4] * 0.6, 1.0)
        if (t * 3) % 1 < 0.5:                                                 # sparks
            r = np.random.default_rng(int(t * 10)).random((2, 30))
            s = self.stacks[int(t) % 3]
            sx, sy = c.proj(s[0] + (r[0] - 0.5) * 20, s[1] + r[1] * 40, s[2])
            L.add(sx, sy, np.array([255, 190, 90]) * 0.9, 1.0)

    def draw_beacons(self, t, docked):
        L = self.L
        for x, y, z in self.beacons:
            on = (t * 0.7 + x * 0.01) % 1 < 0.3
            if on:
                L.add(x, y - 1, np.array([255, 50, 40]) * min(1.0, 900 / z), 1.4)
        mx, my = self.cam.proj(MAST[0], MAST[1] + 6, MAST[2])
        col = np.array([60, 255, 110]) if docked else np.array([255, 60, 40])
        if docked or (t * 1.5) % 1 < 0.4:
            self.L.tube(mx, my, col, 1.2, core=1, halo=1, bloom=1)

    # ---- the frame ---------------------------------------------------------------------------------
    def render(self, t):
        img = self.img.copy()
        dots = []
        nose, ax = self.draw_ship(dots, t)
        center = nose - ax * HULL_L * 0.5
        self.draw_searchlights(t, center)
        self.draw_planes(dots, t, center)
        self.draw_smoke(t)
        self.draw_beacons(t, t >= DOCK)
        if DOCK - 0.1 < t < DOCK + 1.5:                                      # locking on: a burst of sparks
            a = t - DOCK
            r = np.random.default_rng(int(t * 30)).normal(0, 1, (2, 80))
            mx, my = self.cam.proj(MAST[0], MAST[1], MAST[2])
            self.L.add(mx + r[0] * 25 * (a + 0.1), my + r[1] * 18 * (a + 0.1) + 30 * a * a, np.array([255, 220, 150]) * 1.2 * (1 - a / 1.6), 1.2)
        paint(img, dots)
        self.L.draw(img, 2)
        return finish(img, self.lut)


if __name__ == "__main__":
    run(ZeppelinHarbor, "Zeppelin harbour", "zeppelin_harbor", T_END, STAGES)
