"""
holo_sakura.py - a holographic cherry tree on a rooftop above a neon city, in dots.

Rain on a rooftop high above the megacity. A projector on the wet concrete flickers awake and
throws up a cherry tree made of light: the trunk and branches draw themselves upward in cyan
scanlines, then blossoms burst open in a wave of pink. Pixel petals peel away on the wind,
fall through the rain and fizzle into sparks on the puddles, where the whole tree trembles
upside down. Beyond the parapet the towers glitter, flying traffic streams between them and a
giant holographic koi swims through the skyline. Someone sits on a crate watching. The signal
breaks up - the tree glitches, splits into colours, drops to a wireframe - then surges back
brighter than before, throwing petals into a storm.

    python holo_sakura.py            interactive window (Space pause, R restart, S save frame, Esc quit)
    python holo_sakura.py --png      save five stages
    python holo_sakura.py --gif      save the whole minute as a GIF
"""
import math

import numpy as np

from neon import NEON, TAU, Cam, Lights, finish, glow, paint, run, shades, sign_strokes, ss, stamp, stroke_points, \
    tone_lut

T_END = 60.0
STAGES = [6.0, 16.0, 30.0, 45.8, 55.0]
BASE = np.array([-0.6, 0.0, 5.6])            # where the projector sits (X, Y, z)
PINK = np.array([255, 110, 200], np.float32)
CYAN = np.array([60, 230, 255], np.float32)
GLITCHES = [(25.8, 0.5), (36.9, 0.4), (44.0, 4.2)]     # (start, length): the last is the big one


def hash01(*a):
    h = 0
    for i, v in enumerate(a):
        h = h * 1000003 + (np.asarray(v, np.int64) + 7919 * (i + 1)) * 2654435761
    return ((h ^ (h >> 13)) % 100003) / 100003.0


class HoloSakura:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.S = S = min(W, H)
        self.rng = np.random.default_rng(31)
        self.cam = Cam(W, H, 0.5 * W, 0.58 * H, 0.70 * S, 1.6)
        self.lut = tone_lut(1.3, 5.0)
        self.L = Lights(W, H, 4)
        self.build_city()
        self.build_tree()
        self.build_motion()

    # ---- the city beyond the parapet ---------------------------------------------------------------
    def build_city(self):
        W, H, c = self.W, self.H, self.cam
        rng = np.random.default_rng(4)
        img = np.zeros((H, W, 3), np.float32)
        depth = np.full((H, W), 5000.0, np.float32)
        yy = np.arange(H, dtype=np.float32)[:, None] / H
        sky = np.stack([np.interp(yy, [0, 0.3, 0.5], cc) for cc in ((10, 30, 90), (8, 16, 46), (26, 36, 84))], -1)[..., 0, :]
        img[:] = sky[:, None, :]
        xs = np.arange(W)[None, :] / W
        cloud = (0.5 + 0.5 * np.sin(xs * 7 + yy * 11) * np.sin(yy * 19 - xs * 5)) ** 4 * (0.3 + yy)
        img += np.array([90, 26, 90], np.float32) * cloud[..., None] * (yy < 0.5)[..., None]
        self.emit = np.zeros((H, W, 3), np.float32)
        # towers, far to near; each fades into the purple haze with distance
        towers = []
        spec = ((900, 1300, 34, 60, 380, 0.0), (520, 800, 22, 50, 300, 0.0), (300, 450, 12, 40, 200, 0.2),
                (160, 240, 6, 60, 260, 0.45))
        for layer, (z0, z1, n, h0, h1, side) in enumerate(spec):
            for _ in range(n):
                z = rng.uniform(z0, z1)
                u = rng.uniform(side, 0.8) * rng.choice([-1, 1]) if side > 0 else rng.uniform(-0.8, 0.8)
                X = u * z
                w = rng.uniform(16, 40) * (1 + 0.25 * layer)
                h = rng.uniform(h0, h1) * (1.0 + 0.6 * rng.random() ** 3)          # now and then a giant
                towers.append((z, X, w, h, layer))
        towers.sort(key=lambda T: -T[0])
        self.antennas = []
        self.tower_signs = []
        for z, X, w, h, layer in towers:
            x0, ytop = c.proj(X - w / 2, h, z)
            x1, ybot = c.proj(X + w / 2, -60, z)
            xa, xb, ya, yb = int(max(x0, 0)), int(min(x1, W)), int(max(ytop, 0)), int(min(ybot, H))
            if xa >= xb or ya >= yb:
                continue
            fog = min(1.0, (z / 1300) ** 0.7)
            haze = np.array([60, 40, 100], np.float32)
            body = np.array([16, 16, 28], np.float32) * (1 - fog) + haze * fog * 0.55
            img[ya:yb, xa:xb] = body
            depth[ya:yb, xa:xb] = z
            # window grid: little lit points
            gx = np.arange(xa, xb, max(1.5, c.f * 2.2 / z))
            gy = np.arange(ya, yb, max(1.5, c.f * 3.0 / z))
            GX, GY = np.meshgrid(gx, gy)
            hs = hash01((GX * 7).astype(int), (GY * 3).astype(int), int(z))
            lit = hs < 0.30
            if lit.any():
                pal = np.array([(255, 190, 120), (120, 220, 255), (255, 120, 210), (220, 225, 255)], np.float32)
                colw = pal[(hs[lit] * 97).astype(int) % 4] * (0.25 + 0.5 * (1 - fog))
                stamp(self.emit, GX[lit], GY[lit], colw * 0.40, 1.0)
            if rng.random() < 0.35:                                          # a neon strip down part of a face
                sx = rng.uniform(xa, xb)
                q = np.arange(ya + 2, ya + 2 + (yb - ya) * rng.uniform(0.15, 0.4), 1.5)
                col = np.array([NEON["pink"], NEON["cyan"], NEON["violet"]][int(rng.integers(0, 3))], np.float32)
                stamp(self.emit, np.full(len(q), sx), q, col * 0.55 * (1 - 0.7 * fog), 1.0)
            self.antennas.append(((x0 + x1) / 2, ytop, z, rng.uniform(0, 1)))
            if layer >= 2 and rng.random() < 0.7:                            # a sign high on the nearer towers
                n = int(rng.integers(3, 6))
                u, v = stroke_points(sign_strokes(rng, n, True, True), 0.05)
                cell = w * 0.12
                sx0 = X + rng.uniform(-0.3, 0.3) * w
                sy0 = h - rng.uniform(0.15, 0.4) * h
                self.tower_signs.append((sx0 + u * cell, sy0 - v * cell, np.full_like(u, z - 1),
                                         np.array([NEON["pink"], NEON["cyan"], NEON["magenta"], NEON["blue"]][int(rng.integers(0, 4))], np.float32),
                                         rng.uniform(0, 100)))
        # the rooftop: wet concrete with puddles, a low parapet, a water tank, vents
        ys = np.arange(H, dtype=np.float32)[:, None]
        Xg, Zg = None, None
        roof = ys > c.vy + 2
        zr = c.f * c.hc / np.maximum(ys - c.vy, 0.5)
        xr = (np.arange(W)[None, :] - c.vx) * zr / c.f
        par_y = c.proj(0, 1.05, 12.0)[1]
        floor = roof & (zr < 12.0) & np.ones((1, W), bool)
        noise = rng.random((H, W)).astype(np.float32)
        pud = np.sin(xr * 1.1 + np.sin(zr * 0.8) * 1.5) * np.sin(zr * 0.9 + xr * 0.3) + 0.3 * np.sin(zr * 2.1)
        self.puddle = np.where(floor, np.clip((pud + 0.15) * 2.0, 0, 1), 0).astype(np.float32)
        conc = np.array([30, 29, 38], np.float32)
        img = np.where(floor[..., None], conc * (0.7 + 0.35 * noise[..., None]) * (1 - 0.65 * self.puddle[..., None]), img)
        depth = np.where(floor, zr, depth).astype(np.float32)
        tiles = floor & (((xr % 1.5) < 0.04) | ((zr % 1.5) < 0.04))
        img[tiles] *= 0.6
        # the parapet
        pa = (ys >= par_y) & (ys < c.proj(0, 0, 12.0)[1] + 1) & np.ones((1, W), bool)
        img[pa] = np.array([40, 38, 50], np.float32) * (0.8 + 0.3 * noise[pa][:, None])
        depth[pa] = 12.0
        cap = (np.abs(ys - par_y) < 2) & np.ones((1, W), bool)
        img[cap] = np.array([70, 70, 84], np.float32)
        self.par_y = par_y
        # railing posts along the parapet
        for X in np.arange(-12, 12, 1.6):
            x, y0 = c.proj(X, 1.05, 12.0)
            _, y1 = c.proj(X, 1.9, 12.0)
            q = np.linspace(y1, y0, 30)
            stamp(img, np.full(30, x), q, np.array([30, 30, 40], np.float32), 1.2)
            depth[int(max(y1, 0)):int(y0), int(np.clip(x - 1, 0, W - 1)):int(np.clip(x + 2, 0, W))] = 12.0
        q = np.linspace(-12, 12, 900)
        x, y = c.proj(q, np.full(900, 1.9), np.full(900, 12.0))
        stamp(img, x, y, np.array([44, 44, 56], np.float32), 1.3)
        # a water tank on legs (right), vents (left)
        tx, tz = 4.2, 10.0
        for X0, X1, Y0, Y1, col in ((tx - 1.3, tx + 1.3, 1.6, 4.4, (34, 34, 44)), (tx - 1.2, tx - 1.1, 0, 1.6, (26, 26, 34)),
                                    (tx + 1.1, tx + 1.2, 0, 1.6, (26, 26, 34)), (-5.2, -3.6, 0, 1.1, (40, 40, 52)),
                                    (-3.4, -2.9, 0, 1.5, (36, 36, 46))):
            xa, ya = c.proj(X0, Y1, tz if X0 > 0 else 9.0)
            xb, yb = c.proj(X1, Y0, tz if X0 > 0 else 9.0)
            sl = (slice(int(max(ya, 0)), int(min(yb, H))), slice(int(max(xa, 0)), int(min(xb, W))))
            img[sl] = np.array(col, np.float32)
            depth[sl] = tz if X0 > 0 else 9.0
        xa, _ = c.proj(tx - 1.3, 0, tz)                                    # round it: shade across the tank, hoops, a cap
        xb, _ = c.proj(tx + 1.3, 0, tz)
        tank = (depth == tz) & np.ones((1, W), bool)
        u = np.clip((np.arange(W)[None, :] - xa) / max(xb - xa, 1), 0, 1) * np.ones((H, 1))
        shade = 0.55 + 0.8 * np.sin(np.pi * u) ** 1.5 * (0.6 + 0.4 * u)
        img[tank] *= shade[tank][:, None]
        hoops = tank & (np.sin(ys / H * 260) > 0.9)
        img[hoops] *= 1.4
        _, ytop = c.proj(0, 4.4, tz)
        for k_ in range(30):
            q = k_ / 29
            yq = ytop - q * c.f * 0.6 / tz
            half = (1 - q) * (xb - xa) / 2
            xs = np.arange(int((xa + xb) / 2 - half), int((xa + xb) / 2 + half))
            if len(xs) and 0 <= int(yq) < H:
                img[int(yq), np.clip(xs, 0, W - 1)] = np.array([38, 38, 50], np.float32) * (0.8 + 0.4 * q)
                depth[int(yq), np.clip(xs, 0, W - 1)] = tz
        self.img = img
        self.depth = depth
        # stipple
        n = int(W * H * 0.4)
        x, y = rng.uniform(0, W, n), rng.uniform(0, H, n)
        stamp(self.img, x, y, self.img[y.astype(int), x.astype(int)] * (0.82 + 0.36 * rng.random((n, 1))),
              rng.choice([1.0, 1.5, 2.0], n, p=[0.35, 0.45, 0.2]))
        self.img += self.emit
        # wet roof: puddles hold an upside-down copy of the skyline (the static part)
        ys = np.arange(H)
        src = np.clip((2 * c.vy - ys).astype(int), 0, H - 1)
        mirror = self.img[src]
        k = (self.puddle * 0.75 + 0.22 * (self.depth < 12.5))[..., None] * (ys[:, None, None] > self.par_y)
        k = k * (src < self.par_y)[:, None, None]
        self.img += mirror * k

    # ---- the tree of light --------------------------------------------------------------------------
    def build_tree(self):
        rng = self.rng
        segs = []

        def grow(p, d, L, depth, order):
            e = p + d * L
            segs.append((p, e, depth, order))
            if depth == 0:
                return
            k = 2 if depth > 2 else int(rng.integers(2, 4))
            for i in range(k):
                ax = np.cross(d, rng.normal(0, 1, 3))
                ax /= np.linalg.norm(ax) + 1e-9
                ang = rng.uniform(0.5, 1.05)
                nd = d * math.cos(ang) + np.cross(ax, d) * math.sin(ang)
                nd = nd + np.array([0, 0.12 if depth > 3 else -0.10, 0])        # reaching up, then spreading out
                nd /= np.linalg.norm(nd)
                grow(e, nd, L * rng.uniform(0.70, 0.86), depth - 1, order + L)

        grow(np.zeros(3), np.array([0.05, 1.0, 0.0]) / np.linalg.norm([0.05, 1.0, 0.0]), 1.15, 6, 0.0)
        pts, orders, widths = [], [], []
        for p, e, depth, order in segs:
            L = np.linalg.norm(e - p)
            n = max(3, int(L / 0.022))
            q = np.linspace(0, 1, n)[:, None]
            P = p[None] + (e - p)[None] * q
            pts.append(P)
            orders.append(order + L * q[:, 0])
            widths.append(np.full(n, depth))
        P = np.vstack(pts)
        self.br = dict(P=P, order=np.concatenate(orders), depth=np.concatenate(widths))
        self.br["order"] /= self.br["order"].max()
        # trunk and big limbs get a few parallel lines so they read as thick
        thick = self.br["depth"] >= 4
        off = []
        for dx in (-0.035, 0.035):
            off.append((P[thick] + np.array([dx, 0, 0]), self.br["order"][thick]))
        self.br_extra = (np.vstack([o[0] for o in off]), np.concatenate([o[1] for o in off]))
        # blossoms: along the outer twigs
        tips = [(e, order) for p, e, depth, order in segs if depth <= 1]
        cen, bo = [], []
        for e, order in tips:
            for _ in range(int(rng.integers(2, 4))):
                cen.append(e + rng.normal(0, 0.14, 3))
                bo.append(order)
        cen = np.array(cen)
        self.bl = dict(c=cen, order=np.array(bo) / max(bo), size=rng.uniform(0.05, 0.10, len(cen)),
                       hue=rng.random(len(cen)), ph=rng.uniform(0, TAU, len(cen)))
        a = np.repeat(np.arange(5) * TAU / 5, 4) + np.tile([0.0, 0.25, -0.25, 0.0], 5)
        r = np.tile([0.55, 0.85, 0.85, 1.0], 5)
        self.petal_shape = (np.cos(a) * r, np.sin(a) * r)
        # falling petals: each leaves a blossom at its own moment
        n = 700
        self.fall = dict(src=rng.integers(0, len(cen), n), t0=rng.uniform(17, 62, n), vx=rng.normal(0.9, 0.4, n),
                         vz=rng.normal(-0.2, 0.3, n), ph=rng.uniform(0, TAU, n), spin=rng.uniform(2, 6, n))
        storm = rng.random(n) < 0.45
        self.fall["t0"][storm] = rng.uniform(47.5, 60, int(storm.sum()))

    def glitch(self, t):
        """How broken the signal is right now (0 clean, 1 badly)."""
        g = 0.0
        for t0, ln in GLITCHES:
            if t0 <= t <= t0 + ln:
                g = max(g, max(0.0, math.sin(math.pi * (t - t0) / ln)) ** 0.5 * (1.0 if ln > 1 else 0.7))
        return g

    def wind(self, t):
        return 0.5 * math.sin(0.5 * t) + 0.3 * math.sin(1.3 * t + 1) + 0.8 * float(ss(t, 47, 50))

    def tree_points(self, t):
        """Branch points and blossom points in the world, swaying, with how much of each is drawn yet."""
        P = self.br["P"]
        h = np.clip(P[:, 1] / 4.0, 0, 1)
        sway = 0.10 * self.wind(t) * h ** 2
        Pw = P + np.stack([sway, np.zeros_like(sway), 0.04 * math.sin(0.9 * t) * h ** 2], 1) + BASE
        return Pw

    def draw_tree(self, t, refl):
        c, L = self.cam, self.L
        g = self.glitch(t)
        boot = float(ss(t, 1.5, 3.0))
        grow = ss(t, 3.0, 10.0)
        surge = float(ss(t, 48.0, 50.0))
        power = boot * (1 - 0.5 * g) * (1 + 0.35 * surge)
        if power <= 0:
            return
        P = self.tree_points(t)
        o = self.br["order"]
        show = o <= grow * 1.02
        # scanlines climbing the hologram, and a bright edge where it is still drawing itself
        x, y = c.proj(P[:, 0], P[:, 1], P[:, 2])
        scan = 0.55 + 0.45 * (np.sin(y * 0.55 - t * 9) > -0.3)
        front = np.exp(-((o - grow) / 0.03) ** 2) * (grow < 1)
        wire = float(ss(g, 0.6, 0.9))                                        # the big glitch drops it to a wireframe
        k = (scan * power * (0.55 + 0.45 * self.br["depth"] / 6) + 2.5 * front) * show
        jx = self.slice_shift(y, t, g)
        col = CYAN * (1 - wire) + np.array([255, 255, 255], np.float32) * wire * 0.6
        L.tube(x + jx, y, col, k * 0.55, core=0.7, halo=0.9, bloom=0.7)
        ex, eo = self.br_extra
        Pe = ex + BASE + np.stack([0.10 * self.wind(t) * np.clip(ex[:, 1] / 4, 0, 1) ** 2, np.zeros(len(ex)), np.zeros(len(ex))], 1)
        xe, ye = c.proj(Pe[:, 0], Pe[:, 1], Pe[:, 2])
        L.add(xe, ye, CYAN * 0.35 * power * (eo <= grow)[:, None], 1.0)
        if g > 0.15:                                                         # colours splitting apart
            L.add(x[show] + 4 * g, y[show], PINK * 0.25 * g, 1.0)
            L.add(x[show] - 4 * g, y[show], np.array([40, 255, 120]) * 0.2 * g, 1.0)
        rx, ry = c.proj(P[::4, 0], -P[::4, 1], P[::4, 2])
        refl.append((rx, ry, np.broadcast_to(CYAN * 0.4 * power, (len(rx), 3)) * show[::4, None]))
        # blossoms
        B = self.bl
        bloom_t = 9.5 + 8.5 * B["order"] + 1.5 * B["hue"]
        op = np.clip((t - bloom_t) / 0.6, 0, 1) * (1 - 0.8 * wire)
        on = op > 0.01
        if not on.any():
            return
        cen = B["c"][on]
        h = np.clip(cen[:, 1] / 4.0, 0, 1)
        cen = cen + BASE + np.stack([0.10 * self.wind(t) * h ** 2, np.zeros_like(h), 0.04 * math.sin(0.9 * t) * h ** 2], 1)
        cx, cy = c.proj(cen[:, 0], cen[:, 1], cen[:, 2])
        sc = c.f * B["size"][on] / cen[:, 2] * op[on]
        u, v = self.petal_shape
        px = cx[:, None] + u[None] * sc[:, None]
        py = cy[:, None] + v[None] * sc[:, None] * 0.9
        pop = 1 + 3 * np.exp(-((t - bloom_t[on]) / 0.25) ** 2)            # a flash as each one opens
        hue = B["hue"][on]
        col = PINK[None] * (0.75 + 0.25 * hue[:, None]) + np.array([255, 200, 240], np.float32)[None] * 0.15
        scan = 0.6 + 0.4 * (np.sin(cy * 0.55 - t * 9) > -0.3)
        k = (op[on] * pop * scan * power)[:, None]
        jx = self.slice_shift(py.ravel(), t, g)
        L.tube(px.ravel() + jx, py.ravel(), np.repeat(col * k, len(u), 0) * 0.30, 1.0, core=0.35, halo=0.8, bloom=0.5)
        L.add(cx, cy, np.array([255, 240, 250]) * (0.35 * k), 1.0)
        rx, ry = c.proj(cen[:, 0], -cen[:, 1], cen[:, 2])
        refl.append((rx, ry, col * k * 0.35))

    def slice_shift(self, y, t, g):
        """Glitch: horizontal slices of the hologram jump sideways."""
        if g <= 0.05:
            return 0.0
        band = np.floor(y / (6 + 20 * self.rng_slice(t)))
        r = hash01(band.astype(int), int(t * 12))
        return np.where(r < 0.35 * g, (r * 2 - 0.35 * g) * 120 * g * self.S / 1000, 0.0)

    def rng_slice(self, t):
        return (math.sin(t * 7.3) * 0.5 + 0.5)

    def draw_petals(self, t, refl):
        c, L = self.cam, self.L
        F, B = self.fall, self.bl
        age = t - F["t0"]
        on = age > 0
        if not on.any():
            return
        age = age[on]
        src = B["c"][F["src"][on]] + BASE
        g_y = 0.55 * age                                                     # they drift down slowly, like light
        X = src[:, 0] + F["vx"][on] * age * (0.6 + 0.4 * self.wind(t)) + 0.15 * np.sin(1.7 * age + F["ph"][on])
        Y = src[:, 1] - g_y
        Z = src[:, 2] + F["vz"][on] * age
        landed = Y <= 0.02
        Y = np.maximum(Y, 0.02)
        fade = np.where(landed, np.clip(1 - (0.02 - (src[:, 1] - g_y)) / 0.3, 0, 1), 1.0)
        x, y = c.proj(X, Y, Z)
        s = np.maximum(c.f * 0.035 / Z, 1.0)
        a = F["spin"][on] * age + F["ph"][on]
        flip = np.abs(np.cos(a))
        k = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])   # a pixel petal: four dots
        px = x[:, None] + k[None, :, 0] * s[:, None] * flip[:, None]
        py = y[:, None] + k[None, :, 1] * s[:, None]
        col = PINK * 0.55 * fade[:, None]
        L.tube(px.ravel(), py.ravel(), np.repeat(col, 4, 0), 1.0, core=0.7, halo=0.8, bloom=0.4)
        hit = landed & (fade > 0.05)                                         # fizzling into sparks on the wet roof
        if hit.any():
            n = int(hit.sum())
            r = np.random.default_rng(int(t * 30))
            sx = (x[hit][:, None] + r.normal(0, 4, (n, 5))).ravel()
            sy = (y[hit][:, None] - np.abs(r.normal(0, 3, (n, 5)))).ravel()
            L.add(sx, sy, np.array([255, 180, 240])[None] * 0.5 * np.repeat(fade[hit], 5)[:, None], 1.0)
        rx, ry = c.proj(X[~landed], -Y[~landed], Z[~landed])
        refl.append((rx, ry, col[~landed] * 0.8))

    # ---- the rest that moves -------------------------------------------------------------------------
    def build_motion(self):
        rng = self.rng
        n = 2400
        self.rain = dict(X=rng.uniform(-9, 9, n), Y=rng.uniform(0, 12, n), z=1.5 * (40 / 1.5) ** rng.random(n), v=rng.uniform(8, 11, n))
        n = 36
        self.cars = dict(lane=rng.integers(0, 4, n), ph=rng.random(n), dirn=np.where(rng.random(n) < 0.5, -1, 1))
        u = np.linspace(0, 1, 90)
        self.koi_u = u

    def draw_city(self, img, t):
        c, L = self.cam, self.L
        for x, y, z, ph in self.antennas:                                    # red lights on the rooftops
            if (t * 0.8 + ph) % 1.0 < 0.2:
                L.add(x, y - 2, np.array([255, 30, 40]) * (0.6 + 0.4 * min(1, 300 / z)), 1.2)
        for X, Y, Z, col, seed in self.tower_signs:
            x, y = c.proj(X, Y, Z)
            hum = 0.85 + 0.15 * math.sin(40 * t + seed)
            L.tube(x, y, col, 0.65 * hum, core=0.6, halo=0.8, bloom=0.8)
        C = self.cars                                                        # traffic lanes between the towers
        lanes = [(70.0, 160.0, 1.0), (95.0, 210.0, 0.8), (55.0, 120.0, 1.3), (120.0, 260.0, 0.7)]
        Ys = np.array([l[0] for l in lanes])[C["lane"]]
        zs = np.array([l[1] for l in lanes])[C["lane"]]
        sp = np.array([l[2] for l in lanes])[C["lane"]]
        X = ((C["ph"] + 0.010 * t * sp * C["dirn"]) % 1.0 - 0.5) * zs * 1.8
        x, y = c.proj(X, Ys, zs)
        xi, yi = np.clip(x.astype(int), 0, self.W - 1), np.clip(y.astype(int), 0, self.H - 1)
        vis = (self.depth[yi, xi] > zs) & (x > 0) & (x < self.W) & (y > 0) & (y < self.H)
        col = np.where(C["dirn"][:, None] > 0, np.array([255, 235, 210], np.float32), np.array([255, 50, 90], np.float32))
        for k in range(7):
            L.add(x[vis] - C["dirn"][vis] * k * 2.2, y[vis], col[vis] * (0.8 - 0.1 * k), 1.0)
        self.draw_koi(t)

    def koi_shape(self, t):
        """The koi's points in the world, with their colour (0 cyan .. 1 pink) and how bright."""
        a = (t - 13.0) / 32.0
        if not 0 < a < 1:
            return None
        if not hasattr(self, "koi_pts"):
            r = np.random.default_rng(3)
            n = 1300
            u = r.random(n) ** 0.9                                         # 0 head .. 1 tail
            v = r.uniform(-1, 1, n)
            edge = (np.abs(v) > 0.86) | (u < 0.04)
            patch = (np.sin(u * 13 + 1.0) * np.cos(v * 2.5 + u * 4) > 0.15) & (u < 0.8)
            self.koi_pts = (u, v, edge, patch)
            m = 260
            self.koi_tail = (r.random(m), r.uniform(-1, 1, m))
        u, v, edge, patch = self.koi_pts
        zc, L = 230.0, 95.0
        Xh = (-0.95 + 1.9 * a) * zc * 0.62
        Yh = 112 + 22 * math.sin(a * 5)
        wave = lambda q: np.sin(q * 5 - t * 3.0) * 7 * q ** 1.5
        body = 24 * np.sin(np.pi * np.clip(0.04 + 0.86 * u, 0, 1)) ** 0.75 * (1 - 0.55 * u)
        X = Xh - u * L
        Y = Yh + wave(u) + v * body * 0.5
        tu, tv = self.koi_tail                                              # the tail: a fan of fine rays
        Xt = Xh - L * (0.98 + 0.30 * tu)
        Yt = Yh + wave(np.ones_like(tu)) + tv * (8 + 24 * tu) + 6 * math.sin(t * 3.0 - 5)
        hue = np.concatenate([patch.astype(float), np.full(len(tu), 0.5)])
        k = np.concatenate([np.where(edge, 1.0, 0.45), 0.5 * (1 - tu)])
        return np.concatenate([X, Xt]), np.concatenate([Y, Yt]), zc, hue, k, math.sin(math.pi * a) ** 0.4

    def draw_koi(self, t):
        K = self.koi_shape(t)
        if K is None:
            return
        X, Y, zc, hue, k, fade = K
        c, L = self.cam, self.L
        x, y = c.proj(X, Y, np.full(len(X), zc))
        xi, yi = np.clip(x.astype(int), 0, self.W - 1), np.clip(y.astype(int), 0, self.H - 1)
        vis = (self.depth[yi, xi] > zc) & (x > 0) & (x < self.W) & (y > 0) & (y < self.H)
        scan = 0.55 + 0.45 * (np.sin(y * 0.45 + t * 7) > -0.2)
        col = CYAN[None] * (1 - hue[:, None]) + PINK[None] * hue[:, None]
        p = k * scan * fade * 0.55
        L.tube(x[vis], y[vis], col[vis], p[vis], core=0.45, halo=0.8, bloom=0.8)
        self.koi_screen = (x[vis][::3], y[vis][::3], (col * p[:, None])[vis][::3])

    def draw_rain(self, img, t):
        R, c, L = self.rain, self.cam, self.L
        Y = (R["Y"] - R["v"] * t) % 12.0
        X = R["X"] + 0.1 * Y
        x0, y0 = c.proj(X, Y, R["z"])
        x1, y1 = c.proj(X + 0.05, Y + 0.45, R["z"])
        xi, yi = np.clip(x0.astype(int), 0, self.W - 1), np.clip(y0.astype(int), 0, self.H - 1)
        vis = (x0 > 0) & (x0 < self.W) & (y0 > 0) & (y0 < self.H) & (self.depth[yi, xi] >= R["z"] - 0.3)
        col = img[yi[vis], xi[vis]] * 0.30 + 14
        for k in (0.0, 0.33, 0.66, 1.0):
            L.add(x0[vis] + (x1 - x0)[vis] * k, y0[vis] + (y1 - y0)[vis] * k, col * (0.9 - 0.25 * k), 1.0)

    def draw_reflections(self, t, refl):
        if not refl:
            return
        x = np.concatenate([r[0] for r in refl])
        y = np.concatenate([r[1] for r in refl])
        col = np.concatenate([np.broadcast_to(r[2], (len(r[0]), 3)) for r in refl])
        xi, yi = np.clip(x.astype(int), 0, self.W - 1), np.clip(y.astype(int), 0, self.H - 1)
        ok = (y > self.par_y) & (y < self.H) & (x > 0) & (x < self.W) & (self.depth[yi, xi] < 12.5)
        x, y, col = x[ok], y[ok], col[ok]
        pud = self.puddle[yi[ok], xi[ok]]
        k = (0.08 + 0.6 * pud)[:, None]
        for j in range(4):
            yy = y + j * 3.5 * self.S / 1000
            wob = 3.0 * self.S / 1000 * np.sin(yy * 0.3 + t * 6 + x * 0.04) * (1 - 0.6 * pud)
            self.L.soft(x + wob, yy, col * k * (1.4 - 0.3 * j))

    def draw_projector(self, t):
        c, L = self.cam, self.L
        a = np.linspace(0, TAU, 80)
        x, y = c.proj(BASE[0] + 0.38 * np.cos(a), np.full(80, 0.05), BASE[2] + 0.38 * np.sin(a) * 0.6)
        on = float(ss(t, 0.3, 1.5))
        blink = 0.5 + 0.5 * (math.sin(t * 25) > 0) if t < 2.0 else 1.0
        L.tube(x, y, CYAN, 0.8 * on * blink * (1 - 0.4 * self.glitch(t)), core=0.8, halo=1.0, bloom=1.0)
        beam = float(ss(t, 1.0, 2.0) * (1 - ss(t, 3.5, 6.0)))                  # the start-up beam
        if beam > 0:
            r = np.random.default_rng(int(t * 30)).random((2, 500))
            Y = r[0] ** 1.5 * 4.0
            ang = r[1] * TAU
            rad = 0.35 + 0.5 * Y / 4.0
            x, y = c.proj(BASE[0] + rad * np.cos(ang), Y, BASE[2] + rad * np.sin(ang) * 0.6)
            L.add(x, y, CYAN * 0.25 * beam * (1 - Y / 4.0)[:, None], 1.0)

    def draw_watcher(self, dots, t):
        """Someone sitting on a crate at the right, lit pink by the tree, a visor glowing."""
        c, L = self.cam, self.L
        X0, z0 = 2.4, 5.2
        rng = np.random.default_rng(6)
        parts = []

        def blob(cx, cy, rx, ry, n, col):
            a, r = rng.uniform(0, TAU, n), np.sqrt(rng.random(n))
            parts.append((cx + rx * r * np.cos(a), cy + ry * r * np.sin(a), np.broadcast_to(np.asarray(col, np.float32), (n, 3)),
                          np.cos(a) * r ** 6))

        breath = 0.01 * math.sin(1.2 * t)
        blob(0.0, 0.25, 0.45, 0.25, 260, (30, 30, 36))                     # the crate
        blob(0.0, 0.95 + breath, 0.22, 0.32, 260, (22, 20, 26))           # body, hunched forward
        blob(-0.05, 1.42 + breath, 0.13, 0.14, 160, (24, 22, 28))         # head, hooded
        blob(0.12, 0.55, 0.28, 0.08, 110, (20, 18, 24))                   # legs, knees up
        blob(0.25, 0.30, 0.06, 0.22, 60, (20, 18, 24))
        x = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        col = np.concatenate([p[2] for p in parts])
        rim = np.concatenate([p[3] for p in parts])
        px, py = c.proj(X0 + x, y, np.full(len(x), z0))
        tree_lit = float(ss(t, 9, 14)) * (1 - 0.5 * self.glitch(t))
        col = col + PINK * (0.45 * np.clip(-rim, 0, 1) * tree_lit)[:, None] + CYAN * (0.12 * np.clip(rim, 0, 1))[:, None]
        dots.append((px, py, col, np.clip(c.f * 0.02 / z0, 1, 2.6), np.full(len(px), -z0)))
        vx, vy = c.proj(X0 - 0.14 + np.linspace(-0.05, 0.05, 6), np.full(6, 1.45 + breath), np.full(6, z0 - 0.1))
        L.tube(vx, vy, NEON["cyan"], 0.7, core=0.8, halo=0.8, bloom=0.6)

    # ---- the frame ---------------------------------------------------------------------------------
    def render(self, t):
        img = self.img.copy()
        refl = []
        dots = []
        self.draw_city(img, t)
        self.draw_projector(t)
        self.draw_tree(t, refl)
        self.draw_petals(t, refl)
        self.draw_watcher(dots, t)
        if getattr(self, "koi_screen", None) is not None and self.koi_shape(t) is not None:
            kx, ky, kc = self.koi_screen
            refl.append((kx, 2 * self.cam.vy - ky, kc * 0.8))
        self.koi_screen = None
        self.draw_reflections(t, refl)
        paint(img, dots)
        self.draw_rain(img, t)
        g = self.glitch(t)
        if g > 0.5:                                                          # the big glitch dims the whole roof a moment
            img *= 1 - 0.2 * (g - 0.5)
        self.L.draw(img, 2)
        return finish(img, self.lut)


if __name__ == "__main__":
    run(HoloSakura, "Hologram sakura", "holo_sakura", T_END, STAGES)
