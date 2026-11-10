"""
golden_hour.py - the last hour of sun over a wheat field, in dots.

The sun hangs low and straight ahead, so everything is lit from behind: every wheat head
wears a rim of gold, the whiskers of the near ears glow like filaments, and dust hangs
glittering in the air. Gusts of wind run across the field toward you as rolling waves of
sheen. Shadows are impossibly long - the lone oak on the skyline, the telegraph poles and a
cyclist riding away down the farm track all lay dark stripes across the wheat that stretch
longer and longer as the sun sinks. Swallows skim the tops of the ears. The light deepens
from gold to orange to rose; the sun touches the hills and slides away, the shadows let go,
and the field sits in a pink afterglow with the first star out.

    python golden_hour.py            interactive window (Space pause, R restart, S save frame, Esc quit)
    python golden_hour.py --png      save five stages
    python golden_hour.py --gif      save the whole hour as a GIF
"""
import math

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))   # the shared engine: engine/pointillism.py

from pointillism import Backdrop, Haze, Sun, glow, keyed, mix, paint, run, shades, sky, ss, stamp, warm_glow

T_END = 60.0
STAGES = [5.0, 20.0, 35.0, 48.0, 57.0]
TAU = 2 * math.pi

GRAIN = [(196, 150, 78), (214, 170, 92), (228, 188, 108), (176, 132, 70)]
STALK = [(170, 138, 74), (186, 152, 84), (150, 122, 66)]
FAR = [(206, 160, 92), (220, 176, 104), (190, 146, 84)]
DARK = np.array([40, 30, 28], np.float32)
ROAD = [(176, 140, 104), (190, 154, 116), (160, 124, 92)]

LOOKS = [  # golden, deeper gold, orange, sun on the hills, afterglow
    dict(t=0.0, sky=[(0.0, (104, 140, 190)), (0.30, (176, 180, 186)), (0.48, (250, 214, 150)), (0.57, (255, 222, 160))],
         glow=(140, 100, 40), haze=(255, 214, 150), hills=0.55, field=1.0, key=(255, 196, 110), amb=(0.82, 0.74, 0.66)),
    dict(t=18.0, sky=[(0.0, (110, 128, 180)), (0.30, (196, 170, 160)), (0.48, (255, 196, 120)), (0.57, (255, 204, 130))],
         glow=(160, 100, 30), haze=(255, 196, 120), hills=0.50, field=0.95, key=(255, 176, 90), amb=(0.76, 0.66, 0.60)),
    dict(t=34.0, sky=[(0.0, (90, 100, 160)), (0.30, (200, 140, 140)), (0.48, (255, 160, 90)), (0.57, (255, 170, 96))],
         glow=(170, 80, 20), haze=(255, 168, 96), hills=0.42, field=0.85, key=(255, 150, 70), amb=(0.68, 0.56, 0.56)),
    dict(t=48.0, sky=[(0.0, (64, 70, 130)), (0.30, (176, 110, 130)), (0.48, (250, 128, 80)), (0.57, (255, 140, 80))],
         glow=(160, 60, 20), haze=(250, 136, 84), hills=0.32, field=0.68, key=(240, 120, 60), amb=(0.56, 0.46, 0.52)),
    dict(t=60.0, sky=[(0.0, (36, 40, 90)), (0.30, (110, 84, 124)), (0.48, (210, 120, 120)), (0.57, (226, 140, 120))],
         glow=(60, 26, 20), haze=(200, 130, 130), hills=0.24, field=0.42, key=(120, 70, 70), amb=(0.42, 0.38, 0.48)),
]


class GoldenHour:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.S = S = min(W, H)
        self.rng = np.random.default_rng(23)
        self.hz = 0.56 * H                        # the horizon
        self.vx = 0.58 * W                        # straight ahead
        self.f = 0.9 * S                          # lens
        self.hc = 1.45                            # eye height, metres
        self.sun = Sun(np.random.default_rng(5), 0.040 * S, 6.0)
        self.haze = Haze(W, H, max(2, round(4 * S / 1000)))
        self.amb_keys = [(L["t"], L["amb"]) for L in LOOKS]
        self.key_keys = [(L["t"], L["key"]) for L in LOOKS]
        self.build_world()
        self.backdrop = Backdrop([L["t"] for L in LOOKS], [self.background(L) for L in LOOKS])
        self.build_wheat()
        self.build_life()

    # ---- the sun and the ground ----------------------------------------------------------------------
    def sun_dir(self, t):
        """Elevation and bearing of the sun (radians): it sinks through the minute and sets at ~51 s."""
        elev = math.radians(13.5 - 15.5 * (t / T_END) ** 1.15)
        az = math.radians(6.0 + 2.5 * t / T_END)
        return elev, az

    def sun_xy(self, t):
        elev, az = self.sun_dir(t)
        return self.vx + self.f * math.tan(az), self.hz - self.f * math.tan(elev)

    def proj(self, X, Yh, z):
        z = np.maximum(z, 0.05)
        return self.vx + self.f * X / z, self.hz + self.f * (self.hc - Yh) / z

    def build_world(self):
        """The fixed things on the land: the oak, the poles, the wires, the track (as 3D points)."""
        rng = np.random.default_rng(3)
        # the oak on the skyline, far off to the left
        tx, tz = -9.0, 72.0
        self.oak_base = (tx, tz)
        n = 2600
        a, r = rng.uniform(0, TAU, n), np.sqrt(rng.random(n))
        cx = tx + 6.5 * r * np.cos(a) * (1 + 0.15 * np.sin(5 * a))
        cy = 8.5 + 4.2 * r * np.sin(a) * (1 + 0.12 * np.sin(7 * a + 1))
        trunk = np.stack([tx + rng.normal(0, 0.35, 300), rng.uniform(0, 5.5, 300)])
        self.oak = dict(X=np.concatenate([cx, trunk[0]]), Y=np.concatenate([cy, trunk[1]]), z=np.full(n + 300, tz) +
                        np.concatenate([rng.normal(0, 3, n), rng.normal(0, 0.3, 300)]),
                        rim=np.concatenate([np.clip(np.cos(a - 0.5), 0, 1) * r ** 8, np.zeros(300)]))
        # telegraph poles along the far side of the track
        self.poles = [(5.6, z) for z in (7.0, 17.0, 27.0, 37.0, 47.0, 57.0, 67.0, 77.0)]
        self.pole_h = 7.0
        self.track = (1.2, 4.2)                  # the track runs from X = 1.2 m to 4.2 m

    def background(self, L):
        W, H, S = self.W, self.H, self.S
        rng = np.random.default_rng(1234)
        img = sky(W, H, rng, L["sky"], self.hz + 4)
        sx, sy = self.sun_xy(L["t"])
        warm_glow(img, sx, sy, 0.45 * W, L["glow"], 0.55)
        hazec = np.asarray(L["haze"], np.float32)
        key = np.asarray(L["key"], np.float32)
        amb = np.asarray(L["amb"], np.float32)
        # low hills on the skyline, lost in the haze
        xs = np.arange(W, dtype=np.float32)
        yy = np.arange(H, dtype=np.float32)[:, None]
        for base, amp, f1, ph, wash in ((0.0, 0.020, 1.3, 0.4, 0.72), (0.004, 0.012, 2.7, 2.0, 0.55)):
            top = self.hz - H * (base + amp * (0.55 + 0.45 * np.sin(xs / W * TAU * f1 + ph)))
            fill = (yy >= top[None, :]) & (yy < self.hz + 4)
            col = mix(np.array([[90, 70, 70]], np.float32) * amb, hazec[None], wash)[0] * L["hills"] / 0.55 * 0.9
            img[fill] = col * 0.95
            n = int(W * 0.04 * H * 0.5)
            x = rng.uniform(0, W, n)
            y = np.interp(x, xs, top) + rng.random(n) * 0.03 * H
            stamp(img, x, y, col * (0.92 + 0.16 * rng.random((n, 1))), 1.5)
        # the field to the horizon: a texture of ears, paler with distance
        field = yy >= self.hz
        d = np.clip((yy - self.hz) / (H - self.hz), 0, 1)
        fcol = mix(np.asarray(FAR[1], np.float32)[None], hazec[None], 0.55)[0]
        img[:] = np.where(field[..., None], fcol * (1 - 0.35 * d[..., None]) * L["field"], img)
        n = int(W * (H - self.hz) * 0.9)
        x, y = rng.uniform(0, W, n), self.hz + (H - self.hz) * rng.random(n) ** 1.3
        d = (y - self.hz) / (H - self.hz)
        z = self.f * self.hc * 0.35 / np.maximum(y - self.hz, 0.5)       # roughly how far away
        near_sun = np.exp(-((x - sx) / (0.16 * W)) ** 2) * (1 - d) ** 0.5
        col = shades(rng, FAR, rng.random(n), 0.8) * (0.85 - 0.3 * d)[:, None] * amb * L["field"]
        col = mix(col, hazec[None] * L["field"], 0.45 * (1 - d) ** 2) + key * (0.25 * near_sun)[:, None]
        ln = (0.6 + 6 * d ** 1.6) * S / 1000
        k = rng.random(n)
        stamp(img, x, y - ln * k, col * (0.9 + 0.25 * k)[:, None], 1.0 + 1.2 * d)
        # the track: pale dust and two wheel ruts, running to the skyline
        zz = np.geomspace(0.6, 160, 1400)
        for X0, X1, c, w in ((self.track[0], self.track[1], ROAD[1], 1.0), (1.8, 2.15, ROAD[2], 0.8), (3.2, 3.55, ROAD[2], 0.8)):
            m = 22000 if w == 1.0 else 6000
            zq = rng.choice(zz, m)
            X = rng.uniform(X0, X1, m)
            x, y = self.proj(X, np.zeros(m), zq)
            on = (x > -3) & (x < W + 3) & (y < H + 3)
            col = shades(rng, ROAD, rng.random(m)) * (w * amb * L["field"] * 0.95)
            col = mix(col, hazec[None] * L["field"], 0.6 * np.clip(1 - zq / 40, 0, 1) * 0 + 0.5 * np.clip(zq / 120, 0, 1))
            col = col + key * (0.18 * np.exp(-((x - sx) / (0.1 * W)) ** 2))[:, None]
            stamp(img, x[on], y[on], col[on], np.clip(self.f * 0.03 / zq[on], 1.0, 3.0))
        # the oak, dark against the glare with a hot rim on its sunward side
        O = self.oak
        x, y = self.proj(O["X"], O["Y"], O["z"])
        col = DARK * (0.8 + 0.4 * amb.mean()) + key * (0.45 * O["rim"] ** 2)[:, None]
        stamp(img, x, y, mix(col, hazec[None], 0.25), 1.5)
        # the poles and the wires between them
        tops = []
        for X, z in self.poles:
            q = np.linspace(0, self.pole_h, 160)
            x, y = self.proj(np.full_like(q, X), q, np.full_like(q, z))
            r = np.clip(self.f * 0.13 / z, 1.0, 3.0)
            col = DARK * 1.1 + key * 0.12
            stamp(img, x, y, mix(np.broadcast_to(col, (160, 3)).copy(), hazec[None], 0.25 * min(z / 60, 1)), r)
            for arm in (-0.9, 0.0, 0.9):                               # cross-arm and insulators
                ax, ay = self.proj(np.array([X + arm]), np.array([self.pole_h - 0.5]), np.array([z]))
                stamp(img, ax, ay, col, r)
            xa, ya = self.proj(np.linspace(X - 1.0, X + 1.0, 30), np.full(30, self.pole_h - 0.45), np.full(30, z))
            stamp(img, xa, ya, col, r * 0.8)
            tops.append((X, z))
        for (X0, z0), (X1, z1) in zip(tops[:-1], tops[1:]):
            for off in (-0.9, 0.0, 0.9):
                q = np.linspace(0, 1, 500)
                z = z0 + (z1 - z0) * q
                Y = self.pole_h - 0.5 - 0.6 * 4 * q * (1 - q)                # the sag
                x, y = self.proj(X0 + off + (X1 - X0) * q, Y, z)
                glint = np.exp(-((x - sx) / (0.05 * W)) ** 2)
                col = DARK * 1.2 + key[None] * (0.25 + 0.9 * glint)[:, None]
                stamp(img, x, y, col, 1.0)
        return img

    def sky_mask(self):
        if not hasattr(self, "_sky"):
            img = self.backdrop.images[0]
            H = self.H
            yy = np.arange(H)[:, None]
            m = (yy < self.hz - 0.002 * H).astype(np.float32) * np.ones((1, self.W), np.float32)
            dark = img.mean(2) < 0.45 * img[:int(self.hz)].mean()
            m[dark] = 0
            self._sky = m
        return self._sky

    # ---- wheat ----------------------------------------------------------------------------------------
    def build_wheat(self):
        rng, f = self.rng, self.f
        # middle distance: each ear a short stroke
        n = 5400
        z = 3.0 * (14.0 / 3.0) ** rng.random(n)
        X = rng.uniform(-0.70, 0.46, n) * z
        off = (X < self.track[0] - 0.2) | (X > self.track[1] + 0.3)
        z, X = z[off], X[off]
        self.mid = dict(X=X, z=z, h=1.0 + rng.normal(0, 0.05, len(z)), ph=rng.uniform(0, TAU, len(z)),
                        col=shades(rng, GRAIN, rng.random(len(z)), 0.8), lean=rng.normal(0, 0.06, len(z)))
        # near ears in detail: grains in two rows, whiskers, stalks
        n = 210
        z = 0.75 * (3.2 / 0.75) ** rng.random(n) ** 0.9
        X = rng.uniform(-0.72, 0.30, n) * z
        keep = X < self.track[0] - 0.15
        self.near = dict(X=X[keep], z=z[keep], h=1.0 + rng.normal(0, 0.06, int(keep.sum())),
                         ph=rng.uniform(0, TAU, int(keep.sum())), lean=rng.normal(0, 0.08, int(keep.sum())),
                         tilt=rng.normal(0, 0.18, int(keep.sum())))
        k = 14                                                                # spikelets, alternating sides
        sk = np.where(np.arange(k) % 2, 1.0, -1.0)
        uk = np.arange(k) / (k - 1) * 0.92
        gu, gs = [], []
        for du, sv in ((0.0, 0.30), (0.018, 0.72), (0.036, 0.98), (0.010, 0.05)):   # grains fanning out of each
            gu.append(uk + du)
            gs.append(sk * sv)
        self.grain_u = np.concatenate(gu)
        self.grain_s = np.concatenate(gs)
        self.grain_col = shades(rng, GRAIN, rng.random(len(self.grain_u)), 0.8)
        self.awn_u = uk + 0.04                                                # one whisker per spikelet
        self.awn_a = sk * rng.uniform(0.10, 0.30, k)
        self.awn_l = rng.uniform(0.7, 1.05, k)
        # tall wild grass at the left edge, standing up into the sky
        g = 7
        self.tall = dict(x=rng.uniform(-0.02, 0.16, g), h=rng.uniform(0.40, 0.62, g), ph=rng.uniform(0, TAU, g),
                         lean=rng.uniform(0.02, 0.12, g))

    def gust(self, z, X, t):
        """Wind: gusts that run across the field toward us, plus a little restlessness."""
        a = np.sin(0.55 * z + 1.9 * t + 0.08 * X) * 0.5 + 0.5
        b = np.sin(0.21 * z + 0.9 * t - 0.15 * X + 1.0) * 0.5 + 0.5
        return np.clip(a ** 3 * 0.8 + b ** 4 * 0.5, 0, 1)

    def draw_mid(self, dots, t, sx, amb, key, sunk):
        M, f = self.mid, self.f
        g = self.gust(M["z"], M["X"], t)
        lean = M["lean"] + 0.05 * np.sin(1.3 * t + M["ph"]) + 0.30 * g
        q = np.array([0.0, 0.5, 1.0])[None, :]
        L = 0.085
        X = M["X"][:, None] + (lean[:, None] * L * q)
        Y = M["h"][:, None] + L * q * (1 - 0.3 * g[:, None])
        z = M["z"][:, None] - 0.25 * g[:, None] * L * q
        x, y = self.proj(X, Y, z)
        near_sun = np.exp(-((x[:, -1] - sx) / (0.13 * self.W)) ** 2)
        lit = (0.35 + 0.65 * g) * (0.4 + 0.6 * near_sun)
        col = M["col"] * amb + key * (0.35 * lit * sunk)[:, None]
        col = np.repeat(col[:, None, :], 3, 1) * np.array([0.82, 1.0, 1.1])[None, :, None]
        r = np.clip(f * 0.013 / M["z"], 1.0, 2.6)
        dots.append((x.ravel(), y.ravel(), col.reshape(-1, 3), np.repeat(r, 3), np.repeat(-M["z"], 3)))

    def draw_near(self, dots, glows, t, sx, sy, amb, key, sunk):
        N, f = self.near, self.f
        g = self.gust(N["z"], N["X"], t)
        sway = N["lean"] + 0.06 * np.sin(1.1 * t + N["ph"]) + 0.32 * g
        L = 0.095
        bx, by = self.proj(N["X"], N["h"], N["z"])
        scale = f / N["z"]
        ang = -math.pi / 2 + sway + N["tilt"]                             # screen angle of the ear (up = -pi/2)
        ux, uy = np.cos(ang), np.sin(ang)
        px, py = -uy, ux
        gu, gs = self.grain_u[None, :], self.grain_s[None, :]
        length = L * scale
        wid = 0.011 * scale[:, None] * (1 - 0.4 * gu)
        gx = bx[:, None] + ux[:, None] * gu * length[:, None] + px[:, None] * gs * wid
        gy = by[:, None] + uy[:, None] * gu * length[:, None] + py[:, None] * gs * wid
        # backlit: the side of each grain toward the sun burns gold, the rest is in its own shade
        toward = np.clip((sx - gx) / (np.abs(sx - gx) + 30) * 0.5 + 0.5, 0, 1)
        rim = (np.abs(gs) > 0.6) * (0.35 + 0.65 * toward) + 0.2 * gu
        near_sun = np.exp(-(((gx - sx) / (0.25 * self.W)) ** 2 + ((gy - sy) / (0.25 * self.H)) ** 2))
        shade = 0.55 + 0.25 * np.abs(gs) + 0.2 * g[:, None]                  # grains overlap: darker toward the middle
        col = self.grain_col[None] * amb * shade[..., None] + key * ((0.28 * rim + 0.5 * near_sun) * sunk)[..., None]
        r = np.clip(0.0058 * scale, 1.2, 3.0)
        z = np.broadcast_to(-N["z"][:, None], gx.shape)
        dots.append((gx.ravel(), gy.ravel(), col.reshape(-1, 3), np.repeat(r, gx.shape[1]), z.ravel()))
        # stalks, from the ear down out of the frame
        q = np.linspace(0, 1, 18)[None, :]
        _, gy0 = self.proj(N["X"], np.zeros_like(N["z"]) + 0.2, N["z"])
        cx = bx[:, None] - (sway[:, None] * scale[:, None] * 0.25) * q ** 2
        cy = by[:, None] + (gy0 - by)[:, None] * q
        scol = np.array(STALK[0], np.float32) * amb * 0.8 + key * 0.10 * sunk
        dots.append((cx.ravel(), cy.ravel(), np.broadcast_to(scol, (cx.size, 3)), np.repeat(np.clip(0.003 * scale, 1.0, 2.4), 18),
                     np.repeat(-N["z"] - 0.005, 18)))
        # whiskers: fine, and glowing where the sun is behind them
        au, aa, al = self.awn_u[None, :], self.awn_a[None, :], self.awn_l[None, :]
        k = np.linspace(0.15, 1.0, 4)[None, None, :]
        base_x = bx[:, None] + ux[:, None] * au * length[:, None] + px[:, None] * np.sign(aa) * wid[:, :1] * 0.8
        base_y = by[:, None] + uy[:, None] * au * length[:, None] + py[:, None] * np.sign(aa) * wid[:, :1] * 0.8
        aang = ang[:, None] + aa
        alen = length[:, None] * 0.85 * al
        wx = base_x[..., None] + np.cos(aang)[..., None] * alen[..., None] * k
        wy = base_y[..., None] + np.sin(aang)[..., None] * alen[..., None] * k
        ns = np.exp(-(((wx - sx) / (0.22 * self.W)) ** 2 + ((wy - sy) / (0.22 * self.H)) ** 2))
        b = (0.18 + 1.1 * ns) * sunk * (1 - 0.55 * k)
        glows.append((wx.ravel(), wy.ravel(), (key[None] * (0.95 * b.ravel())[:, None]), 1.0))

    def draw_tall(self, dots, glows, t, sx, sy, amb, key, sunk):
        T, W, H = self.tall, self.W, self.H
        for i in range(len(T["x"])):
            q = np.linspace(0, 1, 120)
            sw = T["lean"][i] + 0.03 * math.sin(1.4 * t + T["ph"][i]) + 0.05 * float(self.gust(np.array([1.0]), np.array([0.0]), t)[0])
            x = T["x"][i] * W + sw * W * q ** 2
            y = H * 1.02 - T["h"][i] * H * q
            dots.append((x, y, DARK * 0.9 + key * 0.06, 1.6 * self.S / 1000 + 0.4, np.full(120, 50.0)))
            k = 9                                                          # drooping spikelets near the top
            for j in range(k):
                s = 0.62 + 0.38 * j / k
                ix = int(s * 119)
                side = 1 if j % 2 else -1
                c = np.linspace(0, 1, 14)
                lx = x[ix] + side * c * 0.022 * W
                ly = y[ix] + c ** 2 * 0.03 * H
                dots.append((lx, ly, DARK * 0.9, 1.4 * self.S / 1000 + 0.4, np.full(14, 50.0)))
                ns = math.exp(-(((lx[-1] - sx) / (0.25 * W)) ** 2 + ((ly[-1] - sy) / (0.3 * H)) ** 2))
                glows.append((lx, ly, key * (0.30 + 0.9 * ns) * sunk * 0.5, 1.0))

    # ---- swallows, the cyclist, dust, shadows ---------------------------------------------------------
    def build_life(self):
        rng = self.rng
        self.birds = [dict(cx=rng.uniform(0.15, 0.85), cy=rng.uniform(0.47, 0.62), ax=rng.uniform(0.15, 0.35),
                           ay=rng.uniform(0.03, 0.08), w=rng.uniform(0.35, 0.6), ph=rng.uniform(0, TAU),
                           size=rng.uniform(0.6, 1.2)) for _ in range(6)]
        n = 700
        self.motes = dict(x=rng.random(n), y=rng.uniform(0.15, 0.95, n), ph=rng.uniform(0, TAU, (3, n)),
                          b=rng.random(n) ** 2, s=rng.uniform(0.5, 1.6, n))
        self.ghosts = [(0.55, 0.020, (120, 200, 140)), (0.85, 0.050, (200, 140, 220)), (1.25, 0.030, (240, 200, 120)),
                       (1.55, 0.075, (130, 170, 240)), (1.90, 0.040, (220, 150, 120)), (2.25, 0.020, (160, 230, 220))]
        a = rng.uniform(0, TAU, 220)
        self.ghost_pts = (np.cos(a), np.sin(a), rng.random(220))
        H, W = self.H, self.W
        r0 = int(math.ceil(self.hz / 2) * 2) + 2
        hh, wh = (H - r0) // 2, W // 2
        yy = (r0 + 2 * np.arange(hh) + 0.5)[:, None]
        xx = (2 * np.arange(wh) + 0.5)[None, :]
        self.g_r0, self.g_hw = r0, (hh, wh)
        self.gz = (self.f * self.hc / (yy - self.hz)) * np.ones_like(xx)
        self.gX = (xx - self.vx) * self.gz / self.f
        self.gxs = xx[0]

    def cyclist_at(self, t):
        z = 5.5 + 1.25 * t + 0.012 * t * t
        X = 1.95 + 0.10 * math.sin(0.3 * t)
        return X, z

    def cyclist_points(self, t):
        """The rider from behind, in metres: (lateral, height, rim weight) of points filling the shape."""
        rng = np.random.default_rng(31)
        crank = 3.4 * t
        parts = []

        def capsule(a, b, rad, n):
            a, b = np.asarray(a, float), np.asarray(b, float)
            q = rng.random(n)
            ang = rng.uniform(0, TAU, n)
            rr = np.sqrt(rng.random(n))
            P = a[None] + (b - a)[None] * q[:, None] + rad * rr[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)
            parts.append(np.hstack([P, (rr ** 10)[:, None]]))

        capsule((0.0, 0.05), (0.0, 0.66), 0.035, 160)                     # the back wheel, edge-on
        capsule((0.0, 0.03), (0.0, 0.70), 0.010, 60)
        for sgn in (-1, 1):                                               # legs pumping
            ph = crank + (0 if sgn > 0 else math.pi)
            hip = (0.09 * sgn, 0.95)
            knee = (0.15 * sgn, 0.66 + 0.10 * math.sin(ph))
            foot = (0.11 * sgn, 0.34 + 0.16 * math.sin(ph))
            capsule(hip, knee, 0.075, 140)
            capsule(knee, foot, 0.055, 110)
        for k in range(5):                                                # the back, widening to the shoulders
            q = k / 4
            capsule((-0.12 - 0.06 * q, 0.98 + 0.12 * k), (0.12 + 0.06 * q, 0.98 + 0.12 * k), 0.08, 150)
        for sgn in (-1, 1):
            capsule((0.19 * sgn, 1.44), (0.24 * sgn, 1.14), 0.05, 90)     # arms down to the bars
        capsule((-0.25, 1.12), (0.25, 1.12), 0.022, 70)                   # handlebars
        capsule((-0.07, 0.93), (0.07, 0.93), 0.04, 40)                     # saddle
        capsule((0.0, 1.58), (0.0, 1.66), 0.11, 220)                      # head
        capsule((-0.05, 1.50), (0.05, 1.50), 0.05, 40)                    # neck
        return np.vstack(parts)

    def draw_cyclist(self, dots, t, amb, key, sunk):
        P = self.cyclist_points(t)
        X0, z0 = self.cyclist_at(t)
        x, y = self.proj(X0 + P[:, 0], P[:, 1], np.full(len(P), z0))
        r = np.clip(self.f * 0.028 / z0, 1.0, 2.6)
        col = DARK * (0.45 + 0.25 * amb.mean()) + key * 0.04 * sunk
        dots.append((x, y, col, r, np.full(len(x), -z0)))
        self.cyc_screen = (x, y, r)
        return X0, z0

    def rim_cyclist(self, glows, key, sunk):
        """Backlit: a thin line of light all round the rider's outline, strongest on head and shoulders."""
        x, y, r = self.cyc_screen
        x0, y0 = int(x.min()) - 4, int(y.min()) - 4
        w, h = int(x.max()) + 5 - x0, int(y.max()) + 5 - y0
        m = np.zeros((h, w, 3), np.float32)
        stamp(m, x - x0, y - y0, np.ones(3, np.float32), r)
        m = m[..., 0] > 0.5
        for _ in range(2):                                                # close the little gaps between dots
            d = m.copy()
            d[1:] |= m[:-1]
            d[:-1] |= m[1:]
            d[:, 1:] |= m[:, :-1]
            d[:, :-1] |= m[:, 1:]
            m = d
        for _ in range(2):
            e = m.copy()
            e[1:] &= m[:-1]
            e[:-1] &= m[1:]
            e[:, 1:] &= m[:, :-1]
            e[:, :-1] &= m[:, 1:]
            m = e
        inner = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
        edge = m.copy()
        edge[1:-1, 1:-1] &= ~inner
        ey, ex = np.nonzero(edge)
        if len(ex):
            up = 1 - (ey - ey.min()) / max(ey.max() - ey.min(), 1)
            glows.append((ex + x0 + 0.5, ey + y0 + 0.5, key[None] * (1.3 * (0.4 + 0.6 * up) * sunk)[:, None], 1.3))

    def draw_shadows(self, img, t, cyc, sx, sunk):
        """Long shadows thrown toward us - the oak, the poles, the rider - and the sheen of gusts on the wheat.
        Worked out on a half-size grid of ground patches, one row (one distance) at a time, then laid over the field."""
        elev, az = self.sun_dir(t)
        Z, X = self.gz, self.gX
        hh, wh = self.g_hw
        zr = Z[:, 0]
        shade = np.zeros((hh, wh), np.float32)
        k = float(ss(math.degrees(elev), -0.5, 2.5))
        if k > 0.01:
            dX, dz = -math.sin(az), -math.cos(az)
            reach = 1.0 / math.tan(max(elev, math.radians(0.6)))
            f2 = self.f / 2

            def strip(ax, az_, bx, bz, w):
                """A shadow line from (ax, az) to (bx, bz) on the ground, w metres wide."""
                lo, hi = min(az_, bz), max(az_, bz)
                rows = np.nonzero((zr >= lo) & (zr <= hi))[0]
                if not len(rows):
                    return
                z = zr[rows]
                Xc = ax + (z - az_) / (bz - az_) * (bx - ax)
                c = (self.vx + self.f * Xc / z - 0.5) / 2
                hw = np.maximum(f2 * w / z, 1.1)
                K = int(2 * hw.max() + 4)
                j = np.floor(c)[:, None].astype(int) + np.arange(-K // 2, K // 2 + 1)[None, :]
                v = np.clip(1.6 - np.abs(j - c[:, None]) / hw[:, None], 0, 1)
                ok = (j >= 0) & (j < wh)
                rr = np.broadcast_to(rows[:, None], j.shape)[ok]
                jj = j[ok]
                shade[rr, jj] = np.maximum(shade[rr, jj], v[ok])

            tx, tz = self.oak_base
            cxo, czo = tx + dX * 8.5 * reach, tz + dz * 8.5 * reach          # the crown's shadow: a long oval
            a_, b_ = 5.5 / math.sin(max(elev, math.radians(0.6))) * 0.9, 6.0
            zlo, zhi = czo - a_ - b_, czo + a_ + b_
            rows = np.nonzero((zr >= zlo) & (zr <= zhi))[0]
            if len(rows):
                r0, r1 = rows[0], rows[-1] + 1
                Xs, Zs = X[r0:r1], Z[r0:r1]
                al = (Xs - cxo) * dX + (Zs - czo) * dz
                ac = -(Xs - cxo) * dz + (Zs - czo) * dX
                np.maximum(shade[r0:r1], np.clip((1 - (al / a_) ** 2 - (ac / b_) ** 2) * 2.5, 0, 1), out=shade[r0:r1])
            strip(tx, tz, tx + dX * 5.0 * reach, tz + dz * 5.0 * reach, 0.5)
            for px_, pz in self.poles:
                strip(px_, pz, px_ + dX * self.pole_h * reach, pz + dz * self.pole_h * reach, 0.13)
            X0, z0 = cyc
            strip(X0, z0, X0 + dX * 1.7 * reach, z0 + dz * 1.7 * reach, 0.22)
            shade *= k
        # the gusts over the whole field: the same waves as on the ears, split into a row part and a column
        # part so that no sine is taken per cell
        bcol = 2.4 * (self.gxs - self.vx) / self.f
        a1, b1 = 0.55 * zr + 1.9 * t, bcol
        a2, b2 = 0.21 * zr + 0.9 * t + 1.0, -1.8 * bcol
        sa = np.sin(a1)[:, None] * np.cos(b1)[None, :] + np.cos(a1)[:, None] * np.sin(b1)[None, :]
        sb = np.sin(a2)[:, None] * np.cos(b2)[None, :] + np.cos(a2)[:, None] * np.sin(b2)[None, :]
        g = np.clip(((sa * 0.5 + 0.5) ** 3) * 0.8 + ((sb * 0.5 + 0.5) ** 4) * 0.5, 0, 1)
        near = np.exp(-((self.gxs - sx) / (0.22 * self.W)) ** 2)[None, :]
        sheen = 1 + 0.22 * g * (0.3 + 0.7 * near) * sunk * (1 - 0.6 * shade)
        tint = np.array([0.50, 0.52, 0.72], np.float32)
        fac = sheen[..., None] * (1 - 0.85 * shade[..., None] * (1 - tint))
        r0 = self.g_r0
        view = img[r0:r0 + 2 * hh, :2 * wh].reshape(hh, 2, wh, 2, 3)
        view *= fac.astype(np.float32)[:, None, :, None, :]

    def draw_birds(self, dots, t, amb, key, sunk):
        W, H, S = self.W, self.H, self.S
        if not hasattr(self, "bird_shape"):
            rng = np.random.default_rng(9)
            u, v = [], []
            q = rng.random(120)
            u.append(0.55 - 1.05 * q)                                        # body: slim, tapering behind
            v.append(rng.uniform(-1, 1, 120) * 0.09 * np.sin(np.pi * np.clip(q * 0.95 + 0.05, 0, 1)) ** 0.6)
            for side in (-1, 1):                                             # sickle wings, swept back
                w = rng.random(130)
                c = rng.random(130)
                chord = 0.26 * (1 - w) ** 0.8 + 0.04
                lead = 0.22 - 0.55 * w ** 1.6
                u.append(lead - chord * c)
                v.append(side * (0.06 + 1.0 * w))
                tw = rng.random(30)                                          # the long outer tail streamers
                u.append(-0.50 - 0.55 * tw)
                v.append(side * (0.03 + 0.20 * tw ** 1.2))
            self.bird_shape = (np.concatenate(u), np.concatenate(v))
        bu, bv = self.bird_shape
        for B in self.birds:
            a = B["w"] * t + B["ph"]
            x = W * (B["cx"] + B["ax"] * math.sin(a) + 0.05 * math.sin(2.7 * a))
            y = H * (B["cy"] + B["ay"] * math.sin(2 * a + 0.5) + 0.01 * math.sin(5 * a))
            vx = B["ax"] * math.cos(a) + 0.135 * math.cos(2.7 * a)
            vy = 2 * B["ay"] * math.cos(2 * a + 0.5)
            hd = math.atan2(vy * H, vx * W)
            s = 0.024 * S * B["size"] * (0.7 + 0.6 * (B["cy"] - 0.45) / 0.2)
            beat = math.sin(0.7 * t + B["ph"]) > -0.2
            flap = math.sin(16 * t + B["ph"]) if beat else 0.5              # flap, then glide
            span = 0.55 + 0.45 * abs(flap)
            c, sn = math.cos(hd), math.sin(hd)
            v = bv * np.where(np.abs(bv) > 0.12, span, 1.0)
            X = x + (bu * c - v * sn * 0.8) * s
            Y = y + (bu * sn + v * c * 0.8) * s - np.abs(v) * flap * 0.20 * s
            dots.append((X, Y, DARK * 0.85 + key * 0.05 * sunk, 1.2 * S / 1000 + 0.3, np.full(len(X), 40.0)))

    def draw_motes(self, glows, t, sx, sy, sunk):
        M, W, H = self.motes, self.W, self.H
        x = (M["x"] + 0.004 * t * M["s"] + 0.015 * np.sin(0.4 * t + M["ph"][0])) % 1.0 * W
        y = (M["y"] - 0.003 * t * M["s"] + 0.01 * np.sin(0.5 * t + M["ph"][1])) * H
        fwd = np.exp(-(((x - sx) / (0.30 * W)) ** 2 + ((y - sy) / (0.35 * H)) ** 2))   # dust shines toward the sun
        tw = 0.5 + 0.5 * np.sin(3 * t + M["ph"][2])
        b = (0.04 + 0.9 * fwd) * tw * (0.3 + 0.7 * M["b"]) * sunk
        on = b > 0.02
        glows.append((x[on], y[on], np.array([255, 226, 170])[None] * (b[on] * 0.9)[:, None], 1.0 + 0.6 * (M["s"][on] > 1.2)))

    def draw_flare(self, glows, t, sx, sy, sunk):
        cx, cy = 0.5 * self.W, 0.5 * self.H
        vis = sunk * float(ss(sy, self.hz + 0.03 * self.H, self.hz - 0.02 * self.H))
        if vis < 0.02:
            return
        cu, cv, cr = self.ghost_pts
        for m, rad, col in self.ghosts:
            gx = sx + (cx - sx) * m
            gy = sy + (cy - sy) * m
            r = rad * self.S
            rr = r * np.sqrt(cr)
            glows.append((gx + cu * rr, gy + cv * rr, np.array(col) * 0.06 * vis, 1.6))
            glows.append((gx + cu[:120] * r, gy + cv[:120] * r, np.array(col) * 0.10 * vis, 1.2))
        self.haze.add(np.array([sx]), np.array([sy]), np.array([255, 210, 150]) * 0.9 * vis, 40 * self.S / 1000)

    # ---- the frame ---------------------------------------------------------------------------------
    def render(self, t):
        img = self.backdrop.at(t)
        amb = keyed(t, self.amb_keys)
        key = keyed(t, self.key_keys)
        sx, sy = self.sun_xy(t)
        elev, _ = self.sun_dir(t)
        sunk = float(ss(math.degrees(elev), -1.5, 3.0)) * 0.85 + 0.15         # how much direct sun is left
        warm = float(ss(t, 0, 50))
        core = np.array([255, 246, 220]) * (1 - warm) + np.array([255, 196, 120]) * warm
        self.sun.draw(img, sx, sy, core, np.array([150, 110, 50]) * (0.4 + 0.6 * sunk), self.sky_mask(), None)
        dots, glows = [], []
        cyc = self.draw_cyclist(dots, t, amb, key, sunk)
        self.draw_shadows(img, t, cyc, sx, sunk)
        self.draw_mid(dots, t, sx, amb, key, sunk)
        self.draw_near(dots, glows, t, sx, sy, amb, key, sunk)
        self.draw_tall(dots, glows, t, sx, sy, amb, key, sunk)
        self.draw_birds(dots, t, amb, key, sunk)
        paint(img, dots)
        self.rim_cyclist(glows, key, sunk)
        self.draw_motes(glows, t, sx, sy, sunk)
        self.draw_flare(glows, t, sx, sy, sunk)
        xs, ys, cs, rs = [], [], [], []                                       # all the light in one pass
        for x, y, c, r in glows:
            x = np.atleast_1d(np.asarray(x, np.float32))
            xs.append(x)
            ys.append(np.broadcast_to(np.asarray(y, np.float32), x.shape))
            cs.append(np.broadcast_to(np.asarray(c, np.float32), (len(x), 3)))
            rs.append(np.broadcast_to(np.asarray(r, np.float32), x.shape))
        r = np.concatenate(rs)
        big = r > 1.25
        x, y, c = np.concatenate(xs), np.concatenate(ys), np.concatenate(cs)
        glow(img, x[~big], y[~big], c[~big], r[~big])
        glow(img, x[big], y[big], c[big], r[big])
        st = float(ss(t, 52.0, 58.0))                                        # the first star
        if st > 0:
            glow(img, np.array([0.30 * self.W]), np.array([0.16 * self.H]), np.array([255, 250, 230]) * 0.9 * st, 1.6)
        self.haze.apply(img, 1.0)
        return np.clip(img, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    run(GoldenHour, "Golden hour", "golden_hour", T_END, STAGES)
