"""
kelp.py - looking straight up through a kelp forest, in dots. A seamless one-minute loop.

We are lying on the sea floor looking up. Giant kelp rises on every side, its stalks
converging toward the bright circle where the sky shows through the surface, golden blades
glowing where the light comes through them and leafy mats floating overhead. Sunbeams fan
down and shift as swells pass. A silver school of fish turns in a slow ring. A seal comes
diving out of the canopy straight through them - the school bursts apart and flows back
together behind it - and a sea otter floats on its back in the light, paws busy. Bubbles
wobble up, specks drift, and a pair of bright orange fish weave between the stalks.

    python kelp.py            the forest, looping forever (Space pause, R restart, S save frame, Esc quit)
    python kelp.py --png      save five moments
    python kelp.py --gif      save one seamless loop as a GIF
"""
import math

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))   # the shared engine: engine/pointillism.py

from pointillism import Haze, Sun, glow, paint, run, shades, ss, stamp, unit

T_END = 60.0
TAU = 2 * math.pi
STAGES = [8.0, 22.0, 30.6, 33.2, 47.0]
D = 6.5                                   # depth of the surface above us (metres)

KELP = [(64, 56, 22), (78, 66, 26), (92, 78, 30), (56, 50, 24)]
GOLD = np.array([235, 196, 90], np.float32)
BLADDER = [(150, 124, 48), (176, 148, 60)]
SILVER = np.array([200, 214, 220], np.float32)
DARK = np.array([22, 40, 48], np.float32)
ORANGE = [(250, 112, 24), (255, 132, 40), (236, 96, 20)]
SEAL = [(58, 60, 64), (70, 72, 74), (48, 50, 54)]


def lap(t, n, ph=0.0):
    """An angle that goes round n whole times a minute (so the minute loops)."""
    return TAU * n * t / T_END + ph


def spline(P, s):
    """Catmull-Rom through 3D points P at s in [0, 1]."""
    P = np.asarray(P, np.float64)
    n = len(P) - 1
    f = np.clip(s, 0, 1) * n
    i = min(int(f), n - 1)
    u = f - i
    p0, p1, p2, p3 = P[max(i - 1, 0)], P[i], P[i + 1], P[min(i + 2, n)]
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)


class Kelp:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.S = min(W, H)
        self.rng = np.random.default_rng(29)
        self.cx, self.cy = 0.56 * W, 0.40 * H          # straight up
        self.Rw = 0.30 * self.S                          # Snell's window
        self.f = self.Rw / math.tan(math.radians(48.6)) * 1.0
        self.haze = Haze(W, H, max(2, round(4 * self.S / 1000)))
        self.sun = Sun(np.random.default_rng(4), 0.016 * self.S, 6.0)
        self.build_water()
        self.build_kelp()
        self.build_life()

    def proj(self, X, Y, Z):
        Z = np.maximum(Z, 0.05)
        return self.cx + self.f * X / Z, self.cy + self.f * Y / Z, self.f / Z

    # ---- the water and the window -----------------------------------------------------------------
    def water_col(self, d):
        d = np.asarray(d, np.float64)
        inside = np.stack([np.interp(d, [0, 0.5, 0.85, 0.97, 1.0], c) for c in ((214, 196, 150, 166, 196),
                                                                              (240, 232, 206, 212, 234),
                                                                              (236, 232, 214, 218, 236))], -1)
        out = np.stack([np.interp(d, [1.0, 1.03, 1.3, 2.0, 3.2], c) for c in ((40, 34, 22, 12, 6), (104, 96, 74, 50, 30),
                                                                             (116, 110, 92, 70, 48))], -1)
        return np.where((d <= 1.0)[..., None], inside, out).astype(np.float32)

    def build_water(self):
        W, H, rng = self.W, self.H, np.random.default_rng(6)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.hypot(xx - self.cx, yy - self.cy) / self.Rw
        img = self.water_col(d) * 0.9
        n = int(W * H * 0.4)
        x, y = rng.uniform(0, W, n), rng.uniform(0, H, n)
        dd = np.hypot(x - self.cx, y - self.cy) / self.Rw
        col = self.water_col(dd + rng.normal(0, 0.012, n)) * (1 + rng.normal(0, 0.05, (n, 1)))
        stamp(img, x, y, col, rng.choice([1.0, 1.5, 2.0], n, p=[0.3, 0.45, 0.25]))
        self.base = img
        self.lum = np.clip(img.mean(2) / 230.0, 0, 1)
        n = int(9000 * (self.Rw / 300) ** 2)                  # ripples of light in the window
        a, r = rng.uniform(0, TAU, n), self.Rw * np.sqrt(rng.random(n)) * 1.01
        k = TAU / (0.11 * self.Rw)
        self.net = dict(x=self.cx + np.cos(a) * r, y=self.cy + np.sin(a) * r, u=np.cos(a) * r * k, v=np.sin(a) * r * k,
                        d=r / self.Rw)
        m = 1600
        self.rim_a = np.linspace(0, TAU, m, endpoint=False)
        self.rim_j = rng.normal(0, 1, m)
        a = np.sort(rng.uniform(0, TAU, 44))                    # sunbeams
        self.beams = dict(a=a, w=rng.uniform(0.5, 1.0, 44), n=rng.integers(2, 6, 44), ph=rng.uniform(0, TAU, 44))
        self.sunp = (self.cx - 0.55 * self.Rw, self.cy - 0.35 * self.Rw)

    def draw_window(self, img, t):
        N = self.net
        a = lap(t, 1)
        s1 = np.sin(N["u"] + 5 * a + 1.3 * np.sin(0.8 * N["v"] - 3 * a))
        s2 = np.sin(1.1 * N["v"] - 4 * a + 1.3 * np.sin(0.9 * N["u"] + 2 * a))
        c = np.clip(1 - np.abs(s1 + s2) * 0.9, 0, 1) ** 3
        on = c > 0.04
        glow(img, N["x"][on], N["y"][on], np.array([255, 255, 240]) * (0.40 * c[on] * (1 - 0.5 * N["d"][on]))[:, None], 1.3)
        th = self.rim_a                                           # the edge of the window shivers
        R = self.Rw * (1 + 0.012 * np.sin(5 * th + lap(t, 3)) + 0.008 * np.sin(11 * th - lap(t, 5)) + 0.003 * self.rim_j)
        glow(img, self.cx + np.cos(th) * R, self.cy + np.sin(th) * R, np.array([200, 240, 236]) * 0.22, 1.2)
        sx, sy = self.sunp                                        # the sun, broken up by the swell
        sx += 0.012 * self.Rw * math.sin(lap(t, 4))
        sy += 0.010 * self.Rw * math.cos(lap(t, 3, 1.0))
        self.sun.draw(img, sx, sy, np.array([255, 252, 236]), np.array([90, 90, 70]))
        B = self.beams                                            # beams fanning down from the light
        k = np.linspace(0.15, 1.0, 50)[None, :]
        r = self.Rw * (0.5 + 2.6 * k)
        b = (0.5 + 0.5 * np.sin(lap(t, B["n"], B["ph"]))) ** 2 * B["w"]
        x = sx + np.cos(B["a"])[:, None] * r
        y = sy + np.sin(B["a"])[:, None] * r
        prof = np.exp(-((k - 0.42) / 0.30) ** 2)
        self.haze.add(x.ravel(), y.ravel(), (np.array([70, 120, 110])[None, None, :] * (b[:, None] * prof)[..., None]
                                             * 0.30).reshape(-1, 3), 14 * self.S / 1000)

    # ---- kelp -----------------------------------------------------------------------------------------
    def build_kelp(self):
        rng, f = self.rng, self.f
        st = []
        for h in range(15):                                       # holdfasts, each a bundle of stalks
            a = TAU * h / 15 + rng.normal(0, 0.15)
            r0 = 1.3 + 7.0 * rng.random() ** 1.2
            for k in range(2):
                aa, rr = a + rng.normal(0, 0.05), r0 + rng.normal(0, 0.25)
                top = D if rng.random() < 0.7 else rng.uniform(3.5, D - 0.6)
                st.append(dict(X=rr * math.cos(aa), Y=rr * math.sin(aa), top=top, a=aa, ph=1.3 * h + rng.normal(0, 0.2),
                               cang=aa + rng.normal(0, 0.9), clen=rng.uniform(1.4, 3.0) if top == D else 0.0))
        self.st = st
        P = dict(i=[], zb=[], ang=[], tilt=[], ln=[], wd=[], can=[], ph=[])
        db, du, dv, dt = [], [], [], []

        def blade(i, z, ang, tilt, can):
            ln, wd = rng.uniform(0.9, 1.4), rng.uniform(0.18, 0.28)
            rad = float(np.clip(f * 0.03 / z, 1.4, 3.0))
            nu = int(np.clip(f * ln / z * 0.8 / (1.9 * rad), 4, 28))     # an even grid, so the blade reads solid
            nv = int(np.clip(f * wd / z * 0.8 / (1.9 * rad), 1, 6))
            m = nu * nv
            b = len(P["i"])
            for k, v in zip(P, (i, z, ang, tilt, ln, wd, can, rng.uniform(0, TAU))):
                P[k].append(v)
            db.append(np.full(m, b))
            du.append(np.clip((np.repeat(np.arange(nu), nv) + rng.uniform(0.2, 0.8, m)) / nu, 0, 1))
            dv.append(np.clip((np.tile(np.arange(nv), nu) + rng.uniform(0.2, 0.8, m)) / nv * 2 - 1, -1, 1))
            dt.append(rng.random(m))

        for i, S in enumerate(st):
            z, side = 0.5, 1
            while z < S["top"] - 0.2:                             # blades up the stalk, alternating sides
                blade(i, z, S["a"] + side * rng.uniform(1.0, 2.0), rng.uniform(0.35, 0.85), -1.0)
                z += rng.uniform(0.42, 0.60)
                side = -side
            for sc in np.arange(0.0, S["clen"], 0.35):           # the floating mat at the surface
                blade(i, D - 0.06, S["cang"] + side * rng.uniform(0.9, 1.9), 0.0, sc)
                side = -side
        self.bp = {k: np.array(v, np.float64) for k, v in P.items()}
        self.bp["i"] = self.bp["i"].astype(int)
        self.bd = dict(b=np.concatenate(db).astype(int), u=np.concatenate(du), v=np.concatenate(dv))
        tone = np.concatenate(dt)
        col = shades(rng, KELP, tone, 0.9)
        av = np.abs(self.bd["v"])
        self.bd["col"] = col * np.where(av < 0.15, 0.75, np.where(av > 0.8, 1.15, 1.0))[:, None]   # midrib, thin edges
        self.stX = np.array([S["X"] for S in st])
        self.stY = np.array([S["Y"] for S in st])
        self.stph = np.array([S["ph"] for S in st])
        self.sttop = np.array([S["top"] for S in st])
        self.stca = np.array([S["cang"] for S in st])
        self.stcl = np.array([S["clen"] for S in st])
        m = 220                                                   # points up each stalk, close together near us
        q = np.arange(m) / (m - 1)
        self.stz = 0.35 * (self.sttop[:, None] / 0.35) ** q[None, :]

    def sway(self, z, ph, t):
        """Sideways push of the swell at height z (metres), as (dx, dy)."""
        k = (z / D) ** 1.5
        s = np.sin(lap(t, 5) + ph - 0.45 * z)
        c = np.sin(lap(t, 3) + 1.7 * ph - 0.3 * z)
        return k * (0.50 * s + 0.15 * c), k * (0.36 * s - 0.18 * c)

    def shade(self, x, y, dist, col, fogk=0.8):
        """Backlit kelp turns amber against the light; everything fades into the water with distance."""
        xi = np.clip(x.astype(int), 0, self.W - 1)
        yi = np.clip(y.astype(int), 0, self.H - 1)
        L = self.lum[yi, xi]
        c = col * (0.35 + 0.25 * L)[:, None] + GOLD * (L ** 1.5 * 0.30)[:, None]
        fog = (1 - np.exp(-dist / 12.0))[:, None] * fogk
        return c * (1 - fog) + self.base[yi, xi] * fog

    def draw_kelp(self, dots, t):
        B, W, H = self.bp, self.W, self.H
        i = B["i"]
        stem = B["can"] < 0
        z0 = B["zb"]
        sx, sy = self.sway(z0, self.stph[i], t)
        tx, ty = self.sway(np.full_like(z0, D), self.stph[i], t)
        ca = self.stca[i]
        bx = np.where(stem, self.stX[i] + sx, self.stX[i] + tx + np.cos(ca) * np.maximum(B["can"], 0))
        by = np.where(stem, self.stY[i] + sy, self.stY[i] + ty + np.sin(ca) * np.maximum(B["can"], 0))
        px, py, sc = self.proj(bx, by, z0)                        # skip blades that are off the frame
        reach = sc * B["ln"] * 1.1 + 4
        vis = (px > -reach) & (px < W + reach) & (py > -reach) & (py < H + reach)
        bd = self.bd
        keep = vis[bd["b"]]
        b, u, v, col = bd["b"][keep], bd["u"][keep], bd["v"][keep], bd["col"][keep]
        st_, ph = stem[b], B["ph"][b]
        flut = 0.18 * np.sin(lap(t, 7) + ph + 2 * u)
        tl = B["tilt"][b] + np.where(st_, flut, 0)
        ang = B["ang"][b] + np.where(st_, 0.3 * flut, 0.25 * np.sin(lap(t, 4) + ph))
        ca_, sa_ = np.cos(ang), np.sin(ang)
        ax = np.where(st_, np.sin(tl) * ca_, ca_)
        ay = np.where(st_, np.sin(tl) * sa_, sa_)
        az = np.where(st_, np.cos(tl), -0.12)
        wdt = B["wd"][b] * np.sin(np.pi * np.clip(0.06 + 0.94 * u, 0, 1)) ** 0.6 * (1 + 0.25 * np.sin(26 * u + ph))
        L = B["ln"][b] * u
        X = bx[b] + ax * L - sa_ * v * wdt
        Y = by[b] + ay * L + ca_ * v * wdt
        Z = np.minimum(z0[b] + az * L + 0.04 * np.sin(18 * u + lap(t, 6) + ph) * np.where(st_, 1, 0.3), D - 0.02)
        x, y, s_ = self.proj(X, Y, Z)
        on = (x > -3) & (x < W + 3) & (y > -3) & (y < H + 3) & (Z > 0.3)
        x, y, s_, X, Y, Z, col = x[on], y[on], s_[on], X[on], Y[on], Z[on], col[on]
        dist = np.sqrt(X * X + Y * Y + Z * Z)
        dots.append((x, y, self.shade(x, y, dist, col), np.clip(s_ * 0.03, 1.4, 3.0), -dist))
        sb = vis & stem                                            # gas floats where the blades join
        x, y, s_ = px[sb], py[sb], sc[sb]
        dist = np.sqrt(bx[sb] ** 2 + by[sb] ** 2 + z0[sb] ** 2)
        col = np.broadcast_to(np.array(BLADDER[1], np.float32), (len(x), 3))
        dots.append((x, y, self.shade(x, y, dist, col), np.clip(s_ * 0.035, 1.2, 3.0), -dist + 0.02))
        z = self.stz                                              # the stalks
        sx, sy = self.sway(z, self.stph[:, None], t)
        X, Y = self.stX[:, None] + sx, self.stY[:, None] + sy
        x, y, s_ = self.proj(X, Y, z)
        X, Y, z, x, y, s_ = (a.ravel() for a in (X, Y, z, x, y, s_))
        on = (x > -5) & (x < W + 5) & (y > -5) & (y < H + 5)
        X, Y, z, x, y, s_ = X[on], Y[on], z[on], x[on], y[on], s_[on]
        dist = np.sqrt(X * X + Y * Y + z * z)
        col = np.broadcast_to(np.array(KELP[2], np.float32) * 0.8, (len(x), 3))
        dots.append((x, y, self.shade(x, y, dist, col), np.clip(s_ * 0.028, 1.0, 3.0), -dist + 0.01))
        top = self.sttop >= D                                     # stalks lying along the surface
        q = np.linspace(0, 1, 50)[None, :]
        tx, ty = self.sway(np.full(int(top.sum()), D), self.stph[top], t)
        X = (self.stX[top] + tx)[:, None] + np.cos(self.stca[top])[:, None] * q * self.stcl[top][:, None]
        Y = (self.stY[top] + ty)[:, None] + np.sin(self.stca[top])[:, None] * q * self.stcl[top][:, None]
        x, y, s_ = self.proj(X.ravel(), Y.ravel(), np.full(X.size, D - 0.05))
        dist = np.sqrt(X.ravel() ** 2 + Y.ravel() ** 2 + D * D)
        col = np.broadcast_to(np.array(BLADDER[0], np.float32) * 0.7, (len(x), 3))
        dots.append((x, y, self.shade(x, y, dist, col), np.clip(s_ * 0.035, 1.0, 2.5), -dist + 0.02))

    # ---- fish, the seal, the otter, bubbles ---------------------------------------------------------------
    def build_life(self):
        rng = self.rng
        n = 420
        self.fish = dict(th=rng.uniform(0, TAU, n), R=0.7 + 1.1 * rng.random(n), h=rng.normal(0, 0.35, n),
                         ph=rng.uniform(0, TAU, n), burst=0.6 + 0.8 * rng.random(n), lag=rng.uniform(0, 0.6, n))
        self.seal_path = [(-7.5, -5.0, 6.4), (-4.0, -3.4, 6.1), (-0.8, -1.8, 4.9), (1.3, -0.8, 3.9), (2.5, 0.5, 2.5),
                          (3.0, 1.8, 1.3), (3.3, 2.8, 0.6)]
        n = 70
        self.bub = dict(X=rng.normal(0, 2.2, n), Y=rng.normal(0, 2.2, n), n=rng.integers(2, 5, n), ph=rng.random(n),
                        r=rng.uniform(0.006, 0.02, n), w=rng.uniform(0, TAU, n))
        n = 1400
        self.snow = dict(X=rng.uniform(-5, 5, n), Y=rng.uniform(-5, 5, n), Z=rng.uniform(0.6, 6.2, n),
                         ph=rng.uniform(0, TAU, n), b=rng.random(n))
        self.garibaldi = [(-1.4, 1.2, 2.6, 1.1, 0.6, 2, 0.0), (1.9, 1.7, 3.2, 0.9, 0.7, -3, 2.5)]
        self.otter_pts = self.build_otter(np.random.default_rng(31))

    def school_center(self, t):
        return np.array([1.2 + 0.6 * math.cos(lap(t, 1)), -0.8 + 0.4 * math.sin(lap(t, 1)), 3.9 + 0.2 * math.sin(lap(t, 2))])

    def seal_at(self, t):
        a, b = 25.5, 36.5
        if not a <= t <= b:
            return None
        s = (t - a) / (b - a)
        s = s * 0.8 + 0.2 * s * s
        return spline(self.seal_path, s)

    def fish_at(self, t):
        F = self.fish
        C = self.school_center(t)
        th = F["th"] + lap(t, 4) + 0.15 * np.sin(lap(t, 7) + F["ph"])
        R = F["R"] + 0.25 * np.sin(lap(t, 3) + F["ph"]) + 0.2 * np.sin(2 * th + F["ph"] * 3)
        X = C[0] + R * np.cos(th)
        Y = C[1] + R * np.sin(th) * 0.8
        Z = C[2] + F["h"] + 0.22 * np.sin(2 * th + F["ph"])
        e = ss(t - F["lag"], 30.0, 31.0) * (1 - ss(t - F["lag"], 33.0, 39.0))     # the school flashes apart
        if np.any(e > 0):
            Q = np.array([1.3, -0.8, 3.9])
            dx, dy, dz = X - Q[0], Y - Q[1], Z - Q[2]
            r = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-3
            k = e * F["burst"] * 1.7 / (0.6 + r)
            X, Y, Z = X + dx / r * k * 1.6, Y + dy / r * k * 1.6, Z + dz / r * k
        S = self.seal_at(t)
        if S is not None:                                        # and dodges the seal itself
            dx, dy, dz = X - S[0], Y - S[1], Z - S[2]
            r = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-3
            k = 0.9 * np.exp(-(r / 1.1) ** 2)
            X, Y, Z = X + dx / r * k, Y + dy / r * k, Z + dz / r * k
        return X, Y, Z

    def draw_fish(self, dots, glows, t):
        X, Y, Z = self.fish_at(t)
        X2, Y2, Z2 = self.fish_at(t + 0.04)
        hx, hy, hz = X2 - X, Y2 - Y, Z2 - Z
        n = np.sqrt(hx * hx + hy * hy + hz * hz) + 1e-6
        hx, hy, hz = hx / n, hy / n, hz / n
        s = np.linspace(-0.5, 0.5, 6)[None, :] * 0.16
        PX, PY, PZ = X[:, None] + hx[:, None] * s, Y[:, None] + hy[:, None] * s, Z[:, None] + hz[:, None] * s
        x, y, sc = self.proj(PX.ravel(), PY.ravel(), PZ.ravel())
        dist = np.sqrt(PX ** 2 + PY ** 2 + PZ ** 2).ravel()
        xi = np.clip(x.astype(int), 0, self.W - 1)
        yi = np.clip(y.astype(int), 0, self.H - 1)
        Lw = self.lum[yi, xi][:, None]
        flash = np.repeat(np.abs(np.sin(3 * np.arctan2(hy, hx) + self.fish["ph"])) ** 12, 6)
        col = SILVER * (1 - Lw) * 0.75 + DARK * Lw + (np.array([90, 90, 80]) * flash[:, None] * (1 - Lw))
        fog = (1 - np.exp(-dist / 14.0))[:, None] * 0.8
        col = col * (1 - fog) + self.base[yi, xi] * fog
        taper = np.tile(np.array([0.6, 0.9, 1.0, 1.0, 0.8, 0.6]), len(X))
        dots.append((x, y, col, np.clip(sc * 0.022 * taper, 0.9, 2.6), -dist))
        g = flash > 0.5
        glows.append((x[g][::6], y[g][::6], np.array([200, 220, 230]) * 0.5, 1.2))

    def draw_seal(self, dots, glows, t):
        S = self.seal_at(t)
        if S is None:
            return
        S2 = self.seal_at(t + 0.05)
        if S2 is None:
            S2 = S + (S - self.seal_at(t - 0.05))
        hd = unit(S2 - S)
        side = unit(np.cross(hd, [0, 0, 1.0]) + 1e-6)
        up = np.cross(side, hd)
        rng = np.random.default_rng(5)
        n = 1800
        s = rng.random(n)                                         # 0 nose .. 1 tail
        ang = rng.uniform(0, TAU, n)
        rad = 0.21 * np.sin(np.pi * np.clip(0.08 + 0.85 * s, 0, 1)) ** 0.8 * (1 - 0.35 * s)
        r = np.sqrt(rng.random(n))
        wt = 0.10 * math.sin(lap(t, 70))
        wig = wt * s ** 2                                         # the hind end sculls side to side
        P = S[None, :] - hd[None, :] * (s * 1.6)[:, None] + side[None, :] * (np.cos(ang) * r * rad + wig)[:, None] \
            + up[None, :] * (np.sin(ang) * r * rad)[:, None]
        cols = shades(rng, SEAL, rng.random(n))
        k = 160                                                   # flippers: two at the front, a fan behind
        fs = rng.random(k)
        for sgn in (-1, 1):
            fl = S - hd * 0.45 + side * sgn * (0.2 + 0.25 * fs[:, None]) - hd * 0.12 * fs[:, None] * (1 + 0.5 * math.sin(lap(t, 30)))
            P = np.vstack([P, fl])
            cols = np.vstack([cols, shades(rng, SEAL, rng.random(k)) * 0.8])
        for sgn in (-1, 1):
            fl = S - hd * (1.6 + 0.3 * fs[:, None]) + side * sgn * (0.05 + 0.15 * fs[:, None]) + side * wt * 1.3
            P = np.vstack([P, fl])
            cols = np.vstack([cols, shades(rng, SEAL, rng.random(k)) * 0.75])
        x, y, sc = self.proj(P[:, 0], P[:, 1], P[:, 2])
        dist = np.linalg.norm(P, axis=1)
        xi = np.clip(x.astype(int), 0, self.W - 1)
        yi = np.clip(y.astype(int), 0, self.H - 1)
        Lw = self.lum[yi, xi][:, None]
        col = cols * (0.55 - 0.35 * Lw) + np.array([40, 70, 80]) * 0.3
        fog = (1 - np.exp(-dist / 14.0))[:, None] * 0.7
        col = col * (1 - fog) + self.base[yi, xi] * fog
        dots.append((x, y, col, np.clip(sc * 0.03, 1.2, 3.0), -dist))
        b = np.random.default_rng(int(t * 20)).random(6)        # a few bubbles off its back
        bp = S[None, :] - hd[None, :] * (0.3 + b * 1.2)[:, None]
        bx, by, bs = self.proj(bp[:, 0], bp[:, 1], bp[:, 2] + 0.2)
        glows.append((bx, by, np.array([180, 220, 230]) * 0.5, 1.5))

    def build_otter(self, rng):
        pts, cols = [], []

        def blob(n, cx, cy, a, b, col):
            r, q = np.sqrt(rng.random(n)), rng.uniform(0, TAU, n)
            pts.append(np.stack([cx + a * r * np.cos(q), cy + b * r * np.sin(q)], 1))
            cols.append(np.broadcast_to(np.asarray(col, np.float32), (n, 3)))

        blob(1500, 0.0, 0.0, 0.62, 0.24, (34, 30, 28))          # body (metres, head along +x)
        blob(420, 0.74, 0.0, 0.17, 0.15, (38, 34, 30))           # head
        blob(60, 0.82, -0.13, 0.05, 0.035, (30, 28, 26))
        blob(60, 0.82, 0.13, 0.05, 0.035, (30, 28, 26))
        blob(320, -0.90, 0.0, 0.30, 0.075, (30, 28, 26))         # tail
        for sgn in (-1, 1):
            blob(200, -0.62, sgn * 0.21, 0.15, 0.075, (28, 26, 24))  # hind feet
        return np.concatenate(pts), np.concatenate(cols)

    def draw_otter(self, dots, t):
        pts, cols = self.otter_pts
        cx, cy = 0.2 + 0.45 * math.cos(lap(t, 1)), 0.7 + 0.3 * math.sin(lap(t, 1))
        ang = 2.2 + 0.35 * math.sin(lap(t, 1, 0.5))
        c, s = math.cos(ang), math.sin(ang)
        X = cx + pts[:, 0] * c - pts[:, 1] * s
        Y = cy + pts[:, 0] * s + pts[:, 1] * c
        paws = []                                                # paws busy at its chest
        for sgn in (-1, 1):
            q = 0.40 + 0.07 * math.sin(lap(t, 24) + sgn)
            px, py = q, sgn * (0.12 + 0.04 * math.sin(lap(t, 24) + 1 + sgn))
            paws.append((cx + px * c - py * s, cy + px * s + py * c))
        rng = np.random.default_rng(8)
        for px, py in paws:
            X = np.concatenate([X, px + rng.normal(0, 0.045, 80)])
            Y = np.concatenate([Y, py + rng.normal(0, 0.045, 80)])
        col = np.vstack([cols, np.broadcast_to(np.array([26, 24, 22], np.float32), (160, 3))])
        Z = np.full(len(X), D - 0.03 + 0.03 * math.sin(lap(t, 9)))
        x, y, sc = self.proj(X, Y, Z)
        dots.append((x, y, col, np.clip(sc * 0.035, 1.0, 2.2), -100.0))

    def draw_garibaldi(self, dots, t):
        rng = np.random.default_rng(12)
        for gx, gy, gz, rx, ry, laps, ph in self.garibaldi:
            def at(tt):
                th = lap(tt, laps, ph)
                return np.array([gx + rx * math.cos(th), gy + ry * math.sin(2 * th) * 0.6, gz + 0.3 * math.sin(th)])
            P0, P1 = at(t), at(t + 0.05)
            hd = unit(P1 - P0)
            side = unit(np.cross(hd, [0, 0, 1.0]))
            n = 260
            s, v = rng.random(n), rng.uniform(-1, 1, n)
            body = 0.11 * np.sin(np.pi * np.clip(0.05 + 0.9 * s, 0, 1)) ** 0.7
            wt = 0.03 * math.sin(lap(t, 90))
            wag = wt * s ** 2
            P = P0[None, :] - hd[None, :] * (s * 0.30)[:, None] + side[None, :] * (v * body + wag)[:, None]
            m = 70                                                # the tail fin
            ts, tv = rng.random(m), rng.uniform(-1, 1, m)
            T = P0[None, :] - hd[None, :] * (0.30 + 0.08 * ts)[:, None] + side[None, :] * (tv * 0.05 * (0.5 + ts) + wt * 1.4)[:, None]
            P = np.vstack([P, T])
            x, y, sc = self.proj(P[:, 0], P[:, 1], P[:, 2])
            dist = np.linalg.norm(P, axis=1)
            col = shades(rng, ORANGE, rng.random(len(x))) * (0.85 + 0.15 * np.abs(np.concatenate([v, tv])))[:, None]
            col[:n][s < 0.08] *= 0.6
            fog = (1 - np.exp(-dist / 14.0))[:, None] * 0.6
            xi, yi = np.clip(x.astype(int), 0, self.W - 1), np.clip(y.astype(int), 0, self.H - 1)
            col = col * (1 - fog) + self.base[yi, xi] * fog
            dots.append((x, y, col, np.clip(sc * 0.025, 1.0, 2.6), -dist))

    def draw_specks(self, img, t):
        B = self.bub                                              # bubbles wobbling up
        f = (t * B["n"] / T_END + B["ph"]) % 1.0
        Z = 0.5 + (D - 0.6) * f
        X = B["X"] + 0.08 * np.sin(lap(t, 30) + B["w"] + 9 * f)
        Y = B["Y"] + 0.08 * np.cos(lap(t, 26) + B["w"] + 7 * f)
        x, y, sc = self.proj(X, Y, Z)
        r = np.clip(sc * B["r"], 0.8, 3.0)
        glow(img, x, y, np.array([170, 210, 220]) * 0.45, r)
        glow(img, x - 0.3 * r, y - 0.3 * r, np.array([255, 255, 255]) * 0.5, 0.8)
        N = self.snow                                             # drifting specks
        X = N["X"] + 0.15 * np.sin(lap(t, 1) + N["ph"])
        Y = N["Y"] + 0.15 * np.cos(lap(t, 2) + N["ph"])
        x, y, sc = self.proj(X, Y, N["Z"])
        on = (x > 0) & (x < self.W) & (y > 0) & (y < self.H)
        glow(img, x[on], y[on], np.array([150, 190, 180]) * (0.12 + 0.25 * N["b"][on])[:, None], np.clip(sc[on] * 0.006, 0.8, 1.6))

    # ---- the frame ---------------------------------------------------------------------------------
    def render(self, t):
        t = t % T_END
        img = self.base.copy()
        self.draw_window(img, t)
        dots, glows = [], []
        self.draw_kelp(dots, t)
        self.draw_fish(dots, glows, t)
        self.draw_seal(dots, glows, t)
        self.draw_otter(dots, t)
        self.draw_garibaldi(dots, t)
        paint(img, dots)
        for x, y, c, r in glows:
            glow(img, x, y, c, r)
        self.draw_specks(img, t)
        self.haze.apply(img, 1.0)
        return np.clip(img, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    run(Kelp, "Kelp forest", "kelp", T_END, STAGES, wrap=True)
