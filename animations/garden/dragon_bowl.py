"""
dragon_bowl.py - the Dragon Gate: a goldfish bowl at night, in dots.

The old story says a carp that swims up through the Dragon Gate and leaps the falls becomes a
dragon. Here it happens in a goldfish bowl on a table by a moonlit window. The glass turns the
window upside down in the water; the bowl gathers the moonlight into a bright knot on the table.
A moon jellyfish pulses in the water, glowing brighter as the night deepens, while a fantail
goldfish circles. The fish dives down to the little red gate standing on the gravel and swims
through it - the gate flares gold - the water spins into a whirlpool, and the fish rides it up
and leaps clean out of the bowl. In a flash it unfurls into a golden dragon. The jellyfish lets
its light go as a pearl that floats up out of the water, and the dragon chases it round and
round the bowl - in front of the glass, behind it, seen magnified through the water - swooping
over the top, until it catches the pearl, spirals down, and pours itself back into the bowl.
The pearl sinks back into the jellyfish, and a goldfish swims on as if nothing had happened.

    python dragon_bowl.py            interactive window (Space pause, R restart, S save frame, Esc quit)
    python dragon_bowl.py --png      save five stages
    python dragon_bowl.py --gif      save the whole minute as a GIF
"""
import math

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))   # the shared engine: engine/pointillism.py

from pointillism import Haze, Moon, glow, paint, run, shades, ss, stamp, warm_glow

T_END = 60.0
TAU = 2 * math.pi
STAGES = [8.0, 20.2, 27.9, 44.0, 50.3]
YW, YR = 0.42, 0.80                     # water level and rim height, in bowl radii from its centre
AW, AR = math.sqrt(1 - YW ** 2), math.sqrt(1 - YR ** 2)
BEHIND_ZOOM = 1.18                      # things just behind the bowl look magnified through it
CANDLE = np.array([1.62, -1.0, -0.30])

FISH_TOP, FISH_SIDE, FISH_BELLY = (np.array(c, np.float32) for c in ((255, 150, 40), (250, 96, 24), (255, 214, 160)))
GOLD, CRIMSON, CREAM = (np.array(c, np.float32) for c in ((238, 172, 50), (200, 46, 28), (246, 222, 168)))
FLAME = np.array([255, 110, 30], np.float32)
JELLY = np.array([140, 215, 255], np.float32)
PEARL = np.array([210, 240, 255], np.float32)
GRAVEL = [(214, 208, 194), (46, 44, 48), (196, 84, 60), (74, 112, 186), (226, 188, 92), (96, 158, 112), (170, 160, 150)]


class DragonBowl:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.S = S = min(W, H)
        self.R = 0.25 * S
        self.cx, self.cy = 0.5 * W, 0.57 * H
        self.f = 7.0
        self.ce, self.se = math.cos(0.18), math.sin(0.18)
        k = max(2, round(4 * S / 1000))
        self.haze = Haze(W, H, k)                       # light in the room
        self.wlight = Haze(W, H, k)                     # light in the water, kept inside the bowl
        v = np.arange(4096, dtype=np.float32) * (1.2 / 255.0)
        self.lut = np.clip(255.0 * v * (1 + v / 6.0) / (1 + v), 0, 255).astype(np.uint8)
        self.rad = 1.5 * S / 1000 + 0.4
        self.build_path()
        self.build_room()
        self.build_bowl()
        self.build_glass()
        self.build_creature()
        self.build_jelly()
        self.build_misc()

    # ---- the camera: a little above the bowl, looking slightly down --------------------------------
    def proj(self, X, Y, Z):
        X, Y, Z = (np.asarray(a, np.float64) for a in (X, Y, Z))
        zc = Z * self.ce - Y * self.se
        yc = Y * self.ce + Z * self.se
        s = self.f / (self.f + zc)
        return self.cx + X * self.R * s, self.cy - yc * self.R * s, s, zc

    def table_point(self, x, y):
        """Screen pixels on the table top -> where they are on the table (X, Z)."""
        A = (self.cy - y) / (self.R * self.f)
        Z = (A * (self.f + self.se) + self.ce) / (self.se - A * self.ce)
        s = self.f / (self.f + Z * self.ce + self.se)
        return (x - self.cx) / (self.R * s), Z

    # ---- the creature's path: one line through the whole minute --------------------------------------
    def swim(self, t):
        P = np.interp(t, self.Pt, self.Pv)
        return np.stack([0.42 * np.sin(P), -0.30 + 0.10 * np.sin(0.7 * P + 1.0), 0.05 - 0.26 * np.cos(P)], -1)

    def spline(self, t):
        T, K = self.knots_t, self.knots
        t = np.clip(t, T[0], T[-1])
        i = np.clip(np.searchsorted(T, t, side="right") - 1, 0, len(T) - 2)
        u = ((t - T[i]) / (T[i + 1] - T[i]))[:, None]
        p0, p1, p2, p3 = K[np.maximum(i - 1, 0)], K[i], K[i + 1], K[np.minimum(i + 2, len(K) - 1)]
        return 0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)

    def vortex(self, t):
        tau = np.clip(t - 21.4, 0, None)
        th = self.v_th0 + 1.5 * tau + 0.35 * tau ** 2
        q = ss(tau, 0, 4.6)
        r = self.v_r0 + (0.10 - self.v_r0) * q
        return np.stack([r * np.cos(th), -0.42 + 0.79 * q ** 0.8, 0.05 + 0.9 * r * np.sin(th)], -1)

    def leap(self, t):
        q = np.clip((t - 26.0) / 1.6, 0, 1)
        a, b = self.leap0, np.array([0.0, 1.40, -0.15])
        w = ss(q, 0, 1)
        return np.stack([a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * (1 - (1 - q) ** 2), a[2] + (b[2] - a[2]) * w], -1)

    def orbit(self, t):
        th = self.o_th0 + 1.15 * (t - 27.6)
        b = np.exp(-((t - 39.5) / 1.1) ** 2)                           # one swoop over the top of the bowl
        Rr = (1.48 + 0.22 * np.sin(0.5 * th)) * (1 - 0.78 * b)
        Y = 0.55 + 0.45 * np.sin(0.8 * th + 0.5) + 0.18 * np.sin(1.9 * th) + 0.8 * b
        return np.stack([Rr * np.cos(th), Y, 0.05 + 0.85 * Rr * np.sin(th)], -1)

    def dive(self, t):
        th = self.d_th + 1.4 * (t - 47.5)
        q1, q2 = ss(t, 47.5, 49.2), ss(t, 49.2, 50.9)
        Rr = self.d_R + (0.32 - self.d_R) * q1 + (0.12 - 0.32) * q2
        Y = self.d_Y + (1.30 - self.d_Y) * q1 + (-0.30 - 1.30) * q2
        return np.stack([Rr * np.cos(th), Y, 0.05 + 0.85 * Rr * np.sin(th)], -1)

    def swim2(self, t):
        P = self.p2 + 0.9 * (t - 50.9) - 0.025 * np.clip(t - 50.9, 0, None) ** 2
        return np.stack([0.40 * np.sin(P), -0.27 + 0.08 * np.sin(0.7 * P), 0.05 - 0.25 * np.cos(P)], -1)

    def path(self, t):
        t = np.atleast_1d(np.asarray(t, np.float64))
        pos = self.swim(t)
        for fn, a, b in ((self.spline, 17.4, 18.3), (self.vortex, 21.2, 21.6), (self.leap, 25.9, 26.1),
                         (self.orbit, 27.6, 29.0), (self.dive, 47.3, 47.8), (self.swim2, 50.6, 51.6)):
            w = ss(t, a, b)
            if np.any(w > 0):
                pos = pos + (fn(t) - pos) * w[:, None]
        return pos

    def build_path(self):
        tg = np.arange(-6.0, 62.0, 0.01)
        P = np.cumsum(0.55 + 0.6 * ss(tg, 6.0, 18.0)) * 0.01
        self.Pt, self.Pv = tg, P - np.interp(0.0, tg, P)
        a17 = self.swim(np.array([17.4]))[0]
        self.knots_t = np.array([17.4, 18.5, 19.3, 19.95, 20.6, 21.4])
        self.knots = np.array([a17, (-0.28, -0.47, -0.40), (-0.07, -0.60, -0.44), (0.0, -0.62, -0.10), (0.03, -0.58, 0.30),
                               (0.22, -0.42, 0.30)])
        e = self.knots[-1]
        self.v_r0 = math.hypot(e[0], (e[2] - 0.05) / 0.9)
        self.v_th0 = math.atan2((e[2] - 0.05) / 0.9, e[0])
        self.leap0 = self.vortex(np.array([26.0]))[0]
        self.o_th0 = -math.pi / 2 - 0.3
        th47 = self.o_th0 + 1.15 * (47.5 - 27.6)
        o47 = self.orbit(np.array([47.5]))[0]
        self.d_th, self.d_R, self.d_Y = th47, math.hypot(o47[0], (o47[2] - 0.05) / 0.85), o47[1]
        d = self.dive(np.array([50.9]))[0]
        self.p2 = math.atan2(d[0] / 0.40, -(d[2] - 0.05) / 0.25)
        tt = np.arange(48.5, 51.5, 0.01)                                # when the dragon breaks the surface
        yy = self.path(tt)[:, 1]
        self.t_dive = float(tt[np.argmax(yy < YW)])
        self.J, self.dt = 220, 0.016

    def morph(self, t):
        return float(ss(t, 27.4, 30.0) * (1 - ss(t, 50.0, 52.6)))

    def spine(self, t, L, n=150):
        ts = t - np.arange(self.J) * self.dt
        P = self.path(ts)
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        tgt = np.minimum(np.linspace(0, L, n), s[-1])
        return np.stack([np.interp(tgt, s, P[:, k]) for k in range(3)], -1)

    # ---- the room: wall, window, table, candle -----------------------------------------------------
    def build_room(self):
        W, H, S, R = self.W, self.H, self.S, self.R
        rng = np.random.default_rng(3)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        img = np.zeros((H, W, 3), np.float32)
        img[:] = np.array([30, 27, 34], np.float32) * (0.85 + 0.25 * (yy / H))[..., None]
        # the window: a night sky, the moon, hills and a pine bough, in a wooden frame
        x0, x1, y0, y1 = 0.05 * W, 0.47 * W, 0.07 * H, 0.47 * H
        self.win = (x0, x1, y0, y1)
        inwin = (xx > x0) & (xx < x1) & (yy > y0) & (yy < y1)
        g = np.clip((yy - y0) / (y1 - y0), 0, 1)
        skyc = np.array([10, 16, 46], np.float32) * (1 - g[..., None]) + np.array([44, 58, 104], np.float32) * g[..., None]
        img[inwin] = skyc[inwin]
        n = int(900 * W * H / 1e6)
        sx, sy = rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)
        glow(img, sx, sy, np.array([220, 225, 255])[None] * (rng.random(n) ** 3 * 0.9 + 0.1)[:, None] * 0.8, 1.0)
        for cxf, cyf, sz in ((0.30, 0.13, 0.10), (0.38, 0.30, 0.07), (0.13, 0.36, 0.09)):   # thin clouds
            m = int(1500 * sz / 0.1)
            cxp, cyp = cxf * W, cyf * H
            px = cxp + rng.normal(0, 1, m) * sz * W * 0.5
            py = cyp + rng.normal(0, 1, m) * sz * H * 0.08
            ok = (px > x0) & (px < x1) & (py > y0) & (py < y1)
            glow(img, px[ok], py[ok], np.array([60, 70, 110]) * 0.10, 2.0)
        mask = np.zeros((H, W), np.float32)
        mask[inwin] = 1
        Moon(np.random.default_rng(4), 0.045 * S, 3.2).draw(img, 0.20 * W, 0.19 * H, np.array([250, 246, 228]),
                                                             np.array([50, 56, 80]), mask, mask)
        u = (xx - x0) / (x1 - x0)
        ridge = y1 - (0.07 + 0.04 * np.sin(u * 7 + 1) + 0.02 * np.sin(u * 17)) * H
        hills = inwin & (yy > ridge)
        img[hills] = np.array([16, 22, 38], np.float32)
        q = np.linspace(0, 1, 400)                                     # a pine bough across the top corner
        bx, by = x0 + (0.02 + 0.25 * q) * W, y0 + (0.05 + 0.06 * q ** 1.4) * H
        stamp(img, bx, by, np.array([12, 14, 20], np.float32), 2.5)
        for k in range(26):
            px, py = float(np.interp(k / 26, q, bx)), float(np.interp(k / 26, q, by))
            for side in (-1, 1):
                a = rng.uniform(0.3, 1.1) * side - math.pi / 2 * 0.15
                L = rng.uniform(0.02, 0.05) * S
                qq = np.linspace(0, 1, 30)
                nx, ny = px + np.cos(a + math.pi / 2) * L * qq, py + np.sin(a + math.pi / 2) * L * qq * side
                stamp(img, nx, ny, np.array([12, 16, 22], np.float32), 1.5)
                for j in range(8):
                    tq = np.linspace(0, 1, 6)
                    jx, jy = nx[j * 3] + rng.normal(0, 1) * 0.012 * S * tq, ny[j * 3] + 0.012 * S * tq
                    stamp(img, jx, jy, np.array([14, 20, 24], np.float32), 1.0)
        wood = np.array([46, 34, 28], np.float32)
        fw = 0.013 * W
        frame = ((np.abs(xx - x0) < fw) | (np.abs(xx - x1) < fw) | (np.abs(yy - y0) < fw) | (np.abs(yy - y1) < fw)) & \
            (xx > x0 - fw) & (xx < x1 + fw) & (yy > y0 - fw) & (yy < y1 + fw)
        frame |= inwin & ((np.abs(xx - (x0 + x1) / 2) < fw * 0.5) | (np.abs(yy - (y0 + y1) / 2) < fw * 0.5))
        img[frame] = wood * (0.8 + 0.3 * rng.random((int(frame.sum()), 1)))
        sill = (xx > x0 - 2.2 * fw) & (xx < x1 + 2.2 * fw) & (yy > y1 + fw) & (yy < y1 + 2.6 * fw)
        img[sill] = np.array([66, 50, 40], np.float32) * (1 + 0.3 * (yy[sill] < y1 + 1.4 * fw))[:, None]
        # the table top, seen in perspective
        X, Z = self.table_point(xx, yy)
        self.table_y = float(self.proj(0, -1.0, 3.0)[1])
        table = (yy > self.table_y) & (Z > -6)
        plank = np.floor((Z + 10) / 0.42)
        seam = ((Z + 10) % 0.42) < 0.02 * (1 + 0.5 * np.maximum(Z, 0))
        grain = 0.85 + 0.15 * np.sin(X * 9 + np.sin(X * 1.7 + plank * 3.1) * 2 + plank * 1.3)
        wc = np.array([64, 40, 28], np.float32) * grain[..., None] * (0.85 + 0.3 * ((np.sin(plank * 12.9898) * 43758.5) % 1))[..., None]
        wc[seam] *= 0.45
        img = np.where(table[..., None], wc * 0.55, img).astype(np.float32)
        # moonlight through the window falls on the table; the bowl throws a shadow with a bright knot inside it
        d = np.array([-2.2, 3.0, 4.0])
        d = d / np.linalg.norm(d)
        Pw = np.stack([X, np.full_like(X, -1.0), Z], -1)
        tt = (4.0 - Z) / d[2]
        Xw, Yw = X + d[0] * tt, -1.0 + d[1] * tt
        lit = ss(Xw, -3.4, -3.0) * (1 - ss(Xw, -1.0, -0.6)) * ss(Yw, 0.5, 0.9) * (1 - ss(Yw, 3.0, 3.4))
        lit = lit * (1 - 0.55 * np.exp(-((Xw + 2.0) / 0.09) ** 2)) * (1 - 0.55 * np.exp(-((Yw - 2.0) / 0.09) ** 2))
        rel = -Pw
        proj_ = rel @ d
        dist = np.linalg.norm(rel - proj_[..., None] * d, axis=-1)
        through = (proj_ > 0) & (dist < 1.0)
        caustic = np.where(through, 0.35 + 4.0 * np.exp(-(dist / 0.18) ** 2) + 0.8 * np.exp(-((dist - 0.93) / 0.06) ** 2), 1.0)
        moon_on_table = (lit * caustic * table).astype(np.float32)
        tint = np.where(through[..., None], np.array([0.55, 0.85, 1.0], np.float32), np.array([1.0, 1.0, 1.0], np.float32))
        img += (np.array([40, 48, 74], np.float32) * moon_on_table[..., None] * tint)
        cmask = table & (lit > 0.5) & through & (dist < 0.32)
        self.caustic_pts = np.stack(np.nonzero(cmask)[::-1], -1).astype(np.float32)[::7]
        # the line where table meets wall, and the candle
        edge = np.abs(yy - self.table_y) < 1.5
        img[edge] = np.array([22, 18, 20], np.float32)
        cx_, cy_, cs, _ = self.proj(*CANDLE)
        cw, ch = 0.10 * R * cs, 0.62 * R * cs
        n = 3000
        a, rr = rng.uniform(0, TAU, n), np.sqrt(rng.random(n))
        stamp(img, cx_ + np.cos(a) * rr * cw * 2.2, cy_ + np.sin(a) * rr * cw * 0.6, np.array([150, 110, 50], np.float32) *
              (0.6 + 0.5 * rr)[:, None], 1.5)
        n = 5000
        u, v = rng.random(n), rng.uniform(-1, 1, n)
        px, py = cx_ + v * cw, cy_ - u * ch + (1 - v * v) * 0.04 * cw
        col = np.array([236, 222, 196], np.float32) * (0.55 + 0.45 * u ** 2)[:, None] * (0.75 + 0.25 * (1 - np.abs(v)))[:, None]
        stamp(img, px, py, col, 1.5)
        for k in range(4):                                             # wax drips
            vx = rng.uniform(-0.8, 0.8)
            q = np.linspace(0, rng.uniform(0.15, 0.4), 30)
            stamp(img, np.full(30, cx_ + vx * cw), cy_ - ch + q * ch, np.array([246, 236, 214], np.float32), 1.5)
        self.flame = (cx_, cy_ - ch - 0.02 * R, cs)
        self.candle_box = (cx_ - cw * 1.05, cx_ + cw * 1.05, cy_ - ch, cy_, float(self.proj(*CANDLE)[3]))
        warm_glow(img, self.flame[0], self.flame[1], 0.35 * S, (60, 34, 14), 1.0)
        # a faint beam of moonlight from the window down to the bowl
        ax_, ay_, bx_, by_ = 0.27 * W, 0.27 * H, 0.52 * W, 0.86 * H
        vx_, vy_ = bx_ - ax_, by_ - ay_
        ln = math.hypot(vx_, vy_)
        along = np.clip(((xx - ax_) * vx_ + (yy - ay_) * vy_) / ln ** 2, 0, 1)
        dd = np.abs((xx - ax_) * vy_ - (yy - ay_) * vx_) / ln
        wb = (0.05 + 0.10 * along) * S
        self.beam = (np.exp(-(dd / wb) ** 2) * (along > 0) * (1 - along) ** 0.5).astype(np.float32)
        img += np.array([16, 20, 32], np.float32) * self.beam[..., None] * 0.9
        n = int(W * H * 0.35)                                           # stipple everything
        x, y = rng.uniform(0, W, n), rng.uniform(0, H, n)
        stamp(img, x, y, img[y.astype(int), x.astype(int)] * (0.84 + 0.32 * rng.random((n, 1))), rng.choice([1.0, 1.5, 2.0], n))
        self.bg = img

    # ---- the bowl: what you see through it ---------------------------------------------------------------
    def ellipse(self, Yl, a):
        _, yc, s, _ = self.proj(0, Yl, 0)
        _, yf, _, _ = self.proj(0, Yl, -a)
        _, yb, _, _ = self.proj(0, Yl, a)
        return float(yc), a * self.R * float(s), float(yf - yb) / 2

    def build_bowl(self):
        W, H, R, cx, cy = self.W, self.H, self.R, self.cx, self.cy
        rng = np.random.default_rng(7)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        dx, dy = xx - cx, yy - cy
        r = np.hypot(dx, dy)
        ry, ra, rb = self.ellipse(YR, AR)
        wy, wa, wb = self.ellipse(YW, AW)
        self.rim_e, self.water_e = (ry, ra, rb), (wy, wa, wb)
        inside_rx = np.abs(dx) < ra
        rim_back = ry - rb * np.sqrt(np.clip(1 - (dx / ra) ** 2, 0, 1))
        sil = (r < R) & np.where(inside_rx, yy > rim_back, yy > ry)
        inside_wx = np.abs(dx) < wa
        wfront = wy + wb * np.sqrt(np.clip(1 - (dx / wa) ** 2, 0, 1))
        surf = sil & inside_wx & (((dx / wa) ** 2 + ((yy - wy) / wb) ** 2) < 1)
        water = sil & ~surf & np.where(inside_wx, yy > wfront, yy > wy)
        air = sil & ~surf & ~water
        self.sil, self.water_mask = sil, (water | surf).astype(np.float32)
        bg = self.bg
        img = bg.copy()

        def sample(sx, sy):
            return bg[np.clip(sy.astype(int), 0, H - 1), np.clip(sx.astype(int), 0, W - 1)]

        k = 1.65                                                          # the water inverts the far room
        wcol = sample(cx - dx[water] * k, cy - dy[water] * k)
        edge = 1 - 0.55 * (r[water] / R) ** 6
        img[water] = (wcol * np.array([0.58, 0.78, 0.86], np.float32) + np.array([4, 14, 22], np.float32)) * edge[:, None]
        acol = sample(cx + dx[air] * 0.97, cy + dy[air] * 0.97)
        img[air] = acol * np.array([0.86, 0.92, 0.96], np.float32) * (1 - 0.35 * (r[air] / R) ** 6)[:, None]
        scol = sample(cx + dx[surf] * 0.9, 2 * wy - yy[surf])
        img[surf] = scol * np.array([0.35, 0.45, 0.6], np.float32) + np.array([14, 22, 40], np.float32)
        # gravel on the bottom, and the glass foot
        n = 3200
        a, rr = rng.uniform(0, TAU, n), 0.58 * np.sqrt(rng.random(n))
        top = np.stack([rr * np.cos(a), -0.80 + 0.02 * rng.normal(0, 1, n), 0.0 + rr * np.sin(a)], -1)
        m = 1600
        Yv = rng.uniform(-0.97, -0.80, m)
        a2 = rng.uniform(math.pi, TAU, m)
        rw = np.sqrt(1 - Yv ** 2) * 0.96
        wall = np.stack([rw * np.cos(a2), Yv, rw * np.sin(a2)], -1)
        P = np.vstack([top, wall])
        x, y, s, zc = self.proj(P[:, 0], P[:, 1], P[:, 2])
        order = np.argsort(-zc)
        col = shades(rng, GRAVEL, rng.random(len(P)), 2.5, 0.1) * np.array([0.42, 0.52, 0.58], np.float32)
        for j in range(3):
            ox, oy = rng.normal(0, 1.2, len(P)), rng.normal(0, 0.8, len(P))
            stamp(img, (x + ox)[order], (y + oy)[order], (col * (0.85 + 0.15 * j))[order], (rng.choice([1.5, 2.0, 2.5], len(P)))[order])
        glow(img, x[order][::3] - 1, y[order][::3] - 1, np.array([30, 36, 46], np.float32), 1.0)
        fa = np.linspace(0, TAU, 300)
        fx, fy, _, _ = self.proj(0.26 * np.cos(fa), np.full(300, -0.985), 0.26 * np.sin(fa))
        glow(img, fx, fy, np.array([40, 50, 64], np.float32), 1.5)
        self.base = img.astype(np.float32)

    def build_glass(self):
        """Highlights on the glass, added over whatever is inside the bowl."""
        R, cx, cy = self.R, self.cx, self.cy
        rng = np.random.default_rng(9)
        xs, ys, cs, rs = [], [], [], []
        ry, ra, rb = self.rim_e
        a = np.linspace(0, TAU, 1400, endpoint=False)
        ex, ey = cx + np.cos(a) * R * 0.996, cy + np.sin(a) * R * 0.996
        ok = ey > ry - rb * 0.3
        moon = np.clip(np.cos(a - math.radians(225)), 0, 1) ** 2
        cand = np.clip(np.cos(a - math.radians(355)), 0, 1) ** 4
        k = 0.12 + 0.75 * moon + 0.45 * cand
        col = np.array([120, 140, 190])[None] * moon[:, None] + np.array([200, 140, 80])[None] * cand[:, None] + 24
        for rr, w in ((0.996, 1.0), (0.972, 0.35)):
            xs.append(cx + np.cos(a[ok]) * R * rr)
            ys.append(cy + np.sin(a[ok]) * R * rr)
            cs.append(col[ok] * (k[ok] * w * 0.8)[:, None])
            rs.append(np.full(int(ok.sum()), 1.2))
        n = 2600                                                             # the window, reflected small and curved
        u, v = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
        keep = (np.abs(u) > 0.10) & (np.abs(v) > 0.10)
        u, v = u[keep], v[keep]
        wxp = cx + R * (-0.46 + 0.12 * u * (1 - 0.18 * v) + 0.03 * v)
        wyp = cy + R * (-0.40 + 0.15 * v - 0.02 * u * u)
        fade = (1 - np.abs(u) ** 4) * (1 - np.abs(v) ** 4)
        xs.append(wxp)
        ys.append(wyp)
        cs.append(np.array([80, 92, 126])[None] * fade[:, None] * 0.75)
        rs.append(np.full(len(u), 1.0))
        xs.append(np.array([cx - 0.50 * R]))
        ys.append(np.array([cy - 0.47 * R]))
        cs.append(np.array([[230, 235, 255]]) * 1.6)
        rs.append(np.array([2.5]))
        n = 300                                                              # the candle, reflected on the right
        q = rng.random(n)
        xs.append(cx + R * (0.66 + 0.02 * rng.normal(0, 1, n)))
        ys.append(cy + R * (-0.18 + 0.30 * q))
        cs.append(np.array([255, 190, 110])[None] * ((1 - q) ** 2 * 0.6)[:, None])
        rs.append(np.full(n, 1.2))
        for Yl, aa, front, back in ((0.84, 0.64, 0.9, 0.35), (YR, AR, 0.55, 0.22)):   # the lip of the rim
            th = np.linspace(0, TAU, 700, endpoint=False)
            lx, ly, _, zc = self.proj(aa * np.cos(th), np.full(700, Yl), aa * np.sin(th))
            w = np.where(zc < 0, front, back) * (0.6 + 0.4 * np.clip(-np.cos(th - 0.6), 0, 1))
            xs.append(lx)
            ys.append(ly)
            cs.append(np.array([150, 165, 200])[None] * w[:, None] * 0.55)
            rs.append(np.full(700, 1.2))
        self.glass = tuple(np.concatenate(z).astype(np.float32) for z in (xs, ys, cs, rs))

    # ---- the goldfish that becomes a dragon --------------------------------------------------------
    def build_creature(self):
        rng = np.random.default_rng(12)
        n = 16000
        self.bu, self.bv = rng.random(n), rng.uniform(-1, 1, n)
        self.bt = rng.random(n)
        parts = []                                                           # the dragon's head, in head lengths

        def blob(n_, ca, cb, ra, rb, col, jaw=0):
            r, th = np.sqrt(rng.random(n_)), rng.uniform(0, TAU, n_)
            parts.append((ca + ra * r * np.cos(th), cb + rb * r * np.sin(th), np.broadcast_to(np.asarray(col, np.float32), (n_, 3)),
                          np.full(n_, jaw)))

        blob(900, 0.26, 0.07, 0.29, 0.21, GOLD)                              # skull
        n_ = 700                                                             # the long upper jaw and snout
        a = 0.40 + 0.58 * rng.random(n_)
        top = 0.16 - 0.10 * (a - 0.40) / 0.58 + 0.05 * np.exp(-((a - 0.92) / 0.05) ** 2)
        b = rng.random(n_) * top
        parts.append((a, b, np.broadcast_to((GOLD * 0.55 + np.array([255, 222, 140]) * 0.45).astype(np.float32), (n_, 3)), np.zeros(n_)))
        blob(160, 0.96, 0.14, 0.06, 0.06, np.array([252, 212, 120]))         # nose, curling up
        blob(40, 0.99, 0.12, 0.018, 0.018, np.array([60, 20, 10]))           # nostril
        blob(200, 0.46, 0.22, 0.16, 0.05, CRIMSON)                           # brow ridge
        blob(220, 0.10, 0.15, 0.14, 0.10, CRIMSON)                           # cheek frill
        n_ = 300                                                             # the inside of the mouth
        a = 0.30 + 0.58 * rng.random(n_)
        parts.append((a, -0.01 - rng.random(n_) * 0.05, np.broadcast_to(np.array([120, 20, 24], np.float32), (n_, 3)), np.ones(n_)))
        n_ = 620                                                             # lower jaw
        a = 0.25 + 0.66 * rng.random(n_)
        bot = -0.02 - 0.12 * (1 - (a - 0.25) / 0.66) ** 0.7
        b = -0.02 + rng.random(n_) * (bot + 0.02)
        parts.append((a, b, np.broadcast_to(CREAM, (n_, 3)), np.ones(n_)))
        for ta in np.arange(0.48, 0.93, 0.06):                               # teeth, top and bottom
            blob(10, ta, -0.012, 0.012, 0.014, np.array([252, 250, 240]))
            blob(8, ta + 0.03, -0.035, 0.010, 0.012, np.array([252, 250, 240]), 1)
        x, y, col, jaw = (np.concatenate(z) for z in zip(*parts))
        self.head = dict(a=x, b=y, col=col, jaw=jaw.astype(bool))
        k = 40                                                               # sparks the dragon sheds
        self.spark = dict(off=rng.random(260) * 1.3, lag=rng.uniform(0.05, 1.3, 260), v=rng.normal(0, 0.35, (260, 3)),
                          ph=rng.uniform(0, TAU, 260))
        self.burst = rng.normal(0, 1, (280, 3))
        self.burst /= np.linalg.norm(self.burst, axis=1, keepdims=True)
        del k

    def classify(self, P):
        """For points in the world: 0 outside and in front, 1 in the water, 2 inside the glass above the water,
        3 behind the bowl (seen through it)."""
        X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
        x, y, s, zc = self.proj(X, Y, Z)
        r3 = np.sqrt(X * X + Y * Y + Z * Z)
        inside = (r3 < 0.985) & (Y < YR)
        xi = np.clip(x.astype(int), 0, self.W - 1)
        yi = np.clip(y.astype(int), 0, self.H - 1)
        sil = self.sil[yi, xi]
        cls = np.where(inside & (Y < YW), 1, np.where(inside, 2, np.where(sil & (zc > 0), 3, 0)))
        return cls, zc

    def emit(self, x, y, cls, zc, col, rad, kind, bias=0.0):
        """Queue dots or light, seen through the glass and the water as they should be."""
        x = np.asarray(x, np.float64).copy()
        y = np.asarray(y, np.float64).copy()
        col = np.broadcast_to(np.asarray(col, np.float32), (len(x), 3)).copy()
        rad = np.broadcast_to(np.asarray(rad, np.float32), (len(x),))
        b3 = cls == 3
        if b3.any():
            x[b3] = self.cx + (x[b3] - self.cx) * BEHIND_ZOOM
            y[b3] = self.cy + (y[b3] - self.cy) * BEHIND_ZOOM
            col[b3] *= np.array([0.50, 0.68, 0.74], np.float32)
            keep = ~b3 | (np.hypot(x - self.cx, y - self.cy) < 0.975 * self.R)
            x, y, col, rad, cls, zc = x[keep], y[keep], col[keep], rad[keep], cls[keep], np.asarray(zc)[keep]
        w = cls == 1
        col[w] *= np.array([0.76, 0.90, 0.96], np.float32)
        col[cls == 2] *= 0.94
        x0c, x1c, y0c, y1c, zcc = self.candle_box                         # behind the candle: hidden
        hid = (cls == 0) & (x > x0c) & (x < x1c) & (y > y0c) & (y < y1c) & (np.asarray(zc) > zcc)
        if hid.any():
            keep = ~hid
            x, y, col, rad, cls, zc = x[keep], y[keep], col[keep], rad[keep], cls[keep], np.asarray(zc)[keep]
        inner = cls > 0
        for sel, A, G in ((inner, self.A, self.gA), (~inner, self.B, self.gB)):
            if sel.any():
                if kind == "dot":
                    A.append((x[sel], y[sel], col[sel], rad[sel], -np.asarray(zc)[sel] + bias))
                else:
                    G.append((x[sel], y[sel], col[sel], rad[sel]))

    def draw_creature(self, t):
        m = self.morph(t)
        L = 0.30 + 2.25 * m
        sp = self.spine(t, L)
        n = len(sp)
        u_sp = np.linspace(0, 1, n)
        tang = np.gradient(sp, axis=0)                                       # a serpentine wave for the dragon
        side = np.cross(tang, [0.0, 1.0, 0.0])
        side /= np.linalg.norm(side, axis=1, keepdims=True) + 1e-9
        sp = sp + side * (0.06 * m * np.sin(TAU * 2.2 * u_sp - 6.5 * t) * np.sin(np.pi * u_sp))[:, None]
        cls_sp, zc_sp = self.classify(sp)
        x, y, s, zc = self.proj(sp[:, 0], sp[:, 1], sp[:, 2])
        gx, gy = np.gradient(x), np.gradient(y)
        ln = np.hypot(gx, gy) + 1e-6
        fx, fy = -gx / ln, -gy / ln                                         # forward (toward the head) on screen
        nx, ny = -fy, fx
        flip = ny > 0
        nx, ny = np.where(flip, -nx, nx), np.where(flip, -ny, ny)            # the creature's back is always up
        R = self.R
        ht = 0.092 * np.sin(np.pi * np.clip(u_sp / 0.9, 0, 1)) ** 0.6 * (1 - 0.25 * u_sp)
        dr = 0.092 * (1 - 0.62 * u_sp ** 1.1) + 0.012
        top = ht + (dr - ht) * m
        bot = ht * 1.15 + (dr - ht * 1.15) * m
        area = float(np.sum((top + bot) * R * s * np.hypot(gx, gy)))
        cnt = int(np.clip(area / (1.15 * self.rad ** 2), 300, len(self.bu)))
        u, v, tn = self.bu[:cnt], self.bv[:cnt], self.bt[:cnt]
        idx = np.clip((u * (n - 1)).astype(int), 0, n - 1)
        ui = lambda a: np.interp(u, u_sp, a)
        wtop, wbot = ui(top) * R * ui(s), ui(bot) * R * ui(s)
        off = np.where(v > 0, v * wtop, v * wbot)
        px, py = ui(x) + ui(nx) * off, ui(y) + ui(ny) * off
        col = self.body_colours(u, v, tn, m, t, L)
        self.emit(px, py, cls_sp[idx], zc_sp[idx], col, self.rad, "dot")
        if m < 0.99:
            self.fish_features(t, 1 - m, x, y, s, fx, fy, nx, ny, top, bot, cls_sp, zc_sp, u_sp)
        if m > 0.01:
            self.dragon_features(t, m, sp, x, y, s, fx, fy, nx, ny, top, bot, cls_sp, zc_sp, u_sp)
        return sp, m

    def body_colours(self, u, v, tn, m, t, L):
        up = np.clip(v, 0, 1)[:, None]
        dn = np.clip(-v, 0, 1)[:, None]
        fish = FISH_SIDE + (FISH_TOP - FISH_SIDE) * up ** 0.8 + (FISH_BELLY - FISH_SIDE) * dn ** 1.5
        fish = fish * (1 + 0.10 * (np.sin(u * L * 140 + np.floor(v * 6) * 1.7) > 0.4))[:, None]
        fish[np.abs(u - 0.22) < 0.012] *= 0.7
        cu = u * L / 0.034
        cv = (v + 1) * 2.6 + 0.5 * np.floor(cu)
        scale_edge = ((cu % 1) < 0.2) | (np.abs((cv % 1) - 0.5) > 0.40)
        base = GOLD + (np.array([214, 96, 30], np.float32) - GOLD) * (u ** 1.3)[:, None]
        back = v > 0.62
        drag = np.where(back[:, None], CRIMSON * (0.9 + 0.2 * tn[:, None]), base)
        belly = v < -0.45
        plates = ((u * L / 0.03) % 1) < 0.16
        drag = np.where(belly[:, None], CREAM * np.where(plates, 0.72, 1.0)[:, None], drag)
        drag = drag * np.where(scale_edge & ~belly, 0.84, 1.0)[:, None]
        shine = (np.sin(u * L * 60 - 3.0 * t + v * 4) > 0.93) & ~belly
        drag = drag + shine[:, None] * np.array([90, 70, 30], np.float32)
        col = fish + (drag - fish) * m
        light = 0.50 + 0.38 * (v + 1) / 2
        col = col * light[:, None] + np.array([40, 46, 70], np.float32) * (v > 0.85)[:, None]
        return col

    def fish_features(self, t, k, x, y, s, fx, fy, nx, ny, top, bot, cls, zc, u_sp):
        R = self.R
        n = len(x)
        at = lambda uu: int(np.clip(round(uu * (n - 1)), 0, n - 1))
        i = at(0.10)                                                         # the eye
        ex, ey = x[i] + nx[i] * top[i] * R * s[i] * 0.30, y[i] + ny[i] * top[i] * R * s[i] * 0.30
        er = 0.020 * R * s[i] * k
        rng = np.random.default_rng(1)
        a, rr = rng.uniform(0, TAU, 40), np.sqrt(rng.random(40))
        self.emit(ex + np.cos(a) * rr * er, ey + np.sin(a) * rr * er, np.full(40, cls[i]), np.full(40, zc[i]),
                  np.array([14, 10, 10]), 1.2, "dot", 0.01)
        self.emit(np.array([ex - er * 0.3]), np.array([ey - er * 0.3]), cls[i:i + 1], zc[i:i + 1], np.array([255, 255, 255]) * 0.8 * k,
                  1.0, "glow")
        it = n - 1                                                           # the fantail
        bx, by = -fx[it], -fy[it]
        rays = []
        for lobe in (-1, 1):
            for j in range(14):
                ang = lobe * (0.35 + 0.45 * j / 13)
                ca, sa = math.cos(ang), math.sin(ang)
                dxr = bx * ca - (-nx[it]) * sa * 0 + nx[it] * sa
                dyr = by * ca + ny[it] * sa
                q = np.linspace(0.05, 1, 12)
                Lr = 0.20 * R * s[it] * k * (0.75 + 0.25 * math.sin(j * 0.9)) * (1 + 0.08 * math.sin(7 * t + j))
                wave = 0.025 * R * s[it] * np.sin(6 * q - 7.5 * t + lobe) * q
                rays.append((x[it] + dxr * Lr * q - dyr * wave, y[it] + dyr * Lr * q + dxr * wave, q))
        rx = np.concatenate([r[0] for r in rays])
        ry = np.concatenate([r[1] for r in rays])
        rq = np.concatenate([r[2] for r in rays])
        self.emit(rx, ry, np.full(len(rx), cls[it]), np.full(len(rx), zc[it]),
                  np.array([255, 120, 40])[None] * (0.42 * k * (1 - 0.5 * rq))[:, None], 1.4, "glow")
        fins = []                                                            # the dorsal fin and the long belly fins
        for uu, sgn, ln, nr in ((0.30, 1, 0.10, 12), (0.45, -1, 0.13, 6), (0.26, -1, 0.06, 5)):
            for j in range(nr):
                ii = at(uu + 0.25 * j / max(nr, 1) * (1 if sgn > 0 else 0.4))
                q = np.linspace(0.05, 1, 9)
                h = (top if sgn > 0 else bot)[ii] * R * s[ii]
                Lr = ln * R * s[ii] * k * math.sin(math.pi * (j + 0.5) / nr) ** 0.5
                bx0, by0 = x[ii] + nx[ii] * h * sgn, y[ii] + ny[ii] * h * sgn
                sway = 0.3 * math.sin(4 * t + j * 0.6)
                dxr, dyr = nx[ii] * sgn - fx[ii] * (0.9 + sway), ny[ii] * sgn - fy[ii] * (0.9 + sway)
                fins.append((bx0 + dxr * Lr * q, by0 + dyr * Lr * q, np.full(9, ii)))
        fx_ = np.concatenate([f[0] for f in fins])
        fy_ = np.concatenate([f[1] for f in fins])
        fi = np.concatenate([f[2] for f in fins]).astype(int)
        self.emit(fx_, fy_, cls[fi], zc[fi], np.array([255, 130, 50]) * 0.34 * k, 1.3, "glow")

    def dragon_features(self, t, m, sp, x, y, s, fx, fy, nx, ny, top, bot, cls, zc, u_sp):
        R, n = self.R, len(x)
        at = lambda uu: int(np.clip(round(uu * (n - 1)), 0, n - 1))
        # the head, ahead of the neck
        j = at(0.05)
        hfx, hfy = x[0] - x[j], y[0] - y[j]
        hl = math.hypot(hfx, hfy)
        d3 = float(np.linalg.norm(sp[0] - sp[j])) * R * s[0] + 1e-6
        ff = float(np.clip(hl / d3, 0.35, 1.0))
        if hl < 1e-3:
            hfx, hfy = fx[0], fy[0]
            hl = 1.0
        hfx, hfy = hfx / hl, hfy / hl
        hux, huy = -hfy, hfx
        if huy > 0:
            hux, huy = -hux, -huy
        Lh = 0.52 * R * s[0] * m
        Hd = self.head
        a, b = Hd["a"].copy(), Hd["b"].copy()
        jaw = 0.14 + 0.10 * math.sin(1.3 * t) + 0.35 * math.exp(-((t - 47.0) / 0.5) ** 2)
        ca, sa = math.cos(-jaw), math.sin(-jaw)
        ja, jb = a[Hd["jaw"]] - 0.25, b[Hd["jaw"]] + 0.02
        a[Hd["jaw"]], b[Hd["jaw"]] = 0.25 + ja * ca - jb * sa, -0.02 + ja * sa + jb * ca
        hx = x[0] + hfx * a * Lh * ff + hux * b * Lh
        hy = y[0] + hfy * a * Lh * ff + huy * b * Lh
        hcol = Hd["col"] * (0.55 + 0.35 * np.clip(b * 3 + 0.5, 0, 1))[:, None]
        c0, z0 = cls[:1], zc[:1]
        self.emit(hx, hy, np.repeat(c0, len(hx)), np.repeat(z0, len(hx)), hcol, self.rad, "dot", 0.02)
        P = lambda aa, bb: (x[0] + hfx * aa * Lh * ff + hux * bb * Lh, y[0] + hfy * aa * Lh * ff + huy * bb * Lh)
        ex, ey = P(0.50, 0.13)                                               # the eye, glowing
        a8 = np.linspace(0, TAU, 12, endpoint=False)
        er = 0.035 * Lh
        self.emit(ex + np.cos(a8) * er, ey + np.sin(a8) * er * 0.8, np.repeat(c0, 12), np.repeat(z0, 12),
                  np.array([255, 200, 60]) * 1.4 * m, 1.6, "glow")
        self.emit(np.array([ex]), np.array([ey]), c0, z0, np.array([255, 230, 120]) * 2.4 * m, 2.2, "glow")
        self.emit(np.array([ex, ex]), np.array([ey - 1, ey + 1]), np.repeat(c0, 2), np.repeat(z0, 2), np.array([30, 10, 0]), 1.0, "dot",
                  0.03)
        q = np.linspace(0, 1, 34)                                            # horns
        hxs, hys, hcs = [], [], []
        for ba, bb, shade in ((0.30, 0.24, 0.65), (0.22, 0.26, 1.0)):
            aa, bb_ = ba - 0.48 * q, bb + 0.30 * q - 0.06 * q * q
            px_, py_ = P(aa, bb_)
            hxs.append(px_)
            hys.append(py_)
            hcs.append(np.broadcast_to(np.array([236, 226, 196]) * shade, (34, 3)))
            q2 = np.linspace(0, 1, 12)
            pa, pb = ba - 0.48 * 0.5 - 0.10 * q2, bb + 0.30 * 0.5 - 0.015 + 0.16 * q2
            px_, py_ = P(pa, pb)
            hxs.append(px_)
            hys.append(py_)
            hcs.append(np.broadcast_to(np.array([236, 226, 196]) * shade, (12, 3)))
        hxs, hys, hcs = np.concatenate(hxs), np.concatenate(hys), np.concatenate(hcs)
        self.emit(hxs, hys, np.repeat(c0, len(hxs)), np.repeat(z0, len(hxs)), hcs * m, max(1.0, 2.2 * self.S / 1000), "dot", 0.015)
        wx, wy = [], []                                                      # whiskers streaming back
        for k, (sa_, sb_) in enumerate(((0.98, 0.04), (0.95, -0.03))):
            q = np.linspace(0, 1, 46)
            aa = sa_ - 1.9 * q
            bb_ = sb_ - 0.04 * q + 0.22 * q * np.sin(4.5 * q - 5.0 * t + k * 1.7)
            px_, py_ = P(aa, bb_)
            wx.append(px_)
            wy.append(py_)
        wx, wy = np.concatenate(wx), np.concatenate(wy)
        qq = np.tile(np.linspace(0, 1, 46), 2)
        self.emit(wx, wy, np.repeat(c0, len(wx)), np.repeat(z0, len(wx)), np.array([255, 226, 150])[None] * (0.75 * m * (1 - 0.6 * qq))[:, None],
                  1.0, "glow")
        mx, my, mc, mi = [], [], [], []                                      # the flaming mane: head, neck and back
        rng = np.random.default_rng(int(t * 24))
        for k in range(16):
            aa0, bb0 = 0.28 - 0.30 * k / 15, 0.24 - 0.10 * k / 15
            q = np.linspace(0, 1, 10)
            Ls = (0.30 + 0.15 * math.sin(11 * t + k * 1.3)) * Lh / (0.40 * R * s[0] + 1e-6) * 0.40 * R * s[0] / max(Lh, 1e-6)
            aa = aa0 - 0.45 * q * Ls
            bb_ = bb0 + 0.28 * q * Ls + 0.04 * np.sin(8 * q - 9 * t + k)
            px_, py_ = P(aa, bb_)
            mx.append(px_)
            my.append(py_)
            mc.append(FLAME[None] * (1 - q)[:, None] + np.array([255, 210, 90])[None] * q[:, None])
            mi.append(np.zeros(10, int))
        for k in range(22):
            uu = 0.01 + 0.36 * k / 21
            ii = at(uu)
            h = top[ii] * R * s[ii]
            q = np.linspace(0, 1, 16)
            Ls = (0.11 + 0.04 * math.sin(9 * t + k * 1.7)) * R * s[ii] * m * (1 - 0.6 * uu)
            wave = 0.25 * np.sin(5 * q - 8 * t + k)
            dxr, dyr = nx[ii] * 0.7 - fx[ii] * 0.8, ny[ii] * 0.7 - fy[ii] * 0.8
            mx.append(x[ii] + nx[ii] * h * 0.9 + (dxr + nx[ii] * wave) * Ls * q)
            my.append(y[ii] + ny[ii] * h * 0.9 + (dyr + ny[ii] * wave) * Ls * q)
            mc.append(FLAME[None] * (1 - q)[:, None] ** 0.7 + np.array([255, 205, 90])[None] * q[:, None])
            mi.append(np.full(16, ii))
        it = n - 1                                                           # the tail's flame tuft
        q = np.random.default_rng(3).random((2, 60))
        Ls = 0.16 * R * s[it] * m
        flick = 0.8 + 0.2 * math.sin(15 * t)
        txp = x[it] - fx[it] * Ls * q[0] * flick + nx[it] * Ls * 0.35 * (q[1] - 0.5) * (1 + q[0])
        typ = y[it] - fy[it] * Ls * q[0] * flick + ny[it] * Ls * 0.35 * (q[1] - 0.5) * (1 + q[0])
        mx.append(txp)
        my.append(typ)
        mc.append(FLAME[None] * (1 - q[0])[:, None] + np.array([255, 220, 120])[None] * q[0][:, None])
        mi.append(np.full(60, it))
        mx, my, mc, mi = np.concatenate(mx), np.concatenate(my), np.concatenate(mc), np.concatenate(mi)
        self.emit(mx, my, cls[mi], zc[mi], mc * (0.85 * m), 1.6, "glow")
        hx0, hy0 = P(0.0, 0.25)
        cl0 = int(cls[0])
        tgt = self.haze if cl0 in (0, 3) else self.wlight
        tgt.add(np.array([hx0, x[at(0.2)]]), np.array([hy0, y[at(0.2)]]), FLAME * 0.25 * m, 40 * self.S / 1000)
        # dorsal spines along the back
        sx, sy, si = [], [], []
        for uu in np.arange(0.45, 0.95, 0.03):
            ii = at(uu)
            h = top[ii] * R * s[ii]
            hh = 0.035 * R * s[ii] * m * (1 - 0.6 * uu)
            for qq_ in np.linspace(0, 1, 6):
                sx.append(x[ii] + nx[ii] * (h + hh * qq_) - fx[ii] * hh * 0.6 * qq_)
                sy.append(y[ii] + ny[ii] * (h + hh * qq_) - fy[ii] * hh * 0.6 * qq_)
                si.append(ii)
        si = np.array(si)
        self.emit(np.array(sx), np.array(sy), cls[si], zc[si], CRIMSON * 0.8, 1.5, "dot", 0.005)
        # four legs, paddling at the air
        lx, ly, li, lc = [], [], [], []
        for uu, far, ph in ((0.20, 0, 0.0), (0.23, 1, 1.6), (0.60, 0, 3.1), (0.63, 1, 4.7)):
            ii = at(uu)
            h = bot[ii] * R * s[ii]
            hipx, hipy = x[ii] - nx[ii] * h * 0.8, y[ii] - ny[ii] * h * 0.8
            Ls = R * s[ii] * m
            sw = 0.5 * math.sin(4.2 * t + ph)
            kx = hipx + (-fx[ii] * (0.05 + 0.03 * sw) - nx[ii] * 0.08) * Ls
            ky = hipy + (-fy[ii] * (0.05 + 0.03 * sw) - ny[ii] * 0.08) * Ls
            ftx = kx + (fx[ii] * (0.05 + 0.02 * sw) - nx[ii] * 0.05) * Ls
            fty = ky + (fy[ii] * (0.05 + 0.02 * sw) - ny[ii] * 0.05) * Ls
            q = np.linspace(0, 1, 14)
            segx = np.concatenate([hipx + (kx - hipx) * q, kx + (ftx - kx) * q])
            segy = np.concatenate([hipy + (ky - hipy) * q, ky + (fty - ky) * q])
            for wv in (-1, 0, 1):
                lx.append(segx + nx[ii] * 0 + fx[ii] * wv * 0.012 * Ls)
                ly.append(segy + fy[ii] * wv * 0.012 * Ls)
                li.append(np.full(28, ii))
                lc.append(np.broadcast_to(GOLD * (0.55 if far else 0.85), (28, 3)))
            for ca_ in (-0.5, 0.0, 0.5):                                     # claws
                q2 = np.linspace(0, 1, 6)
                cdx = fx[ii] * math.cos(ca_) - nx[ii] * math.sin(ca_) * 0.6 - nx[ii] * 0.4
                cdy = fy[ii] * math.cos(ca_) - ny[ii] * math.sin(ca_) * 0.6 - ny[ii] * 0.4
                lx.append(ftx + cdx * 0.03 * Ls * q2)
                ly.append(fty + cdy * 0.03 * Ls * q2)
                li.append(np.full(6, ii))
                lc.append(np.broadcast_to(np.array([240, 232, 210]) * (0.6 if far else 1.0), (6, 3)))
        lx, ly, li, lc = np.concatenate(lx), np.concatenate(ly), np.concatenate(li), np.concatenate(lc)
        bias = np.where(np.concatenate([np.full(len(z), 0.0) for z in [lx]]), 0, 0)
        del bias
        self.emit(lx, ly, cls[li], zc[li], lc * m, max(1.0, 1.8 * self.S / 1000), "dot", 0.004)

    # ---- the jellyfish ---------------------------------------------------------------------------
    def build_jelly(self):
        rng = np.random.default_rng(21)
        n = 1500
        self.jb = dict(th=rng.uniform(0, TAU, n), ph=rng.random(n) ** 0.7)
        g = []
        for k in range(4):
            c = k * math.pi / 2 + math.pi / 4
            psi = rng.uniform(0.35, TAU - 0.35, 90)
            g.append(np.stack([0.42 * math.cos(c) + 0.20 * np.cos(psi + c), 0.42 * math.sin(c) + 0.20 * np.sin(psi + c)], -1))
        self.jg = np.vstack(g)
        self.jt = rng.uniform(0, TAU, 110)
        self.jarm = rng.uniform(0, TAU, 4)

    def jelly_pos(self, t):
        return np.array([-0.34 + 0.07 * math.sin(0.13 * t), 0.02 + 0.10 * math.sin(0.11 * t + 1.0) + 0.02 * math.sin(TAU * t / 1.55 - 1.0),
                         0.30 + 0.05 * math.sin(0.09 * t)])

    def jelly_level(self, t):
        g = 0.35 + 0.65 * float(ss(t, 5.0, 17.0))
        g *= 1 + 1.2 * float(ss(t, 29.5, 31.0) * (1 - ss(t, 31.2, 32.2)))
        g *= 1 - 0.62 * float(ss(t, 31.2, 32.6) * (1 - ss(t, 53.0, 54.0)))
        g *= 1 + 1.4 * math.exp(-((t - 53.8) / 0.6) ** 2)
        return g

    def draw_jelly(self, t):
        J, G = self.jelly_pos(t), self.jelly_level(t)
        ph = (t / 1.55) % 1.0
        c = math.sin(math.pi * min(ph / 0.45, 1.0)) ** 2
        rb, hb = 0.15 * (1 - 0.16 * c), 0.55 * 0.15 * (1 + 0.22 * c)
        B = self.jb
        th, p = B["th"], B["ph"]
        sp_ = np.sin(p * math.pi / 2) ** 0.85
        rim = 0.012 * np.sin(6 * th + 8 * t) * p ** 4
        X = J[0] + rb * sp_ * np.cos(th)
        Y = J[1] + hb * np.cos(p * math.pi / 2) + rim
        Z = J[2] + rb * sp_ * np.sin(th)
        fres = 1 - np.abs(sp_ * np.sin(th))
        cols = [JELLY[None] * ((0.12 + 0.80 * fres ** 2 + 0.35 * p ** 6) * G)[:, None]]
        pts = [np.stack([X, Y, Z], -1)]
        gp = self.jg                                                         # the four rings of the gonads
        rho = np.hypot(gp[:, 0], gp[:, 1])
        pts.append(np.stack([J[0] + rb * gp[:, 0], J[1] + hb * np.sqrt(np.clip(1 - rho ** 2, 0, 1)) * 0.9, J[2] + rb * gp[:, 1]], -1))
        cols.append(np.broadcast_to(np.array([255, 140, 230]) * 0.40 * G, (len(gp), 3)))
        q = np.linspace(0, 1, 8)[None, :]                                    # the fringe of fine tentacles
        tt = self.jt[:, None]
        rr = rb * (1 + 0.12 * q)
        Xt = J[0] + rr * np.cos(tt) + 0.015 * np.sin(5 * q - 3 * t + tt) * q
        Yt = J[1] - 0.12 * q * (1 - 0.3 * c)
        Zt = J[2] + rr * np.sin(tt)
        pts.append(np.stack([Xt.ravel(), np.broadcast_to(Yt, Xt.shape).ravel(), Zt.ravel()], -1))
        cols.append(np.broadcast_to(JELLY * 0.16 * G, (Xt.size, 3)))
        q = np.linspace(0, 1, 40)                                            # four frilly oral arms
        for k in range(4):
            a0 = k * math.pi / 2 + self.jarm[k] * 0.2
            cxa = J[0] + 0.03 * math.cos(a0) + 0.05 * np.sin(2.2 * t - 3.5 * q + k) * q
            cza = J[2] + 0.03 * math.sin(a0) + 0.03 * np.cos(1.9 * t - 3.0 * q + k) * q
            cya = J[1] - 0.02 - 0.34 * q
            for f in (-1, 0, 1):
                fr = 0.026 * (1 - 0.4 * q) * np.sin(38 * q + 2 * k + 3 * t) * f
                pts.append(np.stack([cxa + fr * math.cos(a0 + 1.57), cya, cza + fr * math.sin(a0 + 1.57)], -1))
                cols.append(np.array([230, 200, 255])[None] * (0.26 * G * (0.55 + 0.45 * np.sin(9 * q - 5 * t)))[:, None])
        P = np.vstack(pts)
        col = np.vstack(cols)
        cls, zc = self.classify(P)
        x, y, s, _ = self.proj(P[:, 0], P[:, 1], P[:, 2])
        self.emit(x, y, cls, zc, col, 1.2, "glow")
        jx, jy, js, _ = self.proj(J[0], J[1], J[2])
        self.wlight.add(np.array([jx]), np.array([jy]), np.array([50, 150, 255]) * 0.55 * G, 120 * self.S / 1000)
        self.wlight.add(np.array([jx]), np.array([jy]), np.array([120, 200, 255]) * 0.5 * G, 40 * self.S / 1000)
        return J, G

    # ---- the gate, weeds, the pearl, splashes, sparks ---------------------------------------------------
    def build_misc(self):
        rng = np.random.default_rng(30)
        pts, cols = [], []
        verm, black, gold = np.array([214, 60, 40]), np.array([34, 26, 28]), np.array([236, 190, 80])
        verm = np.array([232, 64, 36])
        for sx in (-0.12, 0.12):                                             # two slim posts, leaning in a touch
            n = 180
            yy = rng.uniform(-0.80, -0.445, n)
            lean = -np.sign(sx) * 0.012 * (yy + 0.80) / 0.355
            pts.append(np.stack([sx + lean + rng.uniform(-0.011, 0.011, n), yy, rng.uniform(-0.011, 0.011, n)], -1))
            cols.append(np.broadcast_to(verm, (n, 3)))
            n = 30                                                           # black feet
            pts.append(np.stack([sx + rng.uniform(-0.016, 0.016, n), rng.uniform(-0.80, -0.775, n), rng.uniform(-0.013, 0.013, n)], -1))
            cols.append(np.broadcast_to(black, (n, 3)))
        n = 150                                                              # the tie beam, through the posts
        pts.append(np.stack([rng.uniform(-0.165, 0.165, n), rng.uniform(-0.515, -0.500, n), rng.uniform(-0.009, 0.009, n)], -1))
        cols.append(np.broadcast_to(verm, (n, 3)))
        n = 200                                                              # the red beam under the top
        xb = rng.uniform(-0.18, 0.18, n)
        pts.append(np.stack([xb, rng.uniform(-0.452, -0.438, n) + 0.025 * (np.abs(xb) / 0.18) ** 3, rng.uniform(-0.01, 0.01, n)], -1))
        cols.append(np.broadcast_to(verm, (n, 3)))
        n = 260                                                              # the black top, sweeping up at the ends
        xb = rng.uniform(-0.215, 0.215, n)
        pts.append(np.stack([xb, rng.uniform(-0.436, -0.418, n) + 0.045 * (np.abs(xb) / 0.215) ** 3, rng.uniform(-0.012, 0.012, n)], -1))
        cols.append(np.broadcast_to(black, (n, 3)))
        n = 50                                                               # the gold name plaque
        pts.append(np.stack([rng.uniform(-0.022, 0.022, n), rng.uniform(-0.495, -0.455, n), np.full(n, -0.014)], -1))
        cols.append(np.broadcast_to(gold, (n, 3)))
        self.gate = (np.vstack(pts), np.vstack(cols).astype(np.float32))
        W_ = []
        for bx, bz, h in ((-0.48, 0.15, 0.80), (0.50, 0.10, 0.70), (-0.18, 0.42, 0.72), (0.28, 0.40, 0.55)):
            q = np.linspace(0, 1, 70)
            W_.append((bx, bz, h, q, rng.uniform(0, TAU)))
        self.weeds = W_
        self.drops = [(26.0, self.path(np.array([26.0]))[0], 90), (self.t_dive, self.path(np.array([self.t_dive]))[0], 110),
                      (33.6, np.array([0.0, YW, 0.05]), 24)]
        self.drop_v = rng.normal(0, 1, (110, 3))
        self.drop_k = rng.uniform(0.6, 1.0, 110)
        n = 260
        self.motes = dict(x=rng.uniform(0.15, 0.6, n), y=rng.uniform(0.25, 0.85, n), ph=rng.uniform(0, TAU, n))
        self.bubble_t = np.arange(1.0, 60.0, 3.3)

    def draw_gate(self, t):
        P, col = self.gate
        cls, zc = self.classify(P)
        x, y, s, _ = self.proj(P[:, 0], P[:, 1], P[:, 2])
        lit = math.exp(-((t - 20.2) / 0.7) ** 2)
        c = col * 0.62 + np.array([255, 210, 90]) * 0.5 * lit
        self.emit(x, y, cls, zc, c, self.rad, "dot")
        if lit > 0.02:
            self.emit(x[::3], y[::3], cls[::3], zc[::3], np.array([255, 200, 80]) * 0.9 * lit, 1.6, "glow")
            gx, gy, _, _ = self.proj(0.0, -0.6, 0.0)
            self.wlight.add(np.array([gx]), np.array([gy]), np.array([255, 190, 80]) * 1.2 * lit, 90 * self.S / 1000)

    def draw_weeds(self, t):
        swirl = float(ss(t, 21.5, 23.0) * (1 - ss(t, 26.0, 27.5)))
        pts, cols = [], []
        for bx, bz, h, q, ph in self.weeds:
            Y = -0.80 + h * q
            sw = (0.06 + 0.12 * swirl) * q ** 1.5 * math.sin(0.8 * t + ph) + 0.10 * swirl * q ** 2 * math.sin(3.0 * t)
            X = bx + sw
            Z = bz + 0.04 * q ** 1.5 * math.cos(0.7 * t + ph) + 0.10 * swirl * q ** 2 * math.cos(3.0 * t)
            pts.append(np.stack([X, Y, Z], -1))
            cols.append(np.broadcast_to(np.array([40, 92, 54]), (len(q), 3)))
            for j in range(0, 70, 5):                                        # whorls of fine leaves
                for a in np.linspace(0, TAU, 7)[:-1]:
                    qq = np.linspace(0.2, 1, 5)
                    L = 0.06 * (1 - 0.4 * q[j])
                    pts.append(np.stack([X[j] + np.cos(a + ph) * L * qq, Y[j] + 0.02 * qq, Z[j] + np.sin(a + ph) * L * qq], -1))
                    cols.append(np.broadcast_to(np.array([56, 120, 64]), (5, 3)))
        P = np.vstack(pts)
        col = np.vstack(cols).astype(np.float32)
        cls, zc = self.classify(P)
        x, y, s, _ = self.proj(P[:, 0], P[:, 1], P[:, 2])
        self.emit(x, y, cls, zc, col * 0.75, self.rad, "dot")

    def pearl_pos(self, t, J):
        if t < 31.2:
            return J, 0.0
        if t < 34.0:
            q = float(ss(t, 31.2, 34.0))
            P0, P1 = self.jelly_pos(31.2), np.array([0.05, 1.25, 0.05])
            p = P0 + (P1 - P0) * q + np.array([0.03 * math.sin(5 * t), 0, 0])
            return p, float(ss(t, 31.2, 31.8))
        if t < 47.0:
            lead = 0.9 * (1 - float(ss(t, 44.0, 46.9)))
            tgt = self.path(np.array([t + lead]))[0]
            w = float(ss(t, 34.0, 35.6))
            return np.array([0.05, 1.25, 0.05]) + (tgt - np.array([0.05, 1.25, 0.05])) * w, 1.0
        if t < 50.3:
            h = self.path(np.array([t, t - 0.05]))
            fwd = h[0] - h[1]
            fwd /= np.linalg.norm(fwd) + 1e-9
            return h[0] + fwd * 0.30, 1.0
        if t < 53.2:
            h = self.path(np.array([50.3, 50.25]))
            fwd = (h[0] - h[1]) / (np.linalg.norm(h[0] - h[1]) + 1e-9)
            p0 = h[0] + fwd * 0.30
            q = float(ss(t, 50.3, 53.2))
            return p0 + (J - p0) * q, 1.0 - 0.3 * q
        return J, 0.0

    def draw_pearl(self, t, J):
        p, k = self.pearl_pos(t, J)
        if k <= 0.01:
            return
        trail = [p]
        for j in range(1, 18):
            q, _ = self.pearl_pos(t - 0.035 * j, self.jelly_pos(t - 0.035 * j))
            trail.append(q)
        P = np.array(trail)
        cls, zc = self.classify(P)
        x, y, s, _ = self.proj(P[:, 0], P[:, 1], P[:, 2])
        fade = (1 - np.arange(len(P)) / len(P)) ** 2
        self.emit(x[1:], y[1:], cls[1:], zc[1:], PEARL[None] * (0.6 * k * fade[1:])[:, None], 1.6, "glow")
        rr = 0.055 * self.R * s[0]
        rg = np.random.default_rng(5)
        a, r_ = rg.uniform(0, TAU, 160), np.sqrt(rg.random(160))
        core_x, core_y = x[0] + np.cos(a) * r_ * rr, y[0] + np.sin(a) * r_ * rr
        shade = (1.6 - 1.2 * r_)[:, None]
        self.emit(np.append(core_x, x[0]), np.append(core_y, y[0]), np.full(161, cls[0]), np.full(161, zc[0]),
                  np.vstack([PEARL[None] * shade * 0.9 * k, [[255 * 3 * k, 255 * 3 * k, 255 * 3 * k]]]), 1.6, "glow")
        sw = []                                                              # flame wisps winding round it
        for j in range(3):
            q = np.linspace(0, 1, 16)
            ang = 4 * t + j * TAU / 3 + q * 3.0
            rad_ = rr * (1.4 + 1.2 * q)
            sw.append((x[0] + np.cos(ang) * rad_, y[0] + np.sin(ang) * rad_ * 0.8 - q * rr * 1.5, q))
        wx = np.concatenate([w[0] for w in sw])
        wy = np.concatenate([w[1] for w in sw])
        wq = np.concatenate([w[2] for w in sw])
        self.emit(wx, wy, np.full(len(wx), cls[0]), np.full(len(wx), zc[0]), np.array([180, 230, 255])[None] * (0.6 * k * (1 - wq))[:, None],
                  1.3, "glow")
        tgt = self.wlight if cls[0] in (1, 2) else self.haze
        tgt.add(np.full(3, x[0]), np.full(3, y[0]), np.array([120, 200, 255]) * 1.6 * k, 90 * self.S / 1000)
        a6 = np.linspace(0, TAU, 30, endpoint=False) + 2 * t                 # a ring of sparkle round it
        self.emit(x[0] + np.cos(a6) * rr * 1.35, y[0] + np.sin(a6) * rr * 1.35, np.full(30, cls[0]), np.full(30, zc[0]),
                  np.array([200, 240, 255])[None] * (0.35 * k * np.clip(np.sin(a6 * 5 + 9 * t), 0, 1) ** 3)[:, None], 1.0, "glow")

    def draw_splashes(self, t):
        for t0, P0, n in self.drops:
            a = t - t0
            if not 0 <= a < 2.2:
                continue
            v = self.drop_v[:n] * np.array([0.9, 0.0, 0.9]) + np.array([0, 1, 0]) * (2.2 + 1.4 * self.drop_k[:n])[:, None]
            v *= 0.6 if n < 50 else 1.0
            g = 7.5
            pos = P0[None] + v * a + np.array([0, -0.5 * g * a * a, 0])
            alive = (pos[:, 1] > YW - 0.02) | (np.hypot(pos[:, 0], pos[:, 2]) > AW)
            alive &= pos[:, 1] > -1.0
            if alive.any():
                P = pos[alive]
                cls, zc = self.classify(P)
                x, y, s, _ = self.proj(P[:, 0], P[:, 1], P[:, 2])
                self.emit(x, y, cls, zc, np.array([200, 232, 255]) * 0.9 * (1 - a / 2.2), 1.5, "glow")
        # rings spreading on the water
        xs, ys, cs = [], [], []
        th = np.linspace(0, TAU, 90, endpoint=False)
        for t0, P0, n in self.drops:
            a = t - t0
            if not 0 <= a < 2.6:
                continue
            for d in (0.0, 0.25):
                r = 0.55 * max(a - d, 0) + 0.02
                X, Z = P0[0] + r * np.cos(th), P0[2] + r * np.sin(th)
                ok = np.hypot(X, Z) < AW * 0.98
                x, y, s, _ = self.proj(X[ok], np.full(int(ok.sum()), YW), Z[ok])
                xs.append(x)
                ys.append(y)
                cs.append(np.broadcast_to(np.array([170, 210, 255]) * 0.45 * (1 - a / 2.6) * (n / 90), (len(x), 3)))
        if xs:
            glow_ = (np.concatenate(xs), np.concatenate(ys), np.concatenate(cs))
            self.surface.append(glow_)

    def draw_vortex(self, t):
        vs = float(ss(t, 21.5, 23.0) * (1 - ss(t, 26.0, 27.0)))
        if vs < 0.01:
            return
        xs, ys, cs = [], [], []
        rho = np.linspace(0.06, AW * 0.92, 50)
        for k in range(3):
            ang = k * TAU / 3 + 2.6 * np.log(rho / AW) * -1 + 3.2 * t
            X, Z = rho * np.cos(ang), 0.05 + rho * np.sin(ang)
            Y = YW - 0.12 * vs * (1 - rho / AW) ** 2
            x, y, _, _ = self.proj(X, Y, Z)
            xs.append(x)
            ys.append(y)
            cs.append(np.broadcast_to(np.array([170, 210, 255]) * 0.40 * vs, (50, 3)))
        for d in np.linspace(0.02, 0.55, 12):                                # the funnel under the surface
            rf = 0.45 * (1 - d / 0.55) ** 1.5 + 0.02
            a = np.linspace(0, TAU, 24, endpoint=False) + 6.0 * t * (1 + d * 2)
            x, y, _, _ = self.proj(rf * np.cos(a), np.full(24, YW - d), 0.05 + rf * np.sin(a))
            xs.append(x)
            ys.append(y)
            cs.append(np.broadcast_to(np.array([140, 200, 255]) * 0.16 * vs, (24, 3)))
        self.surface.append((np.concatenate(xs), np.concatenate(ys), np.concatenate(cs)))

    def draw_sparks(self, t, sp, m):
        if m > 0.3:                                                          # shed from the dragon as it flies
            S_ = self.spark
            age = (t - S_["off"]) % 1.3
            tb = t - age
            origin = self.path(tb - S_["lag"])
            P = origin + S_["v"] * age[:, None] * 0.5 + np.stack([np.zeros_like(age), -0.25 * age ** 2, np.zeros_like(age)], -1)
            alive = np.array([self.morph(b) > 0.6 for b in tb])
            if alive.any():
                P = P[alive]
                cls, zc = self.classify(P)
                x, y, s, _ = self.proj(P[:, 0], P[:, 1], P[:, 2])
                b = (1 - age[alive] / 1.3) * (0.6 + 0.4 * np.sin(20 * t + S_["ph"][alive]))
                self.emit(x, y, cls, zc, np.array([255, 200, 90])[None] * (b * 0.9)[:, None], 1.2, "glow")
        for tc, scale in ((27.75, 1.0), (47.0, 0.55)):                       # bursts: the change, and the catch
            a = t - tc
            if 0 <= a < 2.2:
                c = self.path(np.array([tc]))[0]
                r = 0.95 * scale * (1 - math.exp(-a / 0.45))
                P = c[None] + self.burst * r + np.array([0, -0.15 * a * a, 0])
                cls, zc = self.classify(P)
                x, y, s, _ = self.proj(P[:, 0], P[:, 1], P[:, 2])
                self.emit(x, y, cls, zc, np.array([255, 214, 120]) * 1.4 * math.exp(-a / 0.8), 1.5, "glow")

    def draw_bubbles(self, t, sp):
        xs, ys, cs = [], [], []
        for tb in self.bubble_t:
            a = t - tb
            if not 0 < a < 2.4 or self.morph(tb) > 0.1:
                continue
            P0 = self.path(np.array([tb]))[0]
            Y = P0[1] + 0.35 * a
            if Y > YW:
                continue
            X = P0[0] + 0.02 * math.sin(6 * a)
            x, y, s, _ = self.proj(X, Y, P0[2])
            rr = (2 + 1.5 * a) * self.S / 1000
            ang = np.linspace(0, TAU, 8, endpoint=False)
            xs.append(float(x) + np.cos(ang) * rr)
            ys.append(float(y) + np.sin(ang) * rr)
            cs.append(np.broadcast_to(np.array([160, 210, 240]) * 0.5, (8, 3)))
        if xs:
            self.surface.append((np.concatenate(xs), np.concatenate(ys), np.concatenate(cs)))

    # ---- the frame ---------------------------------------------------------------------------------
    def render(self, t):
        W, H, S = self.W, self.H, self.S
        img = self.base.copy()
        self.A, self.B, self.gA, self.gB, self.surface = [], [], [], [], []
        cp = self.caustic_pts                                                # the knot of light on the table shimmers
        if len(cp):
            sh = 0.5 + 0.5 * np.sin(cp[:, 0] * 0.11 + cp[:, 1] * 0.07 + 3 * t) * np.sin(cp[:, 0] * 0.05 - 2 * t)
            glow(img, cp[:, 0], cp[:, 1], np.array([150, 220, 255])[None] * (0.35 * sh ** 3)[:, None], 1.0)
        M = self.motes                                                       # dust turning in the moonbeam
        mx = (M["x"] + 0.01 * np.sin(0.2 * t + M["ph"])) * W
        my = ((M["y"] + 0.004 * t) % 0.7 + 0.2) * H
        bk = self.beam[np.clip(my.astype(int), 0, H - 1), np.clip(mx.astype(int), 0, W - 1)]
        glow(img, mx, my, np.array([160, 180, 230])[None] * (bk * 0.9 * (0.5 + 0.5 * np.sin(2 * t + M["ph"])))[:, None], 1.0)
        self.draw_weeds(t)
        self.draw_gate(t)
        J, G = self.draw_jelly(t)
        sp, m = self.draw_creature(t)
        self.draw_pearl(t, J)
        self.draw_sparks(t, sp, m)
        self.draw_splashes(t)
        self.draw_vortex(t)
        self.draw_bubbles(t, sp)
        paint(img, self.A)                                                   # inside the bowl, and seen through it
        for x, y, c, r in self.gA:
            glow(img, x, y, c, r)
        self.wlight.apply(img, 1.0, 2, self.water_mask)
        wy, wa, wb = self.water_e                                            # the water's edge, and what moves on it
        th = np.linspace(0, TAU, 500, endpoint=False)
        ex, ey = self.cx + wa * np.cos(th), wy + wb * np.sin(th)
        front = np.sin(th) > 0
        glow(img, ex, ey, np.array([120, 150, 200])[None] * np.where(front, 0.55, 0.22)[:, None] * (0.8 + 0.2 * G), 1.2)
        for x, y, c in self.surface:
            glow(img, x, y, c, 1.2)
        gx, gy, gc, gr = self.glass
        glow(img, gx, gy, gc, gr)
        paint(img, self.B)                                                   # out in the room, in front of the glass
        for x, y, c, r in self.gB:
            glow(img, x, y, c, r)
        fl = math.exp(-((t - 27.85) / 0.32) ** 2)                            # the flash as the fish becomes a dragon
        if fl > 0.01:
            img += np.array([255, 196, 120], np.float32) * (0.22 * fl)
            c = self.path(np.array([27.75]))[0]
            px, py, _, _ = self.proj(c[0], c[1], c[2])
            a = np.linspace(0, TAU, 16, endpoint=False) + 0.2
            q = np.linspace(0.04, 1, 160)[None, :]
            ln = (0.40 + 0.25 * np.sin(np.arange(16) * 2.3))[:, None] * S * fl
            glow(img, (px + np.cos(a)[:, None] * ln * q).ravel(), (py + np.sin(a)[:, None] * ln * q).ravel(),
                 (np.array([255, 220, 150])[None] * ((1 - q) ** 1.5 * 0.8 * fl).reshape(-1, 1)).repeat(16, 0)[:16 * 160]
                 if False else np.array([255, 220, 150])[None] * np.tile(((1 - q[0]) ** 1.5 * 0.9 * fl), 16)[:, None], 1.3)
            self.haze.add(np.full(4, px), np.full(4, py), np.array([255, 200, 110]) * 1.6 * fl, 120 * S / 1000)
        fx_, fy_, fs = self.flame                                            # the candle
        flick = 1 + 0.10 * math.sin(13 * t) + 0.06 * math.sin(29 * t + 1)
        q = np.linspace(0, 1, 30)
        hgt = 0.075 * self.R * fs * flick
        sway = 0.006 * self.R * math.sin(7 * t)
        for w_, c_, rr in ((0.7, (255, 150, 50), 2.0), (0.35, (255, 236, 190), 1.4)):
            for side in np.linspace(-1, 1, 5):
                xs_ = fx_ + side * w_ * 0.02 * self.R * fs * np.sin(np.pi * q) ** 0.7 + sway * q ** 2
                glow(img, xs_, fy_ - q * hgt, np.array(c_)[None] * (0.6 * (1 - 0.5 * q))[:, None], rr)
        glow(img, np.array([fx_]), np.array([fy_ + 0.01 * self.R]), np.array([80, 110, 255]) * 0.6, 1.5)
        self.haze.add(np.full(2, fx_), np.full(2, fy_ - hgt * 0.4), np.array([255, 150, 60]) * 0.45 * flick, 60 * S / 1000)
        self.haze.apply(img, 1.0, 2)
        np.clip(img, 0, 4095, out=img)
        return self.lut[img.astype(np.uint16)]


if __name__ == "__main__":
    run(DragonBowl, "The Dragon Gate", "dragon_bowl", T_END, STAGES)
