"""
synthwave_drive.py - an endless night drive into a striped sun, in dots. A seamless one-minute loop.

A low wedge of a car races down a straight highway toward a huge sun banded in gold and hot
pink, sinking behind wireframe mountains. A magenta grid rushes under the wheels, cyan lines
mark the road's edges, palms flick past on both sides and the glossy road throws back a long
pink streak of the sun. The car's tail-light bar burns across its back, a cyan underglow
spills onto the asphalt, and it drifts from lane to lane. Stars prick the violet sky and out
at the edges, storm clouds over the desert flicker with lightning.

    python synthwave_drive.py            the drive, looping forever (Space pause, R restart, S save frame, Esc quit)
    python synthwave_drive.py --png      save five moments
    python synthwave_drive.py --gif      save one seamless loop as a GIF
"""
import math

import numpy as np

from neon import NEON, TAU, Cam, Lights, finish, glow, paint, run, ss, stamp, tone_lut

T_END = 60.0
STAGES = [3.0, 12.38, 24.0, 41.3, 50.0]
V = 25.0                         # metres a second: 1500 m a minute, a whole number of grid squares and palms
GRID = 10.0
PALM_GAP = 50.0
SUN_C = (0.5, 0.40)
BOLTS = [(12.3, -1, 0.9), (12.45, -1, 0.6), (41.2, 1, 1.0), (41.32, 1, 0.5), (53.0, -1, 0.5)]
MAG = np.array(NEON["magenta"], np.float32)
PINK = np.array(NEON["pink"], np.float32)
CYAN = np.array(NEON["cyan"], np.float32)


class SynthwaveDrive:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.S = S = min(W, H)
        self.cam = Cam(W, H, 0.5 * W, 0.53 * H, 0.80 * S, 1.5)
        self.lut = tone_lut(1.35, 5.0)
        self.L = Lights(W, H, 4)
        self.rng = np.random.default_rng(80)
        self.build_backdrop()
        self.build_palms()
        self.build_car()

    # ---- sky, mountains, the desert floor ------------------------------------------------------------
    def build_backdrop(self):
        W, H, c = self.W, self.H, self.cam
        rng = np.random.default_rng(2)
        yy = np.arange(H, dtype=np.float32)[:, None] / H
        hz = c.vy / H
        stops = ((0.0, (14, 4, 40)), (0.25, (40, 8, 80)), (0.42, (120, 20, 120)), (hz - 0.02, (240, 70, 140)), (hz, (255, 120, 120)))
        ts = [s_[0] for s_ in stops]
        sky = np.stack([np.interp(yy, ts, [s_[1][k] for s_ in stops]) for k in range(3)], -1)[..., 0, :]
        img = np.zeros((H, W, 3), np.float32)
        img[:] = sky[:, None, :]
        # storm clouds out at the edges, and streaks of cloud lit pink across the sun
        xs = np.arange(W, dtype=np.float32)[None, :] / W
        edge = np.clip((np.abs(xs - 0.5) - 0.18) / 0.32, 0, 1)
        billow = (0.5 + 0.5 * np.sin(xs * 23 + np.sin(yy * 17) * 2) * np.sin(yy * 31 - xs * 9)) ** 2
        cloud = edge * np.exp(-((yy - (hz - 0.10)) / 0.08) ** 2) * (0.4 + 0.6 * billow)
        self.cloud = cloud.astype(np.float32)
        img = img * (1 - 0.6 * cloud[..., None]) + np.array([40, 10, 60], np.float32) * cloud[..., None]
        streak = np.exp(-((yy - 0.30) / 0.006) ** 2) * 0.5 + np.exp(-((yy - 0.355) / 0.004) ** 2) * 0.4
        img += np.array([255, 90, 170], np.float32) * (streak * np.exp(-((xs - 0.5) / 0.3) ** 2))[..., None] * 0.25
        # mountains: two ranges of ridges
        self.mask = np.zeros((H, W), np.float32)                             # 1 where the mountains are
        self.wire = []
        u = np.arange(W) / W
        for back, (base, amp, n_peaks, col, wcol) in enumerate((
                (hz, 0.13, 7, (34, 10, 60), CYAN), (hz, 0.07, 5, (22, 6, 40), MAG))):
            prof = np.zeros(W)
            r2 = np.random.default_rng(10 + back)
            for _ in range(n_peaks * 3):
                cx, w, h = r2.uniform(-0.1, 1.1), r2.uniform(0.05, 0.16), r2.uniform(0.3, 1.0)
                prof = np.maximum(prof, h * np.clip(1 - np.abs(u - cx) / w, 0, 1) ** 1.2)
            clear = np.clip((np.abs(u - 0.5) - (0.10 if back == 0 else 0.22)) / 0.12, 0, 1)   # a valley for the road
            prof = prof * (0.25 + 0.75 * clear)
            top = H * (base - amp * prof)
            m = (yy * H >= top[None, :]) & (yy * H < c.vy)
            shade = np.clip((yy * H - top[None, :]) / (H * amp + 1), 0, 1)
            mcol = np.array(col, np.float32) * (1.3 - 0.6 * shade[..., None]) + np.array([90, 20, 80], np.float32) * (1 - shade[..., None]) * 0.4
            img = np.where(m[..., None], mcol, img)
            self.mask = np.maximum(self.mask, m.astype(np.float32))
            # its wireframe: the ridge line, lines down its slopes, contours
            xs_ = np.arange(0, W, 2.0)
            ys_ = np.interp(xs_, np.arange(W), top)
            self.wire.append((xs_, ys_, wcol, 1.0))
            for x0 in np.arange(0, W, 38.0 * (1 + back)):
                y0 = float(np.interp(x0, np.arange(W), top))
                q = np.linspace(0, 1, 30)
                self.wire.append((x0 + (r2.uniform(-1, 1) * 30) * q, y0 + (c.vy - y0) * q, wcol, 0.45))
            for f in (0.35, 0.65):
                ys2 = np.interp(xs_, np.arange(W), top + (c.vy - top) * f)
                keep = ys2 < c.vy - 2
                self.wire.append((xs_[keep], ys2[keep], wcol, 0.35))
        # the floor: almost black, a faint violet sheen near the horizon
        floor = yy * H >= c.vy
        g = np.clip((yy * H - c.vy) / (H - c.vy), 0, 1)
        img = np.where(floor[..., None], np.array([10, 2, 22], np.float32) + np.array([60, 10, 60], np.float32) * (1 - g[..., None]) ** 6, img)
        # stipple
        n = int(W * H * 0.35)
        x, y = rng.uniform(0, W, n), rng.uniform(0, H, n)
        stamp(img, x, y, img[y.astype(int), x.astype(int)] * (0.85 + 0.3 * rng.random((n, 1))), rng.choice([1.0, 1.5, 2.0], n))
        self.base = img
        n = 900
        self.stars = (rng.uniform(0, W, n), rng.random(n) ** 1.6 * 0.38 * H, rng.random(n) ** 3, rng.uniform(0, TAU, n),
                      rng.integers(1, 6, n))
        # the sun's own grid of pixels: a disc, banded, fading gold to pink
        R = 0.22 * self.S
        cx, cy = SUN_C[0] * W, SUN_C[1] * H
        yy2, xx2 = np.mgrid[int(cy - R) - 2:int(cy + R) + 3, int(cx - R) - 2:int(cx + R) + 3]
        d = np.hypot(xx2 - cx, yy2 - cy)
        self.sun = dict(y=yy2, x=xx2, inside=np.clip(R + 0.5 - d, 0, 1), v=(yy2 - (cy - R)) / (2 * R), R=R, cx=cx, cy=cy)

    def draw_sun(self, img, t):
        Sn = self.sun
        if "col" not in Sn:                                                   # the parts that never change
            v = Sn["v"]
            Sn["col"] = np.stack([np.interp(v, [0, 0.35, 0.65, 1.0], cc) for cc in ((255, 255, 255, 250), (240, 170, 80, 40),
                                                                                    (90, 70, 120, 170))], -1).astype(np.float32) * 1.25
            ys, xs = Sn["y"], Sn["x"]
            ya, xa = int(ys[0, 0]), int(xs[0, 0])
            Sn["sl"] = (slice(max(ya, 0), min(ya + ys.shape[0], self.H)), slice(max(xa, 0), min(xa + xs.shape[1], self.W)))
            Sn["sub"] = (slice(max(ya, 0) - ya, min(ya + ys.shape[0], self.H) - ya), slice(max(xa, 0) - xa, min(xa + xs.shape[1], self.W) - xa))
            m = self.mask[Sn["sl"]]
            Sn["a0"] = (Sn["inside"][Sn["sub"]] * (1 - m) * (ys[Sn["sub"]] < self.cam.vy)).astype(np.float32)
            Sn["v0"] = v[Sn["sub"]].astype(np.float32)
            Sn["gap"] = (np.clip((Sn["v0"] - 0.40) * 1.6, 0, 1) * 0.75).astype(np.float32)
            Sn["c0"] = Sn["col"][Sn["sub"]]
        # bands across its lower part that thicken toward the bottom, sliding slowly down
        ph = (Sn["v0"] * 9 + t * 3 / T_END * 3) % 1.0
        a = Sn["a0"] * ((ph > Sn["gap"]) | (Sn["v0"] < 0.40))
        a = a[..., None]
        reg = img[Sn["sl"]]
        reg *= 1 - a
        reg += Sn["c0"] * a
        # a soft pink halo round it
        if "halo" not in Sn:
            r = np.random.default_rng(4)
            ang, rr = r.uniform(0, TAU, 1500), 1.0 + r.random(1500) ** 1.8 * 0.9
            Sn["halo"] = (Sn["cx"] + rr * Sn["R"] * np.cos(ang), Sn["cy"] + rr * Sn["R"] * np.sin(ang), np.exp(-(rr - 1) / 0.3))
        hx, hy, hk = Sn["halo"]
        ok = hy < self.cam.vy
        self.L.bloom.splat(hx[ok], hy[ok], PINK[None] * (0.25 * hk[ok])[:, None])

    # ---- the road and the grid -------------------------------------------------------------------------
    def draw_ground(self, t):
        c, L = self.cam, self.L
        W, H = self.W, self.H
        off = (V * t) % GRID
        zl = np.arange(GRID, 320.0, GRID) - off + 2.0                        # cross lines rushing at us
        zl = zl[zl > 1.5]
        xg = np.arange(0, W, 3.5)
        Z = zl[:, None] * np.ones((1, len(xg)))
        X = (xg[None, :] - c.vx) * Z / c.f
        keep = np.abs(X) > 2.5
        gx = np.broadcast_to(xg[None, :], Z.shape)[keep]
        gy = (c.vy + c.f * c.hc / Z)[keep]
        gp = (0.55 * np.minimum(1.0, 18.0 / Z) ** 0.6)[keep]
        zz = 2.0 * (320 / 2.0) ** np.linspace(0, 1, 180)                      # long lines to the vanishing point
        Xs = np.array([X_ for X_ in np.arange(-60, 61, GRID * 0.5) if abs(X_) >= 2.5])
        lx, ly = c.proj(Xs[:, None] * np.ones((1, len(zz))), 0.0, np.broadcast_to(zz[None, :], (len(Xs), len(zz))))
        lp = np.broadcast_to(0.5 * np.minimum(1, 25 / zz) ** 0.4, lx.shape)
        ok = (lx > -5) & (lx < W + 5) & (ly < H + 5)
        L.tube(np.concatenate([gx, lx[ok]]), np.concatenate([gy, ly[ok]]), MAG, np.concatenate([gp, lp[ok]]),
               core=0.58, halo=0.8, bloom=0.5)
        for X in (-2.4, 2.4):                                                # the road edges
            x, y = c.proj(np.full_like(zz, X), np.zeros_like(zz), zz)
            ok = (y < H + 5)
            L.tube(x[ok], y[ok], CYAN, 0.9, core=0.9, halo=1.0, bloom=0.8)
        dash = ((zz + V * t) % GRID) < 3.5                                   # the dashed centre line
        x, y = c.proj(np.zeros_like(zz), np.zeros_like(zz), zz)
        L.tube(x[dash], y[dash], np.array([255, 230, 200], np.float32), 0.6, core=0.8, halo=0.6, bloom=0.4)
        # the glossy road throws back a long pink streak of the sun
        Sn = self.sun
        r = np.random.default_rng(int(t * 20) % 1200)
        n = 900
        yv = c.vy + (H - c.vy) * r.random(n) ** 1.6
        wob = np.sin(yv * 0.08 + TAU * 6 * t / T_END) * (0.01 + 0.04 * (yv - c.vy) / (H - c.vy)) * W
        xv = Sn["cx"] + r.normal(0, 0.035 * W, n) * (0.5 + (yv - c.vy) / (H - c.vy)) + wob
        k = np.exp(-((yv - c.vy) / (0.35 * H))) * (np.sin(yv * 0.6 - TAU * 3 * t / T_END) > -0.2)
        col = np.where(((yv - c.vy) / (H - c.vy) < 0.15)[:, None], np.array([255, 170, 90], np.float32), PINK)
        L.soft(xv, yv, col * (0.5 * k)[:, None])

    # ---- palms ---------------------------------------------------------------------------------------
    def build_palms(self):
        rng = np.random.default_rng(5)
        pts = []
        q = np.linspace(0, 1, 70)
        lean = 0.6
        pts.append(np.stack([lean * q ** 2, 9.0 * q, np.zeros(70)], 1))     # the trunk, curving
        top = np.array([lean, 9.0, 0.0])
        for k in range(9):                                                    # fronds, drooping
            a = TAU * k / 9 + rng.normal(0, 0.15)
            s = np.linspace(0, 1, 40)
            reach = rng.uniform(3.2, 4.4)
            d = np.stack([np.cos(a) * reach * s, 1.2 * s - 3.0 * s ** 2, np.sin(a) * reach * s * 0.4], 1)
            pts.append(top + d)
            for j in range(1, 40, 3):                                         # leaflets along each frond
                lf = np.linspace(0, 1, 5)[:, None]
                base = top + d[j]
                tip = base + np.array([0.0, -0.55 - 0.3 * s[j], 0.0]) + np.array([np.cos(a + 1.4), 0, 0]) * 0.25
                pts.append(base + (tip - base) * lf)
                tip2 = base + np.array([0.0, -0.55 - 0.3 * s[j], 0.0]) + np.array([np.cos(a - 1.4), 0, 0]) * 0.25
                pts.append(base + (tip2 - base) * lf)
        self.palm = np.vstack(pts)

    def draw_palms(self, dots, t):
        c = self.cam
        off = (V * t) % PALM_GAP
        P = self.palm
        for side, X0 in ((-1, -8.0), (1, 8.0), (-1, -15.0), (1, 15.0)):
            stagger = 0.0 if abs(X0) < 10 else PALM_GAP / 2
            for k in range(7):
                z = k * PALM_GAP + stagger - off + 6.0
                if z <= 2.0:
                    z += 7 * PALM_GAP
                step = 1 if z < 50 else (3 if z < 130 else 6)
                Pp = P[::step]
                X = X0 + side * Pp[:, 0]
                x, y = c.proj(X, Pp[:, 1], z + Pp[:, 2])
                ok = (x > -20) & (x < self.W + 20) & (y > -20) & (y < self.H)
                rim = np.clip((Pp[:, 1] - 7) / 4, 0, 1) * 0.5 + 0.2
                col = np.array([16, 4, 26], np.float32) + PINK * (0.18 * rim)[:, None] * min(1, 40 / z)
                r = np.clip(c.f * 0.10 / z * (1 + 0.4 * (step > 1)), 1.0, 3.0)
                dots.append((x[ok], y[ok], col[ok], r, np.full(int(ok.sum()), -z)))

    # ---- the car ---------------------------------------------------------------------------------------
    def build_car(self):
        """The car from behind, in car units: x -1..1 across, y up from the road. Each dot knows its part."""
        rng = np.random.default_rng(9)
        parts = []

        def quad(n, x0, x1, y0, y1, taper, kind, round_=0.0):
            u, v = rng.random(n), rng.random(n)
            y = y0 + (y1 - y0) * v
            half = (x1 - x0) / 2 * (1 - taper * v) * (1 - round_ * (1 - v) ** 3)
            x = (x0 + x1) / 2 + (u * 2 - 1) * half
            parts.append((x, y, np.full(n, kind), u, v))

        for sx in (-0.76, 0.76):
            quad(550, sx - 0.24, sx + 0.24, 0.0, 0.34, 0.0, 4)              # fat tyres, low at the corners
        quad(800, -0.92, 0.92, 0.10, 0.24, 0.0, 6)                           # the black diffuser
        quad(3100, -1.0, 1.0, 0.22, 0.56, 0.02, 0, 0.08)                     # the rear panel
        quad(900, -0.97, 0.97, 0.37, 0.48, 0.0, 7)                           # the tail-light band, recessed and black
        quad(1000, -0.98, 0.98, 0.56, 0.64, 0.04, 3)                         # the trunk deck, catching the sky
        quad(1600, -0.80, 0.80, 0.64, 0.96, 0.40, 1)                         # the cabin
        quad(1100, -0.62, 0.62, 0.67, 0.93, 0.42, 2)                         # rear glass
        quad(380, -0.24, 0.24, 0.26, 0.34, 0.0, 5)                           # the plate
        x, y, kind, u, v = (np.concatenate(a) for a in zip(*parts))
        self.car = dict(x=x, y=y, kind=kind.astype(int), u=u, v=v)

    def draw_car(self, dots, t):
        W, H, L = self.W, self.H, self.L
        C = self.car
        drift = 0.05 * W * math.sin(TAU * 2 * t / T_END) + 0.015 * W * math.sin(TAU * 7 * t / T_END)
        bob = 0.002 * H * math.sin(TAU * 41 * t / T_END)
        cx, base, half = 0.5 * W + drift, 0.99 * H + bob, 0.25 * W
        x = cx + C["x"] * half
        y = base - C["y"] * half * 0.95
        kind, u, v = C["kind"], C["u"], C["v"]
        across = np.abs(u * 2 - 1)
        col = np.zeros((len(x), 3), np.float32)
        panel = kind == 0                                                     # dark paint mirroring the sky and the road
        col[panel] = np.array([26, 8, 40], np.float32) + np.array([170, 40, 140], np.float32) * (v[panel] ** 4 * 0.45)[:, None] \
            + CYAN * (0.25 * across[panel] ** 12)[:, None]
        deck = kind == 3                                                      # the deck: a bright chrome-violet sheen
        col[deck] = np.array([110, 50, 170], np.float32) + np.array([255, 140, 200], np.float32) * (0.35 * (1 - across[deck] ** 2))[:, None]
        cab = kind == 1
        col[cab] = np.array([34, 12, 54], np.float32) + np.array([230, 110, 240], np.float32) * (0.9 * across[cab] ** 8 + 0.6 * (v[cab] > 0.94))[:, None]
        glass = kind == 2                                                     # the sun in the rear glass, banded
        stripe = (np.sin(v[glass] * 34 + TAU * 3 * t / T_END) > 0.0)
        col[glass] = np.array([16, 6, 34], np.float32) + np.array([255, 150, 120], np.float32) * \
            ((0.25 + 0.6 * stripe) * np.exp(-((u[glass] - 0.5) / 0.28) ** 2) * (1 - v[glass] * 0.4) + 0.08)[:, None]
        col[kind == 4] = np.array([7, 4, 10], np.float32) + np.array([60, 20, 80], np.float32) * (v[kind == 4] > 0.9)[:, None]
        col[kind == 6] = np.array([10, 6, 14], np.float32) * (0.6 + 0.6 * (np.sin(u[kind == 6] * 60) > 0.6))[:, None]
        col[kind == 7] = np.array([12, 4, 14], np.float32)
        col[kind == 5] = np.array([70, 50, 90], np.float32)
        dots.append((x, y, col, 1.9 * self.S / 1000 + 0.4, np.full(len(x), 1e4)))
        # the tail-light bar: one long line with segmented lamps, its glow on the road; the underglow
        q = np.linspace(-0.93, 0.93, 260)
        seg = (np.abs(q) > 0.55) | ((q * 30) % 1 < 0.75)
        tx, ty = cx + q * half, base - 0.425 * half * 0.95 + np.zeros_like(q)
        pulse = 0.9 + 0.1 * math.sin(TAU * 17 * t / T_END)
        red = np.array([255, 30, 90], np.float32)
        L.tube(tx[seg], ty[seg], red, 1.25 * pulse, core=1.0, halo=1.3, bloom=1.4)
        L.tube(tx[::2], ty[::2] + 0.006 * H, red, 0.35, core=0.4, halo=0.6, bloom=0.3)
        g = np.random.default_rng(int(t * 25) % 1500).random((2, 600))
        gx = cx + (g[0] * 2 - 1) * half * 1.05
        gy = base + 0.004 * H + g[1] ** 2 * 0.025 * H
        L.soft(gx, gy, CYAN * 0.6)
        L.soft(tx[::4], base + 0.012 * H + np.zeros_like(tx[::4]), red * 0.6)

    def draw_sky(self, img, t):
        L = self.L
        x, y, b, ph, f = self.stars
        tw = 0.6 + 0.4 * np.sin(TAU * f * t / T_END * 8 + ph)
        L.add(x, y, np.array([230, 220, 255])[None] * (40 + 200 * b * tw)[:, None] / 255 * 0.9, 1.0)
        for t0, side, k in BOLTS:                                            # lightning in the storms at the edges
            a = t - t0
            if -0.02 < a < 0.18:
                f = k * math.exp(-a / 0.06) if a >= 0 else 0
                cxs = 0.12 if side < 0 else 0.88
                img += self.base * (self.cloud * f * 2.2)[..., None] * (np.abs(np.arange(self.W)[None, :] / self.W - cxs) < 0.25)[..., None]
                r = np.random.default_rng(int(t0 * 100))
                n = 40
                yb = np.linspace(0.30 * self.H, self.cam.vy - 0.02 * self.H, n)
                xb = cxs * self.W + np.cumsum(r.normal(0, 0.012 * self.W, n))
                L.tube(xb, yb, np.array([230, 190, 255], np.float32), 1.6 * f, core=1.0, halo=1.0, bloom=1.5)

    # ---- the frame ---------------------------------------------------------------------------------
    def render(self, t):
        t = t % T_END
        img = self.base.copy()
        self.draw_sun(img, t)
        self.draw_sky(img, t)
        if not hasattr(self, "wire_all"):
            self.wire_all = tuple(np.concatenate(a) for a in zip(*[(x, y, np.broadcast_to(np.asarray(col, np.float32), (len(x), 3)),
                                                                     np.full(len(x), k)) for x, y, col, k in self.wire]))
        wx, wy, wc, wk = self.wire_all                                        # mountains' wireframe, softly pulsing
        self.L.tube(wx, wy, wc, wk * (0.75 + 0.25 * math.sin(TAU * 2 * t / T_END)), core=0.5, halo=0.7, bloom=0.3)
        self.draw_ground(t)
        dots = []
        self.draw_palms(dots, t)
        self.draw_car(dots, t)
        paint(img, dots)
        self.L.draw(img, 2)
        img[::2] *= 0.93                                                      # a hint of old video
        return finish(img, self.lut)


if __name__ == "__main__":
    run(SynthwaveDrive, "Synthwave drive", "synthwave_drive", T_END, STAGES, wrap=True)
