"""
aurora.py - the northern lights over a frozen lake, in dots.

A still, bitter night. A cabin on the shore has one lit window and a thread of chimney smoke;
the lake is black ice dusted with snow, and the stars turn slowly overhead. A faint green arc
lifts over the mountains, brightens, and folds into a curtain whose rays ripple and pleat, with
violet along its top. A second curtain joins it and everything below - the snow, the ice, the
cabin roof - takes on their green. Snow blows across the ice. Then the sky lets go: the curtain
surges overhead, its lower edge burns pink and rays pour down from straight above in a crown,
before it all breaks into soft patches that pulse on and off as the night settles.

    python aurora.py            interactive window (Space pause, R restart, S save frame, Esc quit)
    python aurora.py --png      save five stages
    python aurora.py --gif      save the whole night as a GIF
"""
import math

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))   # the shared engine: engine/pointillism.py

from pointillism import Haze, glow, keyed, run, shades, sky, ss, stamp

T_END = 60.0
STAGES = [5.0, 21.0, 33.0, 50.6, 57.5]
TAU = 2 * math.pi

GREEN = np.array([60, 255, 130], np.float32)
VIOLET = np.array([140, 70, 210], np.float32)
PINK = np.array([255, 70, 150], np.float32)
WARM = np.array([255, 178, 92], np.float32)
SNOW = [(150, 160, 186), (170, 180, 204), (194, 202, 222), (132, 144, 172)]
ROCK = [(22, 26, 38), (28, 32, 46), (34, 38, 54)]
ICE = [(10, 14, 24), (14, 20, 32), (18, 26, 40)]
SPRUCE = [(8, 12, 16), (10, 16, 20), (14, 20, 24)]
LOGS = [(42, 30, 24), (52, 38, 30), (36, 26, 22)]
HORIZON = 0.615

LEVEL = [(0, 0.12), (6, 0.22), (14, 0.65), (22, 0.85), (32, 1.0), (39, 0.6), (45, 0.65), (48.5, 1.4),
         (52, 1.25), (54.5, 0.35), (60, 0.30)]
CURTAINS = [  # left end, span, base height, ray length, arch, fold frequencies, fold speeds, fold sizes, when
    dict(x0=-0.12, span=1.30, yb=0.40, h=0.24, arch=0.12, f=(1.1, 2.6), w=(0.22, -0.31), A=(0.085, 0.030),
         keys=[(0, 0.5), (6, 1.0), (40, 1.0), (46, 1.0), (54, 0.4), (60, 0.2)], lean=-0.06, ph=0.0),
    dict(x0=0.06, span=0.95, yb=0.30, h=0.20, arch=-0.06, f=(1.0, 2.4), w=(-0.18, 0.27), A=(0.045, 0.022),
         keys=[(0, 0.0), (18, 0.0), (24, 1.0), (38, 0.8), (44, 0.3), (54, 0.2), (60, 0.0)], lean=0.05, ph=2.0),
    dict(x0=-0.05, span=1.10, yb=0.50, h=0.10, arch=0.04, f=(1.7, 3.7), w=(0.30, -0.22), A=(0.030, 0.015),
         keys=[(0, 0.3), (10, 0.5), (30, 0.8), (46, 0.5), (55, 0.1), (60, 0.0)], lean=0.0, ph=4.0),
]


class Aurora:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.S = min(W, H)
        self.rng = np.random.default_rng(13)
        self.hz = int(HORIZON * H)
        k = max(2, round(4 * self.S / 1000))
        self.sky_light = Haze(W, H, k)
        self.warm = Haze(W, H, k)
        self.build_land()
        self.build_curtains()
        self.build_stars()
        self.build_motion()

    # ---- everything that stays put ----------------------------------------------------------------
    def build_land(self):
        W, H, S = self.W, self.H, self.S
        rng = np.random.default_rng(3)
        img = sky(W, H, rng, [(0.0, (3, 5, 14)), (0.35, (6, 10, 24)), (0.58, (12, 18, 36)), (0.62, (16, 24, 42))],
                  self.hz + 2, 0.22)
        snow = np.zeros((H, W, 3), np.float32)                   # how much each pixel is snow (catches the light)
        sky_m = np.ones((H, W, 3), np.float32)
        xs = np.arange(W, dtype=np.float32)
        u = xs / W
        yy = np.arange(H, dtype=np.float32)[:, None]
        # mountains across the far shore
        prof = np.zeros(W)
        prng = np.random.default_rng(8)
        for c, w, h in ((0.12, 0.13, 0.10), (0.46, 0.11, 0.12), (0.86, 0.12, 0.09)):
            prof = np.maximum(prof, h * np.clip(1 - np.abs(u - c) / w, 0, 1) ** 1.1)
        for _ in range(14):
            c, w, h = prng.uniform(-0.05, 1.05), prng.uniform(0.04, 0.11), prng.uniform(0.02, 0.08)
            prof = np.maximum(prof, h * np.clip(1 - np.abs(u - c) / w, 0, 1) ** prng.uniform(0.9, 1.6))
        prof += 0.004 * np.sin(u * 90) + 0.003 * np.sin(u * 230 + 1)
        mtop = H * (HORIZON - 0.012 - prof)
        self.mtop = mtop
        fill = (yy >= mtop[None, :]) & (yy < self.hz)
        img[fill] = np.asarray(ROCK[1], np.float32) * 0.9
        sky_m[fill] = 0
        n = int(W * H * 0.10)
        x = rng.uniform(0, W, n)
        top = np.interp(x, xs, mtop)
        y = top + rng.random(n) * np.maximum(self.hz - top, 0)
        rel = (y - top) / np.maximum(H * HORIZON - top, 1)
        gully = np.sin(x / W * 140 + np.sin(x / W * 31) * 2) * 0.5 + 0.5
        sn = (rel < 0.55 + 0.35 * gully - 0.3 * (prof[np.clip(x.astype(int), 0, W - 1)] < 0.05)) & (rng.random(n) < 0.9)
        col = np.where(sn[:, None], shades(rng, SNOW, rng.random(n)) * (0.24 + 0.10 * (1 - rel))[:, None],
                       shades(rng, ROCK, rng.random(n)) * 0.8)
        stamp(img, x, y, col, rng.choice([1.0, 1.5], n))
        stamp(snow, x[sn], y[sn], np.array([0.30], np.float32), 1.5)
        # a ragged band of spruce along the far shore
        tt = np.zeros(W)
        for _ in range(int(260 * W / 1000)):
            c, w, h = rng.uniform(0, 1), rng.uniform(0.002, 0.005), rng.uniform(0.008, 0.026)
            tt = np.maximum(tt, h * np.clip(1 - np.abs(u - c) / w, 0, 1))
        ttop = H * (HORIZON + 0.004 - tt)
        fill = (yy >= ttop[None, :]) & (yy < self.hz + 2)
        img[fill] = np.asarray(SPRUCE[1], np.float32)
        sky_m[fill] = 0
        # the lake: black ice, drifts of snow, cracks
        ice = (yy >= self.hz + 2) & np.ones((1, W), bool)
        depth = np.clip((yy - self.hz) / (H - self.hz), 0, 1)
        icecol = np.asarray(ICE[2], np.float32) * (1 - depth[..., None]) + np.asarray(ICE[0], np.float32) * depth[..., None]
        img[:] = np.where(ice[..., None], icecol, img)
        n = int(W * (H - self.hz) * 0.25)
        x, y = rng.uniform(0, W, n), rng.uniform(self.hz + 2, H, n)
        stamp(img, x, y, shades(rng, ICE, rng.random(n), 1.0, 0.1) * (1.2 - 0.5 * (y - self.hz)[:, None] / (H - self.hz)),
              1.0)
        d = (y - self.hz) / (H - self.hz)
        streak = np.sin(7.0 / (d + 0.12) + 2.5 * np.sin(x / W * 5 + d * 6))
        patchy = np.sin(x / W * 4.3 + 1) * np.sin(d * 7 + x / W * 2)
        drift = 0.55 * streak + 0.75 * patchy + 0.35 * d
        on = (drift > 0.55) & (rng.random(n) < 0.8)
        m = int(on.sum())
        sx = x[on] + rng.normal(0, 1, m) * (3 + 14 * d[on]) * S / 1000
        col = shades(rng, SNOW, rng.random(m)) * (0.16 + 0.24 * d[on])[:, None]
        stamp(img, sx, y[on], col, 1.0 + 0.8 * (d[on] > 0.5))
        stamp(snow, sx, y[on], (0.35 + 0.4 * d[on])[:, None].astype(np.float32), 1.5)
        for _ in range(26):                                           # cracks, mostly running across
            m = 160
            a = rng.normal(0, 0.25) + rng.choice([0.0, math.pi])
            st = rng.uniform(0, W), rng.uniform(self.hz + 5, H)
            ang = a + np.cumsum(rng.normal(0, 0.05, m))
            dd = (st[1] - self.hz) / (H - self.hz)
            step = (1.2 + 3 * dd) * S / 1000
            cx = st[0] + np.cumsum(np.cos(ang) * step)
            cy = st[1] + np.cumsum(np.sin(ang) * step * 0.25)
            keep = rng.random(m) < 0.8
            stamp(img, cx[keep], cy[keep], np.array([30, 40, 58], np.float32) * (0.7 + 0.4 * rng.random((int(keep.sum()), 1))),
                  0.8)
        # the near shore on the right, where the cabin stands
        sh = H * (0.655 + 0.04 * np.clip((u - 0.60) / 0.4, 0, 1) ** 1.3)
        sp = np.clip((u - 0.58) / 0.42, 0, 1)
        top = sh - 0.012 * H * np.sin(sp * np.pi)
        sb = sh + H * 0.05 * sp ** 0.8
        land = (xs[None, :] > 0.58 * W) & (yy >= top[None, :]) & (yy < sb[None, :])
        self.land_top = sh
        n = int(land.sum() * 0.5)
        ly, lx = np.nonzero(land)
        pick = rng.integers(0, len(lx), n)
        img[land] = np.asarray(SNOW[0], np.float32) * 0.36
        face = np.clip((ly[pick] - np.interp(lx[pick], xs, sh)) / np.maximum(np.interp(lx[pick], xs, sb - sh), 1), 0, 1)
        stamp(img, lx[pick] + rng.random(n), ly[pick] + rng.random(n),
              shades(rng, SNOW, rng.random(n)) * (0.44 - 0.2 * face)[:, None], 1.5)
        snow[land] = 0.8
        sky_m[land] = 0
        # spruces round the cabin
        for tx, tb, th in ((0.645, 0.668, 0.12), (0.700, 0.676, 0.17), (0.915, 0.700, 0.21), (0.975, 0.706, 0.16),
                           (0.875, 0.694, 0.10)):
            self.spruce(img, snow, sky_m, rng, tx * W, tb * H, th * H)
        # a snowbank across the bottom left, with dry reeds (they move, so they are drawn each frame)
        bank = H * (0.90 + 0.12 * (u / 0.55) ** 1.8)
        bk = (yy >= bank[None, :]) & (xs[None, :] < 0.62 * W)
        img[bk] = np.asarray(SNOW[1], np.float32) * 0.42
        ly, lx = np.nonzero(bk)
        n = int(len(lx) * 0.6)
        pick = rng.integers(0, len(lx), n)
        rim = np.clip(1 - (ly[pick] - np.interp(lx[pick], xs, bank)) / (0.03 * H), 0, 1)
        stamp(img, lx[pick] + rng.random(n), ly[pick] + rng.random(n),
              shades(rng, SNOW, rng.random(n)) * (0.40 + 0.18 * rim)[:, None], 1.5)
        snow[bk] = 0.9
        sky_m[bk] = 0
        self.bank = bank
        self.build_cabin(img, snow, sky_m, rng)
        snow, sky_m = snow[..., 0].copy(), sky_m[..., 0].copy()
        # where the aurora shows: the open sky, and as a reflection on bare ice
        refl = (yy >= self.hz + 2).astype(np.float32) * np.ones((1, W), np.float32)
        refl *= 1 - np.clip(snow * 1.4, 0, 0.85)
        refl[(sky_m == 0) & (yy >= self.hz + 2)] = 0
        self.mask = np.where(yy < self.hz + 2, sky_m, refl).astype(np.float32)
        self.snow = snow
        self.snow_row = int(np.nonzero(snow.max(1) > 0)[0][0])
        self.open_rows = int(np.nonzero(self.mask.min(1) < 1)[0][0])
        k = self.sky_light.k
        bh, bw = self.sky_light.buf.shape[:2]
        pad = np.zeros((bh * k, bw * k), np.float32)
        pad[:H, :W] = self.mask
        pad[:self.hz + 2] = 0
        self.ice_mask = pad.reshape(bh, k, bw, k).mean((1, 3))
        self.base = img

    def spruce(self, img, snow, sky_m, rng, x0, y0, h):
        n = int(h * h * 0.9) + 200
        u = rng.random(n) ** 0.8                                   # 0 at the foot, 1 at the tip
        tier = (u * 9) % 1
        half = h * 0.24 * (1 - u) * (0.55 + 0.45 * tier) + 0.003 * h
        v = rng.uniform(-1, 1, n)
        x = x0 + v * half
        y = y0 - u * h + 0.03 * h * tier
        snowy = (rng.random(n) < 0.35) & (tier > 0.6) & (np.abs(v) < 0.9)
        col = np.where(snowy[:, None], shades(rng, SNOW, rng.random(n)) * 0.5, shades(rng, SPRUCE, rng.random(n)))
        stamp(img, x, y, col, 1.5)
        stamp(sky_m, x, y, np.zeros(1, np.float32), 1.5)
        stamp(snow, x[snowy], y[snowy], np.array([0.5], np.float32), 1.5)

    def build_cabin(self, img, snow, sky_m, rng):
        W, H = self.W, self.H
        cx, by = 0.80 * W, 0.688 * H
        w, h = 0.085 * W, 0.050 * H
        self.cabin = (cx, by, w, h)
        # log walls
        n = int(w * h * 1.4)
        x, y = rng.uniform(cx - w / 2, cx + w / 2, n), rng.uniform(by - h, by, n)
        logs = np.floor((by - y) / (h / 7))
        col = shades(rng, LOGS, rng.random(n)) * (0.8 + 0.3 * ((by - y) / (h / 7) % 1 > 0.25))[:, None]
        stamp(img, x, y, col, 1.5)
        # roof: a gable with deep snow
        rh = 0.045 * H
        n = int(w * rh * 2.0)
        x = rng.uniform(cx - w * 0.62, cx + w * 0.62, n)
        y = by - h - rng.random(n) * rh
        roof_top = by - h - rh * (1 - np.abs(x - cx) / (w * 0.62))
        keep = y >= roof_top
        x, y = x[keep], y[keep]
        depth = (y - roof_top[keep]) / (by - h - roof_top[keep] + 1e-3)
        col = np.where((depth < 0.55)[:, None], shades(rng, SNOW, rng.random(len(x))) * 0.55,
                       shades(rng, LOGS, rng.random(len(x))) * 0.7)
        stamp(img, x, y, col, 1.5)
        stamp(snow, x[depth < 0.55], y[depth < 0.55], np.array([1.0], np.float32), 1.5)
        # chimney
        chx, chy = cx + w * 0.28, by - h - rh * 0.55
        n = 300
        x, y = rng.uniform(chx - 0.006 * W, chx + 0.006 * W, n), rng.uniform(chy - 0.03 * H, chy, n)
        stamp(img, x, y, shades(rng, [(40, 36, 36), (52, 46, 44)], rng.random(n)), 1.5)
        stamp(img, rng.uniform(chx - 0.007 * W, chx + 0.007 * W, 60), chy - 0.03 * H - rng.random(60) * 0.004 * H,
              shades(rng, SNOW, rng.random(60)) * 0.55, 1.5)
        self.chimney = (chx, chy - 0.032 * H)
        # the window and the door
        wx, wy, ww, wh = cx - w * 0.18, by - h * 0.55, 0.022 * W, 0.016 * H
        n = int(ww * wh * 1.2)
        x, y = rng.uniform(wx - ww / 2, wx + ww / 2, n), rng.uniform(wy - wh / 2, wy + wh / 2, n)
        frame = (np.abs(x - wx) < 0.0015 * W) | (np.abs(y - wy) < 0.0012 * H)
        col = np.where(frame[:, None], np.array([40, 26, 18], np.float32), WARM * (0.85 + 0.2 * rng.random((n, 1))))
        stamp(img, x, y, col, 1.2)
        self.window = (wx, wy, ww, wh)
        dx, dw, dh = cx + w * 0.18, 0.016 * W, 0.032 * H
        n = int(dw * dh * 1.2)
        stamp(img, rng.uniform(dx - dw / 2, dx + dw / 2, n), rng.uniform(by - dh, by, n),
              shades(rng, [(30, 22, 18), (36, 26, 20)], rng.random(n)), 1.2)
        glow(img, rng.uniform(dx - dw / 2, dx + dw / 2, 80), by - 0.001 * H * rng.random(80), WARM * 0.25, 1.0)
        sky_m[int(by - h - rh - 0.04 * H):int(by) + 1, int(cx - w * 0.65):int(cx + w * 0.65)] = 0

    # ---- the lights ---------------------------------------------------------------------------------
    def build_curtains(self):
        rng = self.rng
        self.cur = []
        for C in CURTAINS:
            n = int(300 * self.W / 1000)
            self.cur.append(dict(s=(np.arange(n) + rng.uniform(0.1, 0.9, n)) / n, r=rng.random(n), ph=rng.uniform(0, TAU, n),
                                 fl=rng.uniform(4, 11, n)))
        m = 34
        self.ray_u = (np.arange(m) / (m - 1)) ** 1.5
        m = 46
        self.fine_u = (np.arange(m) / (m - 1)) ** 1.3
        n = 170                                                       # the crown, pouring down from overhead
        self.crown = dict(a=np.sort(rng.uniform(-0.15, math.pi + 0.15, n)), r=rng.random(n), ph=rng.uniform(0, TAU, n),
                          ln=rng.uniform(0.5, 1.0, n))
        k = 14                                                        # pulsing patches at the end
        self.patch = dict(x=rng.uniform(0.05, 0.95, k), y=rng.uniform(0.10, 0.45, k), r=rng.uniform(0.04, 0.08, k),
                          per=rng.uniform(2.5, 6.0, k), ph=rng.random(k))
        q = 90
        self.patch_off = rng.normal(0, 0.5, (2, q))

    def curtain_rays(self, i, t, level):
        """Base point, top point, brightness and the pinkness of each ray of curtain i."""
        W, H = self.W, self.H
        C, R = CURTAINS[i], self.cur[i]
        s = R["s"]
        storm = float(ss(t, 46.0, 49.0) * (1 - ss(t, 53.0, 56.0))) if i == 0 else 0.0
        speed = 1 + 2.5 * storm
        tt = t * speed
        x = C["x0"] + C["span"] * s + C["A"][0] * np.sin(TAU * C["f"][0] * s + C["w"][0] * tt + C["ph"]) \
            + C["A"][1] * np.sin(TAU * C["f"][1] * s + C["w"][1] * tt + 2 * C["ph"]) + 0.02 * math.sin(0.05 * t + C["ph"])
        yb = C["yb"] - 0.28 * storm + C["arch"] * (s - 0.5) ** 2 * 4 - C["arch"] + 0.012 * np.sin(TAU * 1.6 * s + 0.3 * tt)
        h = C["h"] * (1 + 0.9 * storm) * (0.75 + 0.25 * np.sin(TAU * 2.2 * s - 0.4 * tt + C["ph"]))
        env = np.sin(np.pi * np.clip(s, 0, 1)) ** 0.6
        wave = 0.45 + 0.55 * np.sin(TAU * 3.0 * s - (0.9 + 2 * storm) * t + C["ph"]) ** 2
        flick = 0.75 + 0.25 * np.sin(R["fl"] * t + R["ph"])
        b = level * float(keyed(t, C["keys"])) * env * wave * flick * (0.6 + 0.4 * R["r"])
        X0, Y0 = x * W, yb * H
        lean = C["lean"] + 0.25 * storm * (s - 0.5)
        X1, Y1 = X0 + lean * h * H, Y0 - h * H
        return X0, Y0, X1, Y1, b, storm

    def draw_aurora(self, img, t):
        W, H, S = self.W, self.H, self.S
        level = float(keyed(t, LEVEL))
        A = self.sky_light
        u = self.ray_u[None, :]
        pink_all = 0.0
        fine = []
        for i in range(len(CURTAINS)):
            X0, Y0, X1, Y1, b, storm = self.curtain_rays(i, t, level)
            on = b > 0.01
            if not on.any():
                continue
            X0, Y0, X1, Y1, b = X0[on], Y0[on], X1[on], Y1[on], b[on]
            x = X0[:, None] + (X1 - X0)[:, None] * u
            y = Y0[:, None] + (Y1 - Y0)[:, None] * u
            prof = np.exp(-u / 0.22) * 0.85 + 0.15 * (1 - u)                 # bright lower edge, fading up
            mixv = np.clip(u * 1.6 - 0.25, 0, 1) ** 0.8
            col = GREEN[None, None] * (1 - mixv[..., None]) + VIOLET[None, None] * mixv[..., None]
            col = col + PINK[None, None] * (storm * np.exp(-u / 0.06))[..., None] * 0.8
            pink_all = max(pink_all, storm)
            w = (b[:, None] * prof)[..., None] * col * 0.84
            A.add(x[::2].ravel(), y[::2].ravel(), w[::2].reshape(-1, 3), 6 * S / 1000)    # the soft glow
            j = slice(None, None, 2)                                            # every other ray, finely
            uf = self.fine_u[None, :]
            fx = X0[j, None] + (X1 - X0)[j, None] * uf
            fy = Y0[j, None] + (Y1 - Y0)[j, None] * uf
            pf = np.exp(-uf / 0.22) * 0.85 + 0.15 * (1 - uf)
            mf = np.clip(uf * 1.6 - 0.25, 0, 1) ** 0.8
            cf = GREEN[None, None] * (1 - mf[..., None]) + VIOLET[None, None] * mf[..., None] \
                + PINK[None, None] * (storm * np.exp(-uf / 0.06))[..., None] * 0.8
            wf = (b[j, None] * pf)[..., None] * cf * 0.22
            fine.append((fx.ravel(), fy.ravel(), wf.reshape(-1, 3)))
        cr = float(ss(t, 48.0, 50.0) * (1 - ss(t, 52.5, 55.0)))
        if cr > 0.01:                                                       # the crown
            K = self.crown
            cx, cy = 0.5 * W, -0.22 * H
            a = K["a"] + 0.05 * np.sin(0.7 * t + K["ph"])
            r0, r1 = 0.26 * H, (0.26 + 0.36 * K["ln"]) * H
            b = cr * level * (0.35 + 0.65 * np.sin(9 * t + K["ph"]) ** 2) * (0.3 + 0.7 * K["r"]) * 0.7
            r = r0 + (r1 - r0)[:, None] * u
            x = cx + np.cos(a)[:, None] * r
            y = cy + np.sin(a)[:, None] * r
            prof = (0.3 + 0.7 * u ** 1.2) * (1 - u ** 4)
            col = VIOLET[None, None] * (1 - u[..., None]) + GREEN[None, None] * u[..., None]
            w = (b[:, None] * prof)[..., None] * col * 0.80
            A.add(x[::2].ravel(), y[::2].ravel(), w[::2].reshape(-1, 3), 7 * S / 1000)
            uf = self.fine_u[None, :]
            rf = r0 + (r1 - r0)[::2, None] * uf
            fine.append(((cx + np.cos(a)[::2, None] * rf).ravel(), (cy + np.sin(a)[::2, None] * rf).ravel(),
                         ((b[::2, None] * (0.3 + 0.7 * uf ** 1.2) * (1 - uf ** 4))[..., None]
                          * (VIOLET * (1 - uf[..., None]) + GREEN * uf[..., None]) * 0.2).reshape(-1, 3)))
        pt = float(ss(t, 53.5, 56.0))
        if pt > 0.01:                                                       # pulsing patches
            P = self.patch
            on = (np.sin(TAU * (t / P["per"] + P["ph"])) * 0.5 + 0.5) ** 3 * pt
            x = (P["x"][:, None] + P["r"][:, None] * self.patch_off[0][None, :] * 1.4) * W
            y = (P["y"][:, None] + P["r"][:, None] * self.patch_off[1][None, :] * 0.6) * H
            w = on[:, None, None] * GREEN[None, None] * 0.10 * np.ones((1, x.shape[1], 1))
            A.add(x.ravel(), y.ravel(), w.reshape(-1, 3), 10 * S / 1000)
        # mirror the sky light into the ice below the horizon, softened
        buf, k = A.buf, A.k
        hr = int(round((self.hz - (k - 1) / 2) / k))
        rows = np.arange(hr + 1, buf.shape[0])
        src = 2 * hr - rows
        ok = src >= 1
        rows, src = rows[ok], src[ok]
        buf[rows] += 0.42 * (buf[src] * 0.5 + buf[src - 1] * 0.25 + buf[np.minimum(src + 1, hr)] * 0.25)             * self.ice_mask[rows, :, None]                                   # (the ice mask, at the grid's scale)
        A.apply(img, 1.0, 2, self.mask, self.open_rows, self.hz + 2)
        if fine:                                                            # fine ray streaks in the open sky
            x, y, w = (np.concatenate(a) for a in zip(*fine))
            xi = np.clip(x.astype(int), 0, W - 1)
            yi = np.clip(y.astype(int), 0, H - 1)
            m = (self.mask[yi, xi] > 0.5) & (y < self.hz) & (w.max(1) > 0.4)
            glow(img, x[m], y[m], w[m], 1.0)
        return level, pink_all

    # ---- stars, smoke, snow, reeds ------------------------------------------------------------------------
    def build_stars(self):
        rng, W, H = self.rng, self.W, self.H
        n = int(1700 * W * H / 1e6)
        self.st = dict(x=rng.uniform(-0.2, 1.2, n) * W, y=rng.uniform(-0.3, HORIZON, n) * H,
                       b=rng.random(n) ** 3.5, ph=rng.uniform(0, TAU, n), f=rng.uniform(1.5, 5, n))
        self.pole = (0.12 * W, -0.55 * H)

    def draw_stars(self, img, t):
        S = self.st
        W, H = self.W, self.H
        a = -0.035 * t / T_END
        px, py = self.pole
        c, s = math.cos(a), math.sin(a)
        x = px + (S["x"] - px) * c - (S["y"] - py) * s
        y = py + (S["x"] - px) * s + (S["y"] - py) * c
        tw = 0.75 + 0.25 * np.sin(S["f"] * t + S["ph"])
        b = (S["b"] * 220 + 18) * tw
        xi, yi = np.clip(x.astype(int), 0, W - 1), np.clip(y.astype(int), 0, H - 1)
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < self.hz) & (self.mask[yi, xi] > 0.5)
        col = np.array([0.88, 0.92, 1.0])[None, :] * b[:, None]
        glow(img, x[ok], y[ok], col[ok], 1.0 + 0.6 * (S["b"][ok] > 0.5))
        bright = ok & (S["b"] > 0.25)                                       # the brightest, faint in the ice
        ry = 2 * self.hz - y[bright]
        ryi = np.clip(ry.astype(int), 0, H - 1)
        rxi = xi[bright]
        m = (ry < H) & (self.mask[ryi, rxi] > 0.3)
        glow(img, x[bright][m], ry[m], col[bright][m] * 0.25, 1.0)

    def build_motion(self):
        rng, W, H = self.rng, self.W, self.H
        n = 260                                                             # chimney smoke
        self.smoke = dict(t0=rng.random(n) * 9.0, sp=rng.normal(0, 1, (2, n)), r=rng.random(n))
        n = 420                                                             # snow skating over the ice
        self.drift = dict(x=rng.random(n), y=self.hz + 2 + rng.random(n) ** 0.8 * (H - self.hz - 2),
                          v=rng.uniform(0.6, 1.4, n), ph=rng.uniform(0, TAU, n))
        k = 46                                                              # dry reeds in the snowbank
        rx = rng.uniform(0.01, 0.55, k) * W
        self.reeds = dict(x=rx, y=np.interp(rx, np.arange(W), self.bank) + 0.006 * H,
                          h=rng.uniform(0.03, 0.11, k) * H, lean=rng.normal(0, 0.15, k), ph=rng.uniform(0, TAU, k),
                          col=shades(rng, [(60, 50, 40), (76, 64, 48), (48, 40, 34)], rng.random(k)))

    def wind(self, t):
        return 0.5 + 0.5 * math.sin(0.35 * t) ** 2 + 0.6 * float(ss(t, 24, 28) * (1 - ss(t, 36, 42)))

    def draw_smoke(self, img, t, light):
        W, H = self.W, self.H
        Sm = self.smoke
        age = (t - Sm["t0"]) % 9.0
        f = age / 9.0
        cx, cy = self.chimney
        x = cx - 0.07 * W * f ** 1.4 * self.wind(t) + Sm["sp"][0] * (0.002 + 0.02 * f) * W + 0.006 * W * np.sin(2 * f * TAU + Sm["r"] * 6)
        y = cy - 0.10 * H * (1 - (1 - f) ** 1.8) + Sm["sp"][1] * (0.001 + 0.010 * f) * H
        b = np.sin(np.pi * np.clip(f, 0, 1)) ** 0.8 * (1 - f) * (0.5 + 0.5 * Sm["r"])
        col = (np.array([20, 22, 28]) + light * 0.12)[None, :] * b[:, None]
        glow(img, x, y, col, 2.0)

    def draw_drift(self, img, t):
        W, H = self.W, self.H
        D = self.drift
        g = self.wind(t)
        x = (D["x"] * 1.3 * W + 0.09 * W * D["v"] * (t + 0.6 * math.sin(0.35 * t))) % (1.3 * W) - 0.15 * W
        d = (D["y"] - self.hz) / (H - self.hz)
        y = D["y"] - 0.004 * H * np.abs(np.sin(3 * t + D["ph"]))
        b = (0.25 + 0.75 * np.sin(0.8 * t + D["ph"]) ** 2) * g * (0.4 + 0.8 * d)
        ln = (2 + 8 * d) * g * self.S / 1000
        j = np.arange(3)[:, None]
        X, Y = (x[None, :] - 1.3 * j * ln[None, :]).ravel(), np.broadcast_to(y, (3, len(y))).ravel()
        C = (np.array([60, 66, 80])[None, None, :] * (b[None, :] * (1 - j / 3))[..., None] * 0.6).reshape(-1, 3)
        glow(img, X, Y, C, np.tile(1.0 + 0.5 * (d > 0.5), 3))

    def draw_reeds(self, img, t, light):
        R = self.reeds
        g = self.wind(t)
        m = 40
        s = np.linspace(0, 1, m)[None, :]
        sw = (0.06 + 0.10 * g) * np.sin(1.7 * t + R["ph"])[:, None] + 0.10 * g
        x = R["x"][:, None] + R["h"][:, None] * (R["lean"][:, None] * s + sw * s * s)
        y = R["y"][:, None] - R["h"][:, None] * s * (1 - 0.1 * s)
        col = np.repeat(R["col"][:, None, :], m, 1) * 0.6 + light * 0.10
        stamp(img, x.ravel(), y.ravel(), col.reshape(-1, 3), np.tile(1.6 - 0.8 * s[0], len(R["x"])) * self.S / 1000 + 0.4)

    # ---- the frame ---------------------------------------------------------------------------------
    def render(self, t):
        img = self.base.copy()
        self.draw_stars(img, t)
        level, pink = self.draw_aurora(img, t)
        light = (GREEN * (1 - pink) + PINK * pink * 0.8) * min(level, 1.6)
        r0 = self.snow_row                                                    # the snow takes the colour of the sky
        img[r0:] += self.snow[r0:, :, None] * (light * 0.075).astype(np.float32)
        wx, wy, ww, wh = self.window                                          # the lit window, and its light on the snow
        flick = 0.92 + 0.08 * math.sin(7 * t) * math.sin(3.1 * t)
        self.warm.add(np.array([wx, wx, wx]), np.array([wy, wy + 0.02 * self.H, wy + 0.035 * self.H]),
                      np.array([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.3, 0.3, 0.3]]) * WARM * 0.20 * flick, 30 * self.S / 1000)
        rx = wx + np.random.default_rng(5).normal(0, 0.004, 24) * self.W
        ry = np.linspace(0.715, 0.80, 24) * self.H
        self.warm.add(rx, ry, WARM * 0.05 * flick, 12 * self.S / 1000)
        self.warm.apply(img, 1.0)
        self.draw_smoke(img, t, light)
        self.draw_drift(img, t)
        self.draw_reeds(img, t, light)
        return np.clip(img, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    run(Aurora, "Northern lights", "aurora", T_END, STAGES)
