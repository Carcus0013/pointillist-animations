"""
glow_beach.py - a beach at night where the waves glow, seen from above. A seamless one-minute loop.

There is no moon. Out on the black water a swell line comes in, steepens, and as it breaks
the plankton in it light up: a seam of electric blue peels along the wave, the white water
fizzes with sparks, and the sheet of water slides up the sand with a glowing lip before it
drains back, leaving the wet sand twinkling. The wet sand is a mirror for the Milky Way.
Fish dart through the shallows drawing streaks of light. Someone walks the waterline with a
dog that keeps charging into the surf; every footprint flashes blue as it is pressed into
the sand, and the next wave washes the prints away.

    python glow_beach.py            the beach, looping forever (Space pause, R restart, S save frame, Esc quit)
    python glow_beach.py --png      save five moments
    python glow_beach.py --gif      save one seamless loop as a GIF
"""
import math

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))   # the shared engine: engine/pointillism.py

from pointillism import Haze, glow, run, shades, ss, stamp, unit

T_END = 60.0
TAU = 2 * math.pi
STAGES = [8.9, 11.6, 19.6, 30.4, 40.6]

SAND = [(44, 44, 54), (52, 52, 62), (60, 60, 70), (40, 42, 52)]
SAND_WET = [(18, 22, 34), (22, 26, 40), (28, 32, 46)]
WATER = [(8, 14, 30), (10, 18, 36), (14, 22, 42)]
WRACK = [(30, 30, 30), (38, 36, 32), (26, 30, 28)]
BLUE = np.array([40, 150, 255], np.float32)
CORE = np.array([170, 235, 255], np.float32)

PERIOD = 10.0                 # a wave every ten seconds: six to the minute, so the minute loops
TB = 7.0                      # seconds from first sight to breaking
LIFE = TB + 10.0
WAVES = [  # strength (run-up), tilt (which end breaks first), its own wobble
    (1.00, 0.9, 0.3), (0.80, -0.6, 1.9), (1.15, 0.5, 3.1), (0.75, 1.2, 4.4), (1.05, -0.9, 5.2), (0.90, 0.3, 0.8),
]
WALK = (8.0, 46.0)            # the walker crosses in these seconds
FISH = [  # loop centre, size (fractions of the frame), laps a minute, sprints a minute, phase
    (0.30, 0.30, 0.18, 0.10, 3, 7, 0.0), (0.68, 0.36, 0.22, 0.12, -2, 5, 2.0), (0.52, 0.18, 0.30, 0.08, 2, 9, 4.1),
    (0.18, 0.44, 0.12, 0.05, -4, 6, 1.3),
]


def ease(p):
    p = np.clip(p, 0, 1)
    return p * p * (3 - 2 * p)


class GlowBeach:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.S = min(W, H)
        self.rng = np.random.default_rng(17)
        self.haze = Haze(W, H, max(2, round(4 * self.S / 1000)))
        self.xs = np.arange(W, dtype=np.float32)
        u = self.xs / W
        self.y0 = H * (0.60 + 0.07 * (u - 0.5) + 0.02 * np.sin(u * TAU * 0.7 + 0.3))   # where the sea stops
        self.base = self.y0 - 0.03 * H
        self.brk = self.y0 - 0.23 * H - 0.025 * H * np.sin(u * TAU * 1.3 + 1.0)
        self.build_images()
        self.build_waves()
        self.build_walk()

    # ---- the still parts: sand, wet sand, water, the sky in both ----------------------------------
    def milky_way(self, rng, n_scale=1.0):
        """Stars and the Milky Way as (x, y, brightness) for the reflections."""
        W, H = self.W, self.H
        a = math.radians(-32)
        d = np.array([math.cos(a), math.sin(a)])
        nrm = np.array([-d[1], d[0]])
        n = int(26000 * W * H / 1e6 * n_scale)
        along = rng.uniform(-0.8, 0.8, n) * 1.6 * self.S
        across = rng.normal(0, 0.07, n) * self.S * (1 + 0.5 * np.sin(along / self.S * 3))
        lane = 1 - 0.85 * np.exp(-((across / self.S - 0.012 * np.sin(along / self.S * 5)) / 0.018) ** 2) \
            * (0.5 + 0.5 * np.sin(along / self.S * 2.3 + 1))
        x = 0.5 * W + along * d[0] + across * nrm[0]
        y = 0.55 * H + along * d[1] + across * nrm[1]
        b = rng.random(n) ** 3 * lane * 70 + 8 * lane
        m = int(2200 * W * H / 1e6)                              # and the rest of the sky
        x = np.concatenate([x, rng.uniform(0, W, m)])
        y = np.concatenate([y, rng.uniform(0, H, m)])
        b = np.concatenate([b, rng.random(m) ** 4 * 160 + 10])
        return x, y, b, (d, nrm)

    def build_images(self):
        W, H, rng = self.W, self.H, np.random.default_rng(5)
        yy = np.arange(H, dtype=np.float32)[:, None]
        sx, sy, sb, (d, nrm) = self.milky_way(rng)
        tint = np.array([0.80, 0.86, 1.0], np.float32)
        # dry and wet sand
        dry = np.empty((H, W, 3), np.float32)
        dry[:] = np.asarray(SAND[1], np.float32) * 0.85
        n = int(W * H * 0.55)
        x, y = rng.uniform(0, W, n), rng.uniform(0, H, n)
        stamp(dry, x, y, shades(rng, SAND, rng.random(n), 1.0, 0.08), rng.choice([1.0, 1.5], n))
        wet = np.empty((H, W, 3), np.float32)
        wet[:] = np.asarray(SAND_WET[1], np.float32) * 0.9
        stamp(wet, x, y, shades(rng, SAND_WET, rng.random(n), 1.0, 0.06), rng.choice([1.0, 1.5], n))
        # the band of the sky on the wet sand: a soft glow plus the stars, clear as a mirror
        px = (np.arange(W)[None, :] - 0.5 * W) * nrm[0] + (yy - 0.55 * H) * nrm[1]
        band = np.exp(-(px / (0.10 * self.S)) ** 2)[..., None] * np.array([22, 22, 30], np.float32)
        wet += band
        glow(wet, sx, sy, (sb[:, None] * tint * 0.30).astype(np.float32), 1.0)
        # wetness fades out above the run-up line; a line of weed and shells marks high water
        wl = self.y0 + 0.095 * H + 0.012 * H * np.sin(self.xs / W * TAU * 1.7)
        a = np.clip((wl[None, :] - yy) / (0.05 * H), 0, 1)[..., None] ** 0.7
        beach = dry + (wet - dry) * a
        hw = self.y0 + 0.22 * H + 0.02 * H * np.sin(self.xs / W * TAU * 1.1 + 2)
        n = int(W * 0.9)
        x = rng.uniform(0, W, n)
        y = np.interp(x, self.xs, hw) + rng.normal(0, 0.005 * H, n) * (1 + 0.8 * np.sin(x / W * 23))
        stamp(beach, x, y, shades(rng, WRACK, rng.random(n)), rng.choice([1.0, 1.5, 2.0], n))
        n = int(W * 0.06)
        x = rng.uniform(0, W, n)
        y = np.interp(x, self.xs, hw) + rng.normal(0, 0.03 * H, n)
        stamp(beach, x, y, shades(rng, [(120, 118, 120), (140, 136, 132)], rng.random(n)), 1.5)
        self.beach = beach
        # the water: black-blue, the sky in it broken into ripples
        water = np.empty((H, W, 3), np.float32)
        water[:] = np.asarray(WATER[1], np.float32) * 0.9
        n = int(W * H * 0.30)
        x, y = rng.uniform(0, W, n), rng.uniform(0, H, n)
        stamp(water, x, y, shades(rng, WATER, rng.random(n), 1.0, 0.1), rng.choice([1.0, 1.5], n))
        water += band * 0.8
        jx = sx + 0.004 * W * np.sin(sy / H * 160)
        glow(water, jx, sy, (sb[:, None] * tint * 0.36).astype(np.float32), 1.0)
        self.water = water
        n = int(5000 * W * H / 1e6)                       # ripples that catch the starlight
        self.glint = dict(x=rng.uniform(0, W, n), y=rng.uniform(-0.02, 0.6, n) * H, ph=rng.uniform(0, TAU, n),
                          f=rng.integers(3, 9, n))

    # ---- waves -----------------------------------------------------------------------------------
    def build_waves(self):
        rng, W, H = self.rng, self.W, self.H
        self.wv = []
        for j, (power, tilt, wob) in enumerate(WAVES):
            u = self.xs / W
            run_ = self.y0 + H * power * (0.055 + 0.018 * np.sin(u * TAU * 1.4 + wob) + 0.01 * np.sin(u * TAU * 3.3 + wob))
            n = int(1500 * W / 1000)
            fx = rng.uniform(0, W, n)
            n2 = int(4200 * W / 1000)
            m = int(2600 * W / 1000)
            self.wv.append(dict(
                start=1.5 + j * PERIOD, tilt=tilt, run=run_.astype(np.float32),
                cx=fx, cj=rng.normal(0, 1, n), cn=rng.random(n),                   # the crest
                fx=rng.uniform(0, W, n2), fv=rng.random(n2) ** 1.3, fn=rng.random(n2),  # white water
                fph=rng.uniform(0, TAU, n2),
                sx=rng.uniform(0, W, m), sv=rng.random(m), sd=rng.uniform(0, 1.6, m),   # sparks left on the sand
            ))

    def ages(self, t, x):
        """Age of every wave at the columns x (each end of a wave arrives at its own time)."""
        out = []
        for w in self.wv:
            a = (t - w["start"]) % T_END
            out.append(a - w["tilt"] * (np.asarray(x) / self.W - 0.5))
        return out

    def front(self, w, x, a):
        """Leading edge of a wave's water at columns x (indices into the frame), age a.
        Returns y, and 'on' where the wave is alive."""
        xi = np.clip(np.asarray(x).astype(int), 0, self.W - 1)
        brk, base, run_ = self.brk[xi], self.base[xi], w["run"][xi]
        top = -0.08 * self.H
        y = np.where(a < TB, top + (brk - top) * (np.clip(a, 0, TB) / TB) ** 1.3,
            np.where(a < TB + 2.0, brk + (base - brk) * ease((a - TB) / 2.0),
            np.where(a < TB + 4.5, base + (run_ - base) * np.sin(np.clip((a - TB - 2.0) / 2.5, 0, 1) * np.pi / 2),
                     run_ + (base - run_) * ease((a - TB - 4.5) / 4.5))))
        return y, (a > 0) & (a < TB + 9.0)

    def water_edge(self, t):
        E = self.base + 0.004 * self.H * np.sin(self.xs / self.W * 9 + TAU * 8 * t / T_END)
        for w, a in zip(self.wv, self.ages(t, self.xs)):
            y, on = self.front(w, self.xs, a)
            up = on & (a > TB + 2.0)
            E = np.where(up, np.maximum(E, y), E)
        return E

    def draw_waves(self, img, t, E):
        W, H, S = self.W, self.H, self.S
        hz = self.haze
        for w in self.wv:
            a0 = (t - w["start"]) % T_END
            if a0 > LIFE + 1.5:
                continue
            # the crest: a dim line offshore, a peeling seam of light as it breaks
            x = w["cx"]
            a = a0 - w["tilt"] * (x / W - 0.5)
            y, on = self.front(w, x, a)
            k = np.where(a < TB, 0.07 * ease((a - 2.0) / (TB - 2.0)) + 0.35 * ease((a - TB + 1.3) / 1.3),
                np.where(a < TB + 2.0, 0.45 + 0.55 * np.exp(-np.clip(a - TB, 0, None) / 1.0),
                np.where(a < TB + 4.5, 0.50 * np.exp(-(a - TB - 2.0) / 2.6), 0.16 * (1 - ease((a - TB - 4.5) / 4.5)))))
            patch = 0.5 + 0.5 * np.sin(x / W * TAU * 3 + 2 * w["tilt"] + 0.35 * a) ** 2
            bar = np.clip(np.sin(x / W * TAU * 2.2 + 4 * w["tilt"]) * 1.6 - 0.3, 0, 1)        # an outer sandbar
            k = k + 0.42 * np.exp(-((a - 3.6) / 0.7) ** 2) * bar
            lace = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(x / W * TAU * 11 + 3 * w["tilt"]) * np.sin(x / W * TAU * 4.3 + a)) ** 1.5
            k = k * (0.6 + 0.4 * w["cn"]) * on * np.where(a < TB + 2.0, patch, lace)
            yj = y + w["cj"] * (0.6 + 1.6 * (a > TB)) * S / 1000
            m = k > 0.01
            if m.any():
                glow(img, x[m], yj[m], CORE * k[m, None], 1.0 + 0.6 * (k[m] > 0.4))
                hz.add(x[m][::2], yj[m][::2], BLUE * (0.35 * k[m][::2])[:, None], 14 * S / 1000)
            # white water: sparks in a band behind the front, fizzing and fading
            x = w["fx"]
            a = a0 - w["tilt"] * (x / W - 0.5)
            y, on = self.front(w, x, a)
            since = a - TB
            width = H * np.where(since < 2.0, 0.015 + 0.05 * ease(since / 2.0), 0.065 - 0.04 * ease((since - 2.0) / 7.0))
            fy = y - w["fv"] * width
            flick = 0.55 + 0.45 * np.sin(TAU * 86 * t / T_END + w["fph"] * 5) ** 2
            kf = np.exp(-np.clip(since, 0, None) / 3.4) * (since > 0) * on * flick * (1 - w["fv"]) ** 0.6
            kf = kf * (0.3 + 0.7 * w["fn"])
            m = kf > 0.02
            if m.any():
                glow(img, x[m], fy[m], BLUE * 1.3 * kf[m, None], 1.0 + 0.5 * (w["fn"][m] > 0.7))
                hz.add(x[m][::3], fy[m][::3], BLUE * (0.22 * kf[m][::3])[:, None], 12 * S / 1000)
            # sparks stranded on the sand as the water drains away
            x = w["sx"]
            a = a0 - w["tilt"] * (x / W - 0.5)
            xi = x.astype(int).clip(0, W - 1)
            ys = self.base[xi] + (w["run"][xi] - self.base[xi]) * w["sv"]
            back = TB + 4.5 + 4.5 * self.unease(1 - w["sv"])         # when the edge drains past it
            dt = a - back - 0.2 * w["sd"]
            ks = np.exp(-np.clip(dt, 0, None) / (1.0 + 1.5 * w["sd"])) * (dt > 0) * (0.4 + 0.6 * np.sin(TAU * 124 * t / T_END + 40 * w["sd"]) ** 2)
            ks = ks * (ys > E[xi] + 1)
            m = ks > 0.03
            if m.any():
                glow(img, x[m], ys[m], (BLUE * 0.75 + CORE * 0.25) * ks[m, None], 1.0 + 0.5 * (w["sd"][m] > 1.2))

    @staticmethod
    def unease(v):
        p = np.linspace(0, 1, 101)
        return np.interp(v, ease(p), p)

    def draw_fish(self, img, t, E):
        W, H, S = self.W, self.H, self.S
        for i, (cx, cy, rx, ry, n, m, ph) in enumerate(FISH):
            k = np.arange(36) * 0.045                           # the fish now, and where it was
            tt = t - k
            th = TAU * n * tt / T_END + ph + 0.9 * np.sin(TAU * m * tt / T_END + ph)
            x = W * (cx + rx * np.cos(th))
            y = H * (cy + ry * np.sin(2 * th) * 0.5 + 0.02 * np.sin(3 * th))
            ok = y < np.interp(x, self.xs, E) - 0.02 * H
            sp = np.abs(np.cos(TAU * m * tt / T_END + ph)) ** 3       # brighter while it sprints
            b = (1 - k / k[-1]) ** 1.5 * (0.25 + 0.75 * sp) * ok
            glow(img, x, y, CORE * 0.9 * b[:, None], 1.0 + 0.6 * (k < 0.2))
            self.haze.add(x[::3], y[::3], BLUE * (0.18 * b[::3])[:, None], 12 * S / 1000)

    # ---- the walker and the dog ----------------------------------------------------------------------
    def build_walk(self):
        W, H, rng = self.W, self.H, np.random.default_rng(23)
        a, b = WALK
        self.speed = 1.12 * W / (b - a)
        self.step = 0.024 * self.S
        dist = np.arange(0, 1.12 * W, self.step)
        self.prints = []
        for k, dd in enumerate(dist):
            tk = a + dd / self.speed
            x, y, dx, dy = self.walk_at(tk)
            side = 1 if k % 2 else -1
            px, py = x - dy * side * 0.0065 * self.S, y + dx * side * 0.0065 * self.S
            self.prints.append((tk, px, py, dx, dy))
        P = np.array(self.prints)
        self.pr = dict(t=P[:, 0], x=P[:, 1], y=P[:, 2], dx=P[:, 3], dy=P[:, 4])
        tt = np.arange(a, T_END, 0.05)                       # when each print gets washed away
        wash = np.full(len(P), 1e9)
        for tq in tt:
            E = self.water_edge(tq)
            hit = (tq > self.pr["t"] + 0.3) & (np.interp(self.pr["x"], self.xs, E) > self.pr["y"] - 0.004 * H)
            wash = np.where(hit & (wash > 1e8), tq, wash)
        self.pr["wash"] = wash
        u = rng.random(30)                                  # the shape of a print: heel, sole, toes
        sole = np.stack([np.concatenate([np.linspace(-1, 0.55, 18), np.full(5, 0.85)]),
                         np.concatenate([0.42 * np.sin(np.linspace(0, np.pi, 18)) * rng.uniform(-1, 1, 18),
                                         np.linspace(-0.4, 0.45, 5)])], 1)
        self.sole = sole * np.array([0.0085, 0.0075]) * self.S
        del u
        n = 400                                             # the dog's paws, in quick prints
        self.dogp = dict(t=a + np.arange(n) * 0.11, side=rng.uniform(-1, 1, n))

    def walk_at(self, t):
        a, b = WALK
        x = -0.06 * self.W + self.speed * (t - a)
        xi = min(max(int(x), 0), self.W - 1)
        y = self.y0[xi] + 0.072 * self.H + 0.012 * self.H * math.sin(x / self.W * 5)
        x2 = x + 1.0
        xi2 = min(max(int(x2), 0), self.W - 1)
        y2 = self.y0[xi2] + 0.072 * self.H + 0.012 * self.H * math.sin(x2 / self.W * 5)
        d = unit((x2 - x, y2 - y))
        return x, y, d[0], d[1]

    def dog_at(self, t):
        x, y, dx, dy = self.walk_at(t)
        surf = max(0.0, math.sin(0.62 * t + 0.4)) ** 1.4      # every so often it charges into the surf
        return (x + self.W * (0.075 + 0.035 * math.sin(0.9 * t + 1.0)),
                y - 0.11 * self.H * surf + 0.012 * self.H * math.sin(2.1 * t))

    def draw_prints(self, img, t, E):
        P, S = self.pr, self.S
        age = t - P["t"]
        live = (age > 0) & (t < P["wash"] + 0.25)
        if not live.any():
            return
        idx = np.nonzero(live)[0]
        k = 0.12 + 0.88 * np.exp(-age[idx] / 0.9)
        k = k * np.clip(1 - (age[idx] - 4) / 10, 0, 1)
        k = np.where(t > P["wash"][idx], 1.6 * (1 - (t - P["wash"][idx]) / 0.25), k)   # a last flare as it goes
        ux, uy = P["dx"][idx], P["dy"][idx]
        sx = self.sole[:, 0][None, :]
        sy = self.sole[:, 1][None, :]
        X = P["x"][idx, None] + ux[:, None] * sx - uy[:, None] * sy
        Y = P["y"][idx, None] + uy[:, None] * sx + ux[:, None] * sy
        kk = np.repeat(k, X.shape[1])
        glow(img, X.ravel(), Y.ravel(), CORE * 1.1 * kk[:, None], 1.3 * S / 1000 + 0.3)
        self.haze.add(P["x"][idx], P["y"][idx], BLUE * (0.8 * k)[:, None], 18 * S / 1000)
        a, b = WALK                                      # the dog's paws
        D = self.dogp
        age = t - D["t"]
        m = (age > 0) & (age < 3.0) & (D["t"] < b + 6)
        if m.any():
            pts = [self.dog_at(tk) for tk in D["t"][m]]
            px = np.array([p[0] for p in pts]) + D["side"][m] * 0.006 * S
            py = np.array([p[1] for p in pts]) + D["side"][m] * 0.004 * S
            on = py > np.interp(px, self.xs, E) + 2
            k = np.exp(-age[m] / 0.7) * on
            glow(img, px, py, CORE * 0.9 * k[:, None], 1.3 * S / 1000 + 0.3)
            self.haze.add(px, py, BLUE * (0.3 * k)[:, None], 10 * S / 1000)

    def draw_people(self, img, t, E):
        a, b = WALK
        S = self.S
        if not (a - 0.5 < t < b + 6):
            return
        rng = np.random.default_rng(99)
        x, y, dx, dy = self.walk_at(t)
        px, py = -dy, dx
        ph = (t - a) * self.speed / self.step * math.pi
        S = S * 1.0
        dots = []

        def blob(cx, cy, along, across, n, col, r=1.5):
            u, v = rng.uniform(-1, 1, (2, n))
            keep = u * u + v * v <= 1
            u, v = u[keep], v[keep]
            dots.append((cx + dx * u * along + px * v * across, cy + dy * u * along + py * v * across, col, r))

        if t < b + 1:
            for side in (-1, 1):                           # shoes, arms, then shoulders and head on top
                f = 0.012 * S * math.sin(ph + (side > 0) * math.pi)
                blob(x + dx * f + px * side * 0.0085 * S, y + dy * f + py * side * 0.0085 * S, 0.0080 * S, 0.0038 * S, 60,
                     (40, 40, 46))
            for side in (-1, 1):
                f = -0.013 * S * math.sin(ph + (side > 0) * math.pi)
                blob(x + dx * f + px * side * 0.0190 * S, y + dy * f + py * side * 0.0190 * S, 0.0085 * S, 0.0034 * S, 90,
                     (78, 82, 96))
            blob(x, y, 0.0085 * S, 0.0180 * S, 700, (104, 108, 124))
            blob(x - dx * 0.001 * S, y - dy * 0.001 * S, 0.0062 * S, 0.0062 * S, 220, (52, 40, 34))
        gx, gy = self.dog_at(t)                            # the dog
        hx, hy = self.dog_at(t + 0.05)
        d = unit((hx - gx, hy - gy + 1e-6))
        q = np.array([-d[1], d[0]])
        if -0.05 * self.W < gx < 1.05 * self.W:
            dd = (d[0], d[1])
            dx, dy, px, py = dd[0], dd[1], q[0], q[1]
            gal = math.sin(t * 15)
            for fwd, side in ((0.013, -1), (0.013, 1), (-0.010, -1), (-0.010, 1)):
                f = (fwd + 0.004 * gal * (1 if fwd > 0 else -1)) * S
                blob(gx + dx * f + px * side * 0.0045 * S, gy + dy * f + py * side * 0.0045 * S, 0.003 * S, 0.0022 * S, 20,
                     (60, 54, 48))
            blob(gx, gy, 0.016 * S, 0.0068 * S, 360, (118, 104, 84))
            blob(gx + dx * 0.019 * S, gy + dy * 0.019 * S, 0.0065 * S, 0.0055 * S, 120, (124, 110, 88))
            for side in (-1, 1):
                blob(gx + dx * 0.017 * S + px * side * 0.0055 * S, gy + dy * 0.017 * S + py * side * 0.0055 * S,
                     0.0035 * S, 0.0022 * S, 30, (84, 70, 56))
            wag = 0.45 * math.sin(t * 11)
            tdx, tdy = -dx * math.cos(wag) - px * math.sin(wag), -dy * math.cos(wag) - py * math.sin(wag)
            blob(gx + tdx * 0.022 * S, gy + tdy * 0.022 * S, 0.006 * S, 0.0022 * S, 50, (110, 96, 78))
            wet = gy < np.interp(gx, self.xs, E) + 3
            if wet:                                          # splashing through the glowing surf
                n = 220
                rr = np.random.default_rng(int(t * 30))
                ang, r = rr.uniform(0, TAU, n), 0.010 * S + rr.random(n) ** 0.7 * 0.030 * S
                k = rr.random(n) ** 3 * 0.9
                glow(img, gx + np.cos(ang) * r, gy + np.sin(ang) * r * 0.8, CORE * k[:, None], 1.0)
                self.haze.add(gx + np.cos(ang[:12]) * r[:12], gy + np.sin(ang[:12]) * r[:12], BLUE * 0.12, 22 * S / 1000)
        for x_, y_, col, r in dots:
            stamp(img, x_, y_, np.asarray(col, np.float32), r)

    # ---- the frame -----------------------------------------------------------------------------------
    def render(self, t):
        t = t % T_END
        W, H = self.W, self.H
        E = self.water_edge(t)
        img = np.empty((H, W, 3), np.float32)               # water above the edge, sand below
        ya, yb = max(0, int(E.min()) - 8), min(H, int(E.max()) + 2)
        img[:ya] = self.water[:ya]
        img[yb:] = self.beach[yb:]
        yy = np.arange(ya, yb, dtype=np.float32)[:, None]
        depth = E[None, :] - yy
        a = np.clip(depth / 5.0, 0, 1) * (0.55 + 0.45 * np.clip(depth / (0.05 * H), 0, 1))
        img[ya:yb] = self.beach[ya:yb] + (self.water[ya:yb] - self.beach[ya:yb]) * a[..., None]
        G = self.glint                                      # starlight on the ripples
        ok = G["y"] < np.interp(G["x"], self.xs, E) - 6
        tw = np.sin(TAU * G["f"] * t / T_END + G["ph"]) ** 8 * ok
        m = tw > 0.05
        glow(img, G["x"][m], G["y"][m], np.array([150, 160, 190]) * 0.45 * tw[m, None], 1.0)
        self.draw_waves(img, t, E)
        self.draw_fish(img, t, E)
        self.draw_prints(img, t, E)
        self.draw_people(img, t, E)
        self.haze.apply(img, 1.0)
        return np.clip(img, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    run(GlowBeach, "Glowing beach", "glow_beach", T_END, STAGES, wrap=True)
