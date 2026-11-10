"""
black_hole.py - a star's last minute beside a black hole, in dots.

A black hole sits in front of the Milky Way, invisible except for what it does to the light
behind it: stars slide round it in arcs and doubled images, and as a distant spiral galaxy
drifts exactly behind it, the galaxy is drawn out into a full Einstein ring. Then a star like
the Sun comes falling in. As it swings past it is stretched long, and at its closest it is
torn apart - spaghettified - into a glowing stream. Half the stream is flung away for good;
the other half arcs out, bent into ghostly arches by the hole's gravity, and comes falling
back. Where it crashes into itself there is a flash, and a ring of that light sweeps out
through the dust around the hole. The returning gas winds into a blazing disk - its near
side brighter, rushing toward us - with a thin ring of light hugging the shadow and the far
side of the disk lifted into an arch over the top. Jets punch out of the poles. Last, a
little probe falls in, its beacon blinking slower and slower and redder and redder, until
it freezes at the edge of the dark and fades away.

    python black_hole.py            interactive window (Space pause, R restart, S save frame, Esc quit)
    python black_hole.py --png      save five stages
    python black_hole.py --gif      save the whole minute as a GIF
"""
import math

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))   # the shared engine: engine/pointillism.py

from pointillism import Haze, glow, run, shades, ss, stamp

T_END = 60.0
STAGES = [10.8, 15.0, 26.6, 42.0, 50.0]
TAU = 2 * math.pi

TP = 12.0                     # the star's closest approach
TILT = math.radians(8)        # we see the disk from just above its plane
PHI_P = math.radians(-118)    # where the closest approach happens (in front of the hole, lower left)
STAR = np.array([255, 236, 200], np.float32)
GAS = np.array([255, 178, 120], np.float32)
JET = np.array([170, 196, 255], np.float32)
HOT = [(0.0, (190, 50, 30)), (0.25, (240, 110, 50)), (0.45, (255, 170, 90)), (0.65, (255, 222, 170)),
       (0.85, (250, 244, 236)), (1.0, (214, 228, 255))]


def heat(tau):
    """Colour of glowing gas, from dull red (0) to blue-white (1)."""
    ts = np.array([h[0] for h in HOT])
    cs = np.array([h[1] for h in HOT], np.float64)
    tau = np.clip(tau, 0, 1)
    return np.stack([np.interp(tau, ts, cs[:, c]) for c in range(3)], -1).astype(np.float32)


class BlackHole:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.S = S = min(W, H)
        self.rng = np.random.default_rng(42)
        self.cx, self.cy = 0.5 * W, 0.47 * H
        self.rs = 0.030 * S                       # Schwarzschild radius, in pixels
        self.Rsh = 2.6 * self.rs                  # the shadow
        self.risco = 3.0 * self.rs
        self.REbg = 0.165 * S                     # Einstein radius for the far background
        self.sd, self.cd = math.sin(TILT), math.cos(TILT)
        self.rp = 0.15 * S                        # the star's closest approach
        self.mu = 1.091 * self.rp ** 3            # GM, in pixels^3 / s^2 (sets the pace)
        v = np.arange(4096, dtype=np.float32) * (1.6 / 255.0)               # a soft shoulder for the bright parts
        self.tone = np.clip(255.0 * v * (1 + v / 9.0) / (1 + v), 0, 255).astype(np.uint8)
        self.lights = []
        self.gas = Haze(W, H, max(2, round(4 * S / 1000)))           # one soft-light layer for everything
        self.haze = self.gas
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        rr = np.hypot(xx - self.cx, yy - self.cy)
        front = np.clip((yy - (self.cy + 0.012 * S)) / (0.02 * S), 0, 1)
        self.gas_mask = np.clip((rr - self.Rsh) / 3.0, 0, 1) + (rr < self.Rsh) * front
        self.build_sky()
        self.build_stream()
        self.build_disk()
        self.build_extras()

    # ---- geometry: the disk plane, and light bent round the hole -------------------------------------
    def view(self, r, phi, h=0.0):
        """A point in the disk plane (polar r, phi; height h) as X right, Y up, Z away from us."""
        c, s = np.cos(phi), np.sin(phi)
        return r * c, r * s * self.sd + h * self.cd, r * s * self.cd - h * self.sd

    def images(self, X, Y, Z=None, RE=None):
        """Where a point behind the hole is seen: the main image pushed outward and a faint second image
        on the far side (point-lens optics). Points in front of the hole are seen where they are."""
        X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
        b = np.hypot(X, Y) + 1e-6
        if RE is None:
            behind = Z > 0
            RE = np.sqrt(2.0 * self.rs * np.maximum(Z, 0) * 1.3) + 1e-6
        else:
            behind = np.ones_like(b, bool)
        q = np.sqrt(b * b + 4 * RE * RE)
        t1 = np.where(behind, (b + q) / 2, b)
        t2 = (b - q) / 2
        u = b / RE
        core = (u * u + 2) / (2 * u * np.sqrt(u * u + 4))
        m1 = np.where(behind, np.minimum(core + 0.5, 6.0), 1.0)
        m2 = np.minimum(np.abs(core - 0.5), 4.0)
        ux, uy = X / b, Y / b
        v1 = (t1 > self.Rsh) | ~behind
        v2 = behind & (-t2 > self.Rsh * 1.01)
        return (ux * t1, uy * t1, m1, v1), (ux * t2, uy * t2, m2, v2)

    def screen(self, X, Y):
        return self.cx + X, self.cy - Y

    def light(self, x, y, col, rad):
        x = np.atleast_1d(np.asarray(x, np.float32))
        self.lights.append((x, np.atleast_1d(np.asarray(y, np.float32)),
                            np.broadcast_to(np.asarray(col, np.float32), (len(x), 3)),
                            np.broadcast_to(np.asarray(rad, np.float32), (len(x),))))

    # ---- the background: dust, stars, a galaxy -------------------------------------------------------
    def build_sky(self):
        W, H, S, rng = self.W, self.H, self.S, np.random.default_rng(7)
        # glowing dust and Milky Way haze in the source plane, then seen through the lens
        src = np.zeros((H, W, 3), np.float32)
        n = int(W * H * 0.35)
        x, y = rng.uniform(0, W, n), rng.uniform(0, H, n)
        u, v = x / W, y / H
        band = np.exp(-((v - 0.62 + 0.35 * (u - 0.5)) / 0.16) ** 2)       # a band of the galaxy, slanting
        cloud = (0.5 + 0.5 * np.sin(u * 9 + np.sin(v * 7) * 2) * np.sin(v * 11 + u * 4)) ** 2
        lum = band * (0.25 + 0.75 * cloud) * rng.random(n) ** 1.5
        col = heat(0.30 + 0.5 * rng.random(n)) * 0.10 + np.array([30, 24, 48], np.float32) * 0.6
        col = col * (1.8 * lum)[:, None] + np.array([4, 4, 10], np.float32) * rng.random((n, 1))
        stamp(src, x, y, col, rng.choice([1.0, 1.5, 2.0], n))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        tx, ty = xx - self.cx, self.cy - yy
        th = np.hypot(tx, ty) + 1e-6
        k = 1 - self.REbg ** 2 / th ** 2                                       # the lens equation, run backwards
        sx = np.clip(self.cx + tx * k, 0, W - 1).astype(int)
        sy = np.clip(self.cy - ty * k, 0, H - 1).astype(int)
        base = src[sy, sx] * np.clip((th - self.Rsh) / 2.0, 0, 1)[..., None]
        self.base = base + np.array([2, 2, 6], np.float32)
        self.base[th < self.Rsh] = 0
        # stars, spread wide so the drift never runs out of them
        n = int(5200 * W * H / 1e6)
        self.stars = dict(x=rng.uniform(-0.9, 1.6, n) * S, y=rng.uniform(-0.6, 0.6, n) * S,
                          b=rng.random(n) ** 5 * 320 + 18, col=heat(0.45 + 0.55 * rng.random(n)))
        # a spiral galaxy, far behind, drifting so that it passes exactly behind the hole
        m = 9000
        arm = rng.integers(0, 2, m)
        t = rng.random(m) ** 0.8
        a = arm * math.pi + 2.6 * np.log(1 + 6 * t) + rng.normal(0, 0.22, m)
        r = 0.11 * S * t
        bulge = rng.random(m) < 0.3
        r[bulge] = 0.025 * S * np.abs(rng.normal(0, 1, int(bulge.sum())))
        a[bulge] = rng.uniform(0, TAU, int(bulge.sum()))
        gx, gy = r * np.cos(a), r * np.sin(a) * 0.55
        rot = 0.5
        col = np.where(bulge[:, None], heat(0.7 + 0.1 * rng.random(m)), heat(0.85 + 0.15 * rng.random(m)))
        knots = (~bulge) & (rng.random(m) < 0.06)
        col[knots] = np.array([255, 140, 190], np.float32)
        self.gal = dict(x=gx * math.cos(rot) - gy * math.sin(rot), y=gx * math.sin(rot) + gy * math.cos(rot),
                        col=col * (0.05 + 0.10 * np.exp(-t * 3))[:, None])

    def drift(self, t):
        """How far the background has slid past the hole."""
        return np.array([-0.13 * self.S + 0.0118 * self.S * t, 0.012 * self.S])

    def draw_sky(self, img, t):
        dx, dy = self.drift(t)
        St = self.stars
        X, Y = St["x"] + dx - 0.35 * self.S, St["y"] + dy
        near = (np.abs(X) < 0.75 * self.S) & (np.abs(Y) < 0.75 * self.S)
        X, Y, b, col = X[near], Y[near], St["b"][near], St["col"][near]
        (x1, y1, m1, v1), (x2, y2, m2, v2) = self.images(X, Y, RE=self.REbg)
        xs, ys, cs, rs = [], [], [], []
        for x, y, m, v in ((x1, y1, m1, v1), (x2, y2, m2, v2)):
            sx, sy = self.screen(x[v], y[v])
            xs.append(sx)
            ys.append(sy)
            cs.append(col[v] * (b[v] * np.minimum(m[v], 8) / 255)[:, None])
            rs.append(1.0 + 0.5 * (b[v] * m[v] > 200))
        G = self.gal
        X, Y = G["x"] + dx, G["y"] + dy
        (x1, y1, m1, v1), (x2, y2, m2, v2) = self.images(X, Y, RE=self.REbg)
        for x, y, m, v in ((x1, y1, m1, v1), (x2, y2, m2, v2)):
            sx, sy = self.screen(x[v], y[v])
            xs.append(sx)
            ys.append(sy)
            cs.append(G["col"][v] * np.minimum(m[v], 5)[:, None] ** 0.7)
            rs.append(np.full(int(v.sum()), 1.0))
        self.light(np.concatenate(xs), np.concatenate(ys), np.concatenate(cs), np.concatenate(rs))

    # ---- the star, and the stream it becomes ------------------------------------------------------
    def build_stream(self):
        rng = self.rng
        n = 14000
        bound = np.arange(n) < n // 2
        q = np.where(bound, -(1 - 0.55 * rng.random(n) ** 1.2), 0.04 + 0.96 * rng.random(n))
        e = 1 + 0.6 * q
        tp = TP + rng.normal(0, 0.10, n)                                     # the star had some size
        a = self.rp / np.abs(1 - e)
        nmean = np.sqrt(self.mu / a ** 3)
        period = np.where(bound, TAU / nmean, 1e9)
        d = rng.normal(0, 1, (3, n))
        d = d / np.linalg.norm(d, axis=0) * rng.random(n) ** (1 / 3)
        self.st = dict(bound=bound, q=q, e=e, tp=tp, a=a, n=nmean, period=period,
                       ox=d[0], oy=d[1], oz=d[2], b=0.6 + 0.8 * rng.random(n),
                       join=tp + period + 0.3, rc=self.rp * (0.62 + 1.55 * rng.random(n) ** 1.3),
                       tin=rng.uniform(45, 140, n), h=rng.normal(0, 1, n))
        self.c0 = math.sqrt(2 * self.rp ** 3 / self.mu)

    def parabola(self, t):
        """The star's own path in: distance and angle from the closest approach (Barker's equation)."""
        B = 1.5 * (t - TP) / self.c0
        y = np.cbrt(B + np.sqrt(B * B + 1))
        D = y - 1 / y
        f = 2 * np.arctan(D)
        return 2 * self.rp / (1 + np.cos(f)), PHI_P + f

    def kepler(self, t):
        """Every scrap of the torn star on its own orbit: ellipses come back, hyperbolas leave."""
        P = self.st
        dt = t - P["tp"]
        e = P["e"]
        r = np.empty_like(e)
        f = np.empty_like(e)
        b = P["bound"]
        M = (P["n"][b] * dt[b] + math.pi) % TAU - math.pi                # ellipses
        eb = e[b]
        E = M + eb * np.sin(M)
        for _ in range(8):
            E = E - (E - eb * np.sin(E) - M) / (1 - eb * np.cos(E))
        r[b] = P["a"][b] * (1 - eb * np.cos(E))
        f[b] = 2 * np.arctan2(np.sqrt(1 + eb) * np.sin(E / 2), np.sqrt(1 - eb) * np.cos(E / 2))
        u = ~b                                                                # hyperbolas
        M = P["n"][u] * dt[u]
        eu = e[u]
        Hh = np.arcsinh(M / eu)
        for _ in range(10):
            Hh = Hh - (eu * np.sinh(Hh) - Hh - M) / (eu * np.cosh(Hh) - 1)
        r[u] = P["a"][u] * (eu * np.cosh(Hh) - 1)
        f[u] = 2 * np.arctan(np.sqrt((eu + 1) / (eu - 1)) * np.tanh(Hh / 2))
        return r, PHI_P + f

    def star_body(self, r, phi):
        """The star as a ball of points: round far out, drawn long toward the hole as it closes in."""
        P = self.st
        R = 0.017 * self.S
        k = 1 + 3.0 * (self.rp / r) ** 3
        X, Y, Z = self.view(r, phi)
        rad = np.array([X, Y, Z]) / math.sqrt(X * X + Y * Y + Z * Z)
        nrm = np.array([0.0, self.cd, -self.sd])
        a2 = np.cross(rad, nrm)
        a2 /= np.linalg.norm(a2)
        a3 = np.cross(rad, a2)
        off = (np.outer(rad, P["ox"] * k) + np.outer(a2, P["oy"] / math.sqrt(k)) + np.outer(a3, P["oz"] / math.sqrt(k))) * R
        return np.array([X, Y, Z]), off

    def draw_stream(self, pts, t):
        P = self.st
        if t < TP - 0.15:
            r, phi = self.parabola(t)
            c, off = self.star_body(r, phi)
            k = slice(None, None, 3)
            X, Y, Z = c[0] + off[0, k], c[1] + off[1, k], c[2] + off[2, k]
            heatk = float(ss(t, TP - 4, TP))
            depth = 1 - (P["ox"][k] ** 2 + P["oy"][k] ** 2 + P["oz"][k] ** 2) ** 0.5
            tau = 0.55 + 0.12 * depth + 0.12 * heatk                          # limb darker, hotter as it is torn
            col = heat(tau) * (0.10 + 0.10 * depth + 0.12 * heatk)[:, None]
            pts.append((X, Y, Z, col, 1.3))
            self.star_now = (c[0], c[1], c[2], 1.0 + heatk)
            return
        self.star_now = None
        r, phi = self.kepler(t)
        late = t - P["tp"]
        grow = np.exp(-np.maximum(late, 0) / 0.9)                            # the star's body melting into the stream
        r0, p0 = self.parabola(TP - 0.15)
        _, off = self.star_body(r0, p0)
        j = ss(t, P["join"], P["join"] + 2.2)                                # returning gas settles into the disk
        if np.any(j > 0):
            age = np.maximum(t - P["join"], 0)
            rc = P["rc"] * (1 - age / P["tin"])
            om = np.sqrt(self.mu / np.maximum(rc, self.risco) ** 3)
            pc = PHI_P + om * age
            x0, y0 = r * np.cos(phi), r * np.sin(phi)
            x1, y1 = rc * np.cos(pc), rc * np.sin(pc)
            x, y = x0 + (x1 - x0) * j, y0 + (y1 - y0) * j
            r, phi = np.hypot(x, y), np.arctan2(y, x)
        alive = (r > self.risco) & (r < 1.4 * self.S)
        spread = np.maximum(late, 0)
        b = P["b"] * np.where(P["bound"], 1.0, 0.8) / (1 + 0.10 * spread) * alive
        flash = 1 + 5 * np.exp(-np.maximum(t - P["join"] - 0.5, 0) / 0.7) * (t > P["join"])   # crashing into itself
        b = b * np.where(j > 0, flash, 1.0)
        tau = 0.70 - 0.25 * np.clip(spread / 12, 0, 1) + 0.2 * j
        b = b * (1 - 0.6 * j)
        h = P["h"] * 0.006 * self.S * (1 + 2 * np.clip(spread / 10, 0, 1)) * (1 - j)
        X, Y, Z = self.view(r, phi, h)
        X, Y, Z = X + off[0] * grow, Y + off[1] * grow, Z + off[2] * grow
        on = b > 0.02
        dense = 0.45 + 0.55 * np.clip(spread / 5, 0, 1)                      # (dimmer while still packed together)
        col = heat(tau[on]) * (b[on] * 0.32 * dense[on])[:, None]
        pts.append((X[on], Y[on], Z[on], col, 1.2, "gas"))

    # ---- the accretion disk -------------------------------------------------------------------------
    def build_disk(self):
        rng = self.rng
        n = 8000
        r = self.risco * (1 + 3.6 * rng.random(n) ** 1.5)
        clump = (0.5 + 0.5 * np.sin(rng.uniform(0, TAU, n) * 3)) ** 2
        self.disk = dict(r=r, p=rng.uniform(0, TAU, n), b=0.35 + 0.65 * rng.random(n) * (0.5 + clump),
                         h=rng.normal(0, 1, n))
        self.disk_ring = np.linspace(0, TAU, 1400, endpoint=False)

    def disk_level(self, t):
        return float(ss(t, 32.0, 44.0)) * (1 - 0.25 * float(ss(t, 54.0, 60.0)))

    def gas_light(self, r, phi, Z, base):
        """Brightness and colour of orbiting gas: hotter inside, and much brighter on the side rushing toward us."""
        beta = 0.55 * np.sqrt(self.risco / np.maximum(r, self.risco))
        toward = -beta * np.cos(phi) * self.cd                                # line-of-sight speed (approaching > 0)
        g = 1 / (np.sqrt(1 - beta ** 2) * (1 - toward))
        g = g * np.sqrt(np.clip(1 - self.rs / r, 0.05, 1))                    # and the pull of gravity
        tau = np.clip((self.risco / r) ** 0.75 * 0.95 * g ** 0.9, 0, 1)
        emis = (self.risco / r) ** 1.9 * np.sqrt(np.clip(1 - np.sqrt(self.risco / r), 0, 1)) * 1.3
        return heat(tau) * (base * emis * np.minimum(g, 1.8) ** 3)[:, None]

    def draw_disk(self, pts, t):
        L = self.disk_level(t)
        if L <= 0.01:
            return
        D = self.disk
        om = np.sqrt(self.mu / D["r"] ** 3)
        trail = np.array([0.0, 0.035, 0.07, 0.105])[:, None] * np.sqrt(self.risco / D["r"])[None]
        phi = (D["p"] + om * t)[None] - trail                                   # a short smear along the orbit
        r = np.broadcast_to(D["r"], phi.shape)
        X, Y, Z = self.view(r, phi, np.broadcast_to(D["h"] * 0.003 * self.S * (D["r"] / self.risco) ** 0.5, phi.shape))
        w = np.array([1.0, 0.75, 0.5, 0.28])[:, None] * D["b"][None] * L
        col = self.gas_light(r.ravel(), phi.ravel(), Z.ravel(), w.ravel()).reshape(4, -1, 3)
        pts.append((X[0], Y[0], Z[0], col[0], 1.2, "gas"))                      # each clump as a dot,
        pts.append((X[1:].ravel(), Y[1:].ravel(), Z[1:].ravel(), col[1:].reshape(-1, 3), 1.2, "soft"))   # its smear as glow

    # ---- jets, the light echo, the probe -------------------------------------------------------------
    def build_extras(self):
        rng = self.rng
        n = 2600
        self.jet = dict(u=rng.random(n), w=rng.normal(0, 1, n), ph=rng.uniform(0, TAU, n))
        n = 9000                                                              # dust round the hole, lit by the echo
        a, rr = rng.uniform(0, TAU, n), 0.12 * self.S + rng.random(n) ** 0.7 * 0.75 * self.S
        clump = (0.5 + 0.5 * np.sin(a * 5 + rr / self.S * 9)) ** 3
        keep = rng.random(n) < 0.25 + 0.75 * clump
        self.dust = dict(x=(rr * np.cos(a))[keep], y=(rr * np.sin(a) * 0.8)[keep], r=rr[keep])
        self.echo_j = rng.normal(0, 1, 4 * 1400)

    def draw_jets(self, img, t):
        L = 0.46 * self.S * float(ss(t, 35.0, 44.0))
        if L < 2:
            return
        J = self.jet
        s = (J["u"] + 0.35 * (t - 35.0)) % 1.0
        d = s * L
        knot = 0.35 + 0.65 * np.sin(14 * s - 5 * t + J["ph"] * 0.2) ** 4
        w = (0.004 + 0.045 * s) * self.S * J["w"] * 0.5 + 0.012 * self.S * s * np.sin(1.3 * t + 5 * s)
        fade = (1 - s) ** 2.0 * np.clip(d / (0.03 * self.S), 0, 1)
        xs, ys, cs = [], [], []
        for sgn, power in ((1, 1.0), (-1, 0.12)):                             # the one tilted at us is brighter
            X = w
            Y = sgn * (self.Rsh * 0.4 + d) * self.cd
            Z = -sgn * d * self.sd
            ok = (np.hypot(X, Y) > self.Rsh) | (Z < 0)
            sx, sy = self.screen(X[ok], Y[ok])
            xs.append(sx)
            ys.append(sy)
            cs.append(JET[None] * (power * 0.65 * knot * fade)[ok][:, None])
        x, y, c = np.concatenate(xs), np.concatenate(ys), np.concatenate(cs)
        self.light(x, y, c, 1.2)
        self.haze.add(x[::6], y[::6], c[::6] * 0.6, 14 * self.S / 1000)

    def draw_echo(self, img, t):
        t0 = TP + TAU / math.sqrt(self.mu / (self.rp / 0.6) ** 3) + 0.6        # the first gas to come back
        a = t - t0
        if not 0 < a < 9:
            return
        R = 0.10 * self.S * a + 0.05 * self.S
        D = self.dust
        k = np.exp(-((D["r"] - R) / (0.03 * self.S)) ** 2) * (1 - a / 9) ** 1.5
        on = k > 0.03
        sx, sy = self.screen(D["x"][on], D["y"][on])
        self.light(sx, sy, np.array([200, 210, 255])[None] * (3.0 * k[on])[:, None], 1.2)
        ang = np.tile(self.disk_ring, 4)                                   # the front of the light itself: a soft band
        band = R * (1 + 0.012 * np.sin(ang * 5 + a) + self.echo_j * (0.05 + 0.02 * a))
        sx, sy = self.screen(band * np.cos(ang), band * np.sin(ang))
        self.gas.splat(sx, sy, np.array([150, 170, 255]) * 0.22 * (1 - a / 9) ** 2)
        if a < 1.5:                                                         # and the flash itself
            f = (1 - a / 1.5) ** 2
            self.haze.add(np.full(6, self.cx), np.full(6, self.cy), np.array([255, 230, 200]) * 0.9 * f, 40 * self.S / 1000)

    def draw_probe(self, img, t):
        if t < 39.0:
            return
        S = self.S
        tau = 2.6
        g = math.exp(-max(t - 44.0, 0) / tau)                                # the light from it, ever more redshifted
        r = self.Rsh * 1.03 + 0.55 * S * (1 - float(ss(t, 39.0, 44.0)) * 0.12) * g
        ang = 0.62 + 0.25 * (1 - g)
        X, Y = r * math.cos(ang), r * math.sin(ang)
        sx, sy = self.screen(X, Y)
        k = np.linspace(-1, 1, 17)
        bx = np.concatenate([k * 0.015 * S, np.zeros(17), [0.0, 0.003 * S, -0.003 * S]])   # solar panels and a mast
        by = np.concatenate([np.zeros(17), k * 0.007 * S, [0.0, 0.003 * S, 0.003 * S]])
        fade = g ** 1.3 * float(ss(t, 39.0, 40.5))
        body = np.array([190, 190, 205]) * (0.5 + 0.5 * g) * fade
        self.light(sx + bx * (0.4 + 0.6 * g), sy + by, body, 1.3)
        if t < 44:                                                           # its beacon: steady at first,
            phase = (t - 39.0) / 0.45                                        # then slower and slower
        else:
            phase = 5 / 0.45 + tau / 0.45 * (1 - g)
        on = (phase % 1.0) < 0.22
        if on and fade > 0.01:
            red = 1 - g
            col = np.array([120, 220, 255]) * (1 - red) + np.array([255, 50, 20]) * red
            self.light(np.array([sx]), np.array([sy - 0.006 * S]), col * 2.2 * fade, 2.5)
            self.haze.add(np.array([sx]), np.array([sy]), col * 0.5 * fade, 20 * S / 1000)

    # ---- the frame ---------------------------------------------------------------------------------
    def render(self, t):
        img = self.base.copy()
        self.draw_sky(img, t)
        self.draw_echo(img, t)
        pts = []
        self.draw_stream(pts, t)
        self.draw_disk(pts, t)
        xs, ys, cs, rs = [], [], [], []
        ringk = np.zeros(2)
        for item in pts:
            X, Y, Z, col, rad = item[:5]
            mode = item[5] if len(item) > 5 else "dot"
            (x1, y1, m1, v1), (x2, y2, m2, v2) = self.images(X, Y, Z)
            for x, y, m, v, w in ((x1, y1, m1, v1, 1.0), (x2, y2, m2, v2, 0.8)):
                sx, sy = self.screen(x[v], y[v])
                c = col[v] * (np.minimum(m[v], 4) * w)[:, None]
                if mode != "soft":
                    rr = np.full(int(v.sum()), rad)
                    self.light(sx, sy, c * (0.55 if mode == "gas" else 1.0), rr)
                if mode == "dot":
                    self.haze.splat(sx, sy, c * 0.9)
                else:                                                       # hot gas: a smooth glow underneath
                    self.gas.splat(sx, sy, c * (1.9 if mode == "gas" else 2.4))
        if self.star_now is not None:                                        # the star's own glow
            X, Y, Z, k = self.star_now
            (x1, y1, m1, v1), _ = self.images(np.array([X]), np.array([Y]), np.array([Z]))
            if v1[0]:
                sx, sy = self.screen(x1, y1)
                self.haze.add(sx, sy, np.array([255, 210, 150]) * 0.7 * k * min(m1[0], 3), 40 * self.S / 1000)
        L = self.disk_level(t)                                               # the photon ring hugging the shadow
        if L > 0.01:
            a = self.disk_ring
            side = 0.35 + 0.65 * np.clip(-np.cos(a), 0, 1) ** 1.5
            for R, k in ((1.03, 0.10),):
                sx, sy = self.screen(self.Rsh * R * np.cos(a), self.Rsh * R * np.sin(a))
                self.light(sx, sy, np.array([255, 226, 190])[None] * (k * L * side)[:, None], 1.1)
            sx, sy = self.screen(self.Rsh * 1.08 * np.cos(a), self.Rsh * 1.08 * np.sin(a))
            self.gas.splat(sx, sy, np.array([255, 210, 170])[None] * (1.2 * L * side)[:, None])
        self.draw_jets(img, t)
        self.draw_probe(img, t)
        x, y, c, r = (np.concatenate(a) for a in zip(*self.lights))               # all the light, in two passes
        self.lights = []
        keep = c.max(1) > 1.2                                                 # (skipping what would not show)
        x, y, c, r = x[keep], y[keep], c[keep], r[keep]
        big = r > 1.25
        glow(img, x[~big], y[~big], c[~big], r[~big])
        glow(img, x[big], y[big], c[big], r[big])
        self.gas.apply(img, 1.0, 2, self.gas_mask, int(self.cy - self.Rsh - 4), int(self.cy + self.Rsh + 4))
        return self.tone[np.clip(img, 0, 4095).astype(np.int16)]


if __name__ == "__main__":
    run(BlackHole, "Black hole - a star's last minute", "black_hole", T_END, STAGES)
