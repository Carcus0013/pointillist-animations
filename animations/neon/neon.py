"""
neon.py - shared pieces for the Neon set (cyberpunk, synthwave, dieselpunk), built on the
Garden's pointillism engine: a camera, a filmic tone curve, made-up sign glyphs, neon tubes
with bloom, and rain.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))   # the shared engine

from pointillism import (Backdrop, Haze, Sun, bezier, glow, keyed, mix, paint, run, shades, sky, ss,  # noqa: E402,F401
                         stamp, unit, warm_glow)

TAU = 2 * math.pi

NEON = dict(
    pink=(255, 46, 160), magenta=(230, 60, 255), cyan=(30, 225, 255), blue=(50, 110, 255),
    violet=(150, 70, 255), amber=(255, 160, 50), red=(255, 40, 60), green=(50, 255, 150), white=(235, 240, 255),
)


def tone_lut(exposure=1.0, shoulder=6.0):
    """A soft shoulder for very bright light: neon cores go white, halos keep their colour."""
    v = np.arange(4096, dtype=np.float32) * (exposure / 255.0)
    return np.clip(255.0 * v * (1 + v / shoulder) / (1 + v), 0, 255).astype(np.uint8)


def finish(img, lut):
    np.minimum(img, 4095, out=img)                     # (light only ever adds, so nothing is below zero)
    return lut[img.astype(np.uint16)]


class Cam:
    """A pinhole camera looking along +z. X right, Y up, z away; eye at height hc."""

    def __init__(self, W, H, vx, vy, f, hc):
        self.W, self.H, self.vx, self.vy, self.f, self.hc = W, H, vx, vy, f, hc

    def proj(self, X, Y, z):
        z = np.maximum(np.asarray(z, np.float64), 0.05)
        return self.vx + self.f * np.asarray(X) / z, self.vy + self.f * (self.hc - np.asarray(Y)) / z


# ---- made-up glyphs for signs (no real words) ---------------------------------------------------
def glyph(rng):
    """Strokes of an invented character in a unit cell, as a list of (x0, y0, x1, y1)."""
    s = []
    kind = rng.integers(0, 4)
    if kind == 0:                                              # a box with a bar
        s += [(0.15, 0.15, 0.85, 0.15), (0.85, 0.15, 0.85, 0.85), (0.85, 0.85, 0.15, 0.85), (0.15, 0.85, 0.15, 0.15)]
        y = rng.uniform(0.35, 0.65)
        s.append((0.15, y, 0.85, y))
    elif kind == 1:                                            # a stem with branches
        x = rng.uniform(0.35, 0.65)
        s.append((x, 0.08, x, 0.92))
        for _ in range(rng.integers(2, 4)):
            y = rng.uniform(0.15, 0.85)
            s.append((rng.uniform(0.05, 0.3), y, rng.uniform(0.7, 0.95), y + rng.uniform(-0.15, 0.15)))
    elif kind == 2:                                            # a roof and legs
        s += [(0.1, 0.25, 0.9, 0.25), (0.5, 0.1, 0.5, 0.25)]
        s += [(0.5, 0.25, 0.15, 0.9), (0.5, 0.25, 0.85, 0.9)]
        if rng.random() < 0.5:
            s.append((0.3, 0.6, 0.7, 0.6))
    else:                                                      # hooks and ticks
        s += [(0.15, 0.2, 0.8, 0.2), (0.8, 0.2, 0.55, 0.9), (0.2, 0.5, 0.45, 0.65)]
        if rng.random() < 0.6:
            s.append((0.6, 0.45, 0.9, 0.55))
    return s


def stroke_points(strokes, step):
    """Points along strokes, `step` apart (in the strokes' own units)."""
    xs, ys = [], []
    for x0, y0, x1, y1 in strokes:
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step) + 1)
        q = np.linspace(0, 1, n)
        xs.append(x0 + (x1 - x0) * q)
        ys.append(y0 + (y1 - y0) * q)
    return np.concatenate(xs), np.concatenate(ys)


def sign_strokes(rng, n, vertical=True, frame=True):
    """A sign of n glyphs in a column (or row), with an optional frame; units are glyph cells."""
    strokes = []
    for i in range(n):
        for x0, y0, x1, y1 in glyph(rng):
            if vertical:
                strokes.append((x0, y0 + i * 1.15, x1, y1 + i * 1.15))
            else:
                strokes.append((x0 + i * 1.1, y0, x1 + i * 1.1, y1))
    if frame:
        w, h = (1.0, n * 1.15 - 0.15) if vertical else (n * 1.1 - 0.1, 1.0)
        m = 0.22
        strokes += [(-m, -m, w + m, -m), (w + m, -m, w + m, h + m), (w + m, h + m, -m, h + m), (-m, h + m, -m, -m)]
    return strokes


# ---- light -------------------------------------------------------------------------------------
class Lights:
    """Collects light for one frame and lays it down in three layers: crisp cores (dot by dot), a tight soft halo
    (half size), and a wide bloom (quarter size). The soft layers are cheap however many points they take."""

    def __init__(self, W, H, k=4):
        self.items = []
        S = min(W, H)
        kh = max(1, round(3 * S / 1000))
        self.halo = Haze(W, H, kh)
        self.bloom = Haze(W, H, 2 * kh)                      # exactly twice as coarse, so it folds into the halo grid
        self.hk = self.halo.k ** 2 / 4.0                     # keep the halo's strength the same at any size

    def add(self, x, y, col, rad=1.0):
        x = np.atleast_1d(np.asarray(x, np.float32))
        if len(x):
            self.items.append((x, np.broadcast_to(np.asarray(y, np.float32), x.shape),
                               np.broadcast_to(np.asarray(col, np.float32), (len(x), 3)),
                               np.broadcast_to(np.asarray(rad, np.float32), x.shape)))

    def soft(self, x, y, col):
        """Light with no crisp centre: halos, reflections, haze."""
        x = np.atleast_1d(np.asarray(x, np.float32))
        if len(x):
            self.halo.splat(x, np.broadcast_to(np.asarray(y, np.float32), x.shape), np.asarray(col, np.float32) / self.hk)

    def tube(self, x, y, col, power, core=1.0, halo=1.0, bloom=1.0):
        """A neon tube: a hot near-white core, a coloured halo, and a wide bloom. col and power per point or shared."""
        x = np.atleast_1d(np.asarray(x, np.float32))
        n = len(x)
        if not n:
            return
        y = np.broadcast_to(np.asarray(y, np.float32), (n,))
        col = np.broadcast_to(np.asarray(col, np.float32), (n, 3))
        p = np.broadcast_to(np.asarray(power, np.float32), (n,))[:, None]
        hot = col * 0.45 + 150.0
        self.add(x, y, hot * p * (0.9 * core), 1.0)
        self.soft(x, y, col * p * (0.9 * halo))
        if bloom > 0:
            self.bloom.splat(x[::2], y[::2], col[::2] * p[::2] * (0.9 * bloom))

    def draw(self, img, blur=2, mask=None):
        if self.items:
            x, y, c, r = (np.concatenate(a) for a in zip(*self.items))
            keep = c.max(1) > 1.0
            x, y, c, r = x[keep], y[keep], c[keep], r[keep]
            big = r > 1.25
            glow(img, x[~big], y[~big], c[~big], r[~big])
            glow(img, x[big], y[big], c[big], r[big])
        self.items = []
        # fold the blurred bloom into the halo grid, then lay both down in a single pass
        self.bloom.flush()
        bb = self.bloom.buf
        for _ in range(blur):
            c = bb * 0.5
            c[1:] += 0.25 * bb[:-1]
            c[:-1] += 0.25 * bb[1:]
            bb = c * 0.5
            bb[:, 1:] += 0.25 * c[:, :-1]
            bb[:, :-1] += 0.25 * c[:, 1:]
        up = np.repeat(np.repeat(bb, 2, 0), 2, 1) * 4.0
        hb = self.halo.buf
        h, w = min(hb.shape[0], up.shape[0]), min(hb.shape[1], up.shape[1])
        self.halo.flush()
        hb[:h, :w] += up[:h, :w]
        self.bloom.buf[:] = 0
        self.halo.apply(img, 1.0, 1, mask)


def flicker(t, seed, rate=1.0, broken=0.0):
    """A neon tube's brightness: a gentle hum, and if `broken`, sudden stutters."""
    h = 0.93 + 0.07 * math.sin(60 * t + seed * 7.1)
    if broken > 0:
        k = math.sin(t * 2.3 * rate + seed * 3.7) + math.sin(t * 5.1 * rate + seed)
        if k > 2 - broken * 1.6:
            h *= 0.15 + 0.85 * (math.sin(t * 90 + seed) > 0.2)
    return h
