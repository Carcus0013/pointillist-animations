"""
pointillism.py - the little engine shared by the Garden pieces.

Everything is drawn as dots. Each frame a scene collects opaque dots as groups of
(x, y, colour, radius, depth) and `paint` lays them down far to near; light (sun glow,
glints, sparkles) is added on top with `glow`. Backdrops that change with the time of
day are built once as a few key images and cross-faded per frame (`Backdrop`).

`run(make_scene, ...)` gives every piece the same window and exports:
    Space pause, R restart, S save frame, Esc quit;  --png for five stages, --gif for the whole life.
"""
import math
import sys
import time

import numpy as np
from PIL import Image


# ---- easing and interpolation ----------------------------------------------------------
def ss(x, a, b):
    u = np.clip((np.asarray(x, np.float64) - a) / (b - a), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def keyed(t, keys):
    """Smoothly interpolate a value between (time, value) keys."""
    ts = [k[0] for k in keys]
    if t <= ts[0]:
        return np.asarray(keys[0][1], np.float32)
    if t >= ts[-1]:
        return np.asarray(keys[-1][1], np.float32)
    i = int(np.searchsorted(ts, t, side="right")) - 1
    w = np.float32(ss(t, ts[i], ts[i + 1]))
    return np.asarray(keys[i][1], np.float32) * (1 - w) + np.asarray(keys[i + 1][1], np.float32) * w


def unit(v):
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


# ---- dots -----------------------------------------------------------------------------
def _disk_offsets(r=3):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    d2 = (xx ** 2 + yy ** 2).ravel()
    keep = np.nonzero(d2 <= r * r)[0]
    order = keep[np.argsort(d2[keep], kind="stable")]
    return np.stack([xx.ravel()[order], yy.ravel()[order]], 1).astype(np.int32), d2[order]


OFFS, OFF_D2 = _disk_offsets(3)


def _footprint(shape, x, y, rad):
    H, W = shape[:2]
    k = int(np.searchsorted(OFF_D2, float(rad.max()) ** 2 + 0.01, side="right"))
    off, d2 = OFFS[:k], OFF_D2[:k]
    xs = np.rint(x).astype(np.int32)[:, None] + off[None, :, 0]
    ys = np.rint(y).astype(np.int32)[:, None] + off[None, :, 1]
    inside = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    return xs, ys, d2, inside


RGB = np.dtype((np.void, 12))                      # one float32 colour as a single 12-byte record


def _rows(m):
    """Row number of every True in a 2D mask, in order (quicker than np.nonzero(m)[0])."""
    return np.repeat(np.arange(m.shape[0]), m.sum(1))


def stamp(img, x, y, col, rad):
    """Paint round dots (later dots cover earlier ones)."""
    x = np.atleast_1d(np.asarray(x, np.float32))
    if not len(x):
        return
    H, W = img.shape[:2]
    y = np.atleast_1d(np.asarray(y, np.float32))
    rad = np.broadcast_to(np.asarray(rad, np.float32), x.shape)
    col = np.broadcast_to(np.asarray(col, np.float32), (len(x), img.shape[2]))
    near = (x > -4) & (x < W + 4) & (y > -4) & (y < H + 4)          # skip dots wholly off the frame
    if not near.all():
        x, y, rad, col = x[near], y[near], rad[near], col[near]
        if not len(x):
            return
    fast = img.flags.c_contiguous and img.dtype == np.float32 and img.shape[2] == 3
    if fast:                                                         # move colours as whole records: much quicker
        flat = img.reshape(-1, 3).view(RGB).ravel()
        col = np.ascontiguousarray(col, np.float32).view(RGB).ravel()
    step = 8192                                                      # in runs, each sized to its own biggest dot
    for i in range(0, len(x), step):
        xc, yc, rc, cc = x[i:i + step], y[i:i + step], rad[i:i + step], col[i:i + step]
        xs, ys, d2, inside = _footprint(img.shape, xc, yc, rc)
        m = inside & (d2[None, :] <= (rc * rc)[:, None] + 0.01)
        if fast:
            flat[(ys * W + xs)[m]] = cc[_rows(m)]
        else:
            img[ys[m], xs[m]] = cc[_rows(m)]


def glow(img, x, y, col, rad):
    """Add soft light (dots brighten whatever is under them)."""
    x = np.atleast_1d(np.asarray(x, np.float32))
    if not len(x):
        return
    y = np.atleast_1d(np.asarray(y, np.float32))
    rad = np.broadcast_to(np.asarray(rad, np.float32), x.shape)
    nc = img.shape[2]
    col = np.ascontiguousarray(np.broadcast_to(np.asarray(col, np.float32), (len(x), nc)))
    xs, ys, d2, inside = _footprint(img.shape, x, y, rad)
    fall = np.clip(1 - np.sqrt(d2)[None, :] / (rad[:, None] + 0.6), 0, 1)
    m = inside & (fall > 0)
    if not m.any():
        return
    xs, ys = xs[m], ys[m]
    rows = _rows(m)
    if nc == 3:
        vals = col.view(RGB).ravel()[rows].view(np.float32).reshape(-1, 3) * fall[m][:, None]
    else:
        vals = col[rows] * fall[m][:, None]
    x0, y0 = int(xs.min()), int(ys.min())
    bw, bh = int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1
    if bw * bh > 7 * len(xs):                       # sparse: add dot by dot
        np.add.at(img, (ys, xs), vals)
        return
    idx = (ys - y0) * bw + (xs - x0)               # dense: sum per pixel, then add the patch
    patch = img[y0:y0 + bh, x0:x0 + bw]
    for c in range(nc):
        patch[..., c] += np.bincount(idx, vals[:, c], bw * bh).reshape(bh, bw)


def paint(img, dots):
    """Paint groups of (x, y, colour, radius, depth), far to near."""
    xs, ys, cs, rs, zs = [], [], [], [], []
    for x, y, c, r, z in dots:
        x = np.atleast_1d(np.asarray(x, np.float32)).ravel()
        n = len(x)
        if not n:
            continue
        xs.append(x)
        ys.append(np.broadcast_to(np.asarray(y, np.float32), (n,)).ravel())
        cs.append(np.broadcast_to(np.asarray(c, np.float32), (n, 3)))
        rs.append(np.broadcast_to(np.asarray(r, np.float32), (n,)))
        zs.append(np.broadcast_to(np.asarray(z, np.float32), (n,)))
    if not xs:
        return
    x, y, c, r, z = (np.concatenate(a) for a in (xs, ys, cs, rs, zs))
    o = np.argsort(z, kind="stable")
    stamp(img, x[o], y[o], np.clip(c[o], 0, 255), r[o])


def lights(img, groups):
    """Add groups of (x, y, colour, radius) as light."""
    for x, y, c, r in groups:
        glow(img, x, y, c, r)


def lit(col, x, y, ox, oy, size, resp, light, wash=0.0):
    """Low sunlight on a part centred at (ox, oy): shadows take the ambient tint, the side
    facing the light catches a warm rim. `wash` lets light through thin things."""
    amb, key, (lx, ly) = light
    f = ((np.asarray(x) - ox) * lx + (np.asarray(y) - oy) * ly) / size
    k = (np.clip(0.5 + 0.6 * f, 0, 1) ** 2 + wash) * resp
    return np.asarray(col, np.float32) * amb + key * np.broadcast_to(k, np.shape(x))[:, None]


# ---- colour and shape sampling ------------------------------------------------------------
def shades(rng, pal, g, spread=0.5, jitter=0.06):
    """Pick a colour per dot from a palette, guided by g in [0, 1] with some randomness."""
    pal = np.asarray(pal, np.float32)
    g = np.atleast_1d(np.asarray(g, np.float64))
    i = np.clip(np.rint(np.clip(g, 0, 1) * (len(pal) - 1) + rng.normal(0, spread, g.shape)), 0, len(pal) - 1)
    return (pal[i.astype(int)] * (1 + rng.normal(0, jitter, (len(g), 1)))).astype(np.float32)


def gradient(stops, g):
    """Colour at heights g (fractions of the frame) from (height, colour) stops."""
    ys = np.array([s[0] for s in stops], np.float64)
    cs = np.array([s[1] for s in stops], np.float64)
    g = np.asarray(g, np.float64)
    return np.stack([np.interp(g, ys, cs[:, c]) for c in range(3)], -1).astype(np.float32)


def sample_u(rng, n, width):
    """Positions along a shape, denser where it is wider, so the fill looks even."""
    out, got = [], 0
    while got < n:
        u = rng.random(2 * n)
        keep = u[rng.random(2 * n) < width(u)]
        out.append(keep)
        got += keep.size
    return np.concatenate(out)[:n]


def sample_v(rng, n, rim=0.12):
    v = rng.uniform(-1, 1, n)
    r = rng.random(n) < rim
    v[r] = np.sign(v[r]) * rng.uniform(0.85, 1.0, int(r.sum()))
    return v


def shuffle(d, rng):
    p = rng.permutation(len(next(iter(d.values()))))
    return {k: v[p] for k, v in d.items()}


def join(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def mix(a, b, w):
    return a + (b - a) * np.broadcast_to(np.asarray(w, np.float32), (len(a),))[:, None]


def bezier(P, s):
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = P
    q = 1 - s
    x = q ** 3 * x0 + 3 * q * q * s * x1 + 3 * q * s * s * x2 + s ** 3 * x3
    y = q ** 3 * y0 + 3 * q * q * s * y1 + 3 * q * s * s * y2 + s ** 3 * y3
    dx = 3 * q * q * (x1 - x0) + 6 * q * s * (x2 - x1) + 3 * s * s * (x3 - x2)
    dy = 3 * q * q * (y1 - y0) + 6 * q * s * (y2 - y1) + 3 * s * s * (y3 - y2)
    return x, y, dx, dy


# ---- backdrops ------------------------------------------------------------------------------
def sky(W, H, rng, stops, bottom, density=0.26):
    """A stippled vertical gradient down to `bottom`; returns an (H, W, 3) float image.
    The random draws do not depend on the colours, so skies built with the same seed line up."""
    grad = gradient(stops, np.arange(H) / H)
    img = np.repeat((grad * 0.8)[:, None, :], W, axis=1)
    n = int(W * bottom * density)
    x, y = rng.uniform(0, W, n), rng.uniform(0, bottom, n)
    col = gradient(stops, y / H + rng.normal(0, 0.025, n)) * (1 + rng.normal(0, 0.045, (n, 1)))
    odd = rng.random(n) < 0.03
    col[odd] = gradient(stops, y[odd] / H + rng.choice([-0.09, 0.09], int(odd.sum())))
    stamp(img, x, y, col.astype(np.float32), rng.choice([1.0, 1.5, 2.0], n, p=[0.3, 0.45, 0.25]))
    return img


def warm_glow(img, cx, cy, R, col, squash=1.0):
    """A smooth coloured haze, e.g. the horizon around a low sun."""
    H, W = img.shape[:2]
    x0, x1 = max(0, int(cx - 3 * R)), min(W, int(cx + 3 * R))
    y0, y1 = max(0, int(cy - 3 * R * squash)), min(H, int(cy + 3 * R * squash))
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d2 = ((xx - cx) / R) ** 2 + ((yy - cy) / (R * squash)) ** 2
    img[y0:y1, x0:x1] += np.asarray(col, np.float32) * np.exp(-d2)[..., None]


def stars(img, rng, n, ymax, b):
    x, y = rng.uniform(0, img.shape[1], n), rng.random(n) ** 1.3 * ymax
    lum = rng.uniform(0.2, 1.0, n) ** 2 * b
    rad = rng.choice([1.0, 1.6], n, p=[0.85, 0.15])
    if b > 0:
        glow(img, x, y, np.array([210, 215, 255]) * lum[:, None], rad)


def moon(img, rng, mx, my, mr, b):
    """A moon with seas and a soft halo (random draws happen even when b is 0)."""
    n = 2400
    a, r = rng.uniform(0, 2 * np.pi, n), mr * (1.05 + rng.random(n) ** 1.8 * 1.6)
    hc = np.array([34, 34, 44]) * (1 - (r / mr - 1) / 1.7)[:, None] * b
    n2 = int(6000 * (mr / 50) ** 2) + 200
    a2, r2 = rng.uniform(0, 2 * np.pi, n2), mr * np.sqrt(rng.random(n2))
    pal = shades(rng, [(252, 248, 228), (238, 234, 212), (222, 218, 198)], rng.random(n2), 0.8)
    rad = rng.choice([1.0, 1.5], n2, p=[0.6, 0.4])
    if b <= 0:
        return
    glow(img, mx + np.cos(a) * r, my + np.sin(a) * r, hc, 1.5)
    px, py = mx + np.cos(a2) * r2, my + np.sin(a2) * r2
    seas = sum(w * np.exp(-((px - mx - ox * mr) ** 2 + (py - my - oy * mr) ** 2) / (sr * mr) ** 2)
               for ox, oy, sr, w in ((-0.3, -0.2, 0.35, 0.22), (0.25, 0.1, 0.3, 0.18), (0.05, 0.45, 0.22, 0.15)))
    col = pal * ((1 - seas) * (1 - 0.18 * (r2 / mr) ** 3))[:, None]
    under = img[np.clip(py.astype(int), 0, img.shape[0] - 1), np.clip(px.astype(int), 0, img.shape[1] - 1)]
    stamp(img, px, py, under + (col - under) * b, rad)


def cloud_streaks(img, rng, items, lit_pal, shade_pal):
    """Long thin clouds lit from below."""
    H, W = img.shape[:2]
    for cx, cy, a, b, n in items:
        n = int(n * W * H / 1e6)
        dx = np.clip(rng.normal(0, 0.42, n), -1, 1)
        thick = (1 - dx ** 2) ** 0.8
        dy = rng.normal(0, 0.5, n) * thick
        x = (cx + a * dx) * W
        y = (cy + b * dy) * H + 0.006 * H * np.sin(dx * 5 + cx * 10)
        under = np.clip(dy / (thick + 1e-3) + 0.5, 0, 1) ** 0.8
        col = mix(shades(rng, shade_pal, rng.random(n)), shades(rng, lit_pal, rng.random(n)), under)
        stamp(img, x, y, col, rng.choice([1.0, 1.5, 2.0], n, p=[0.3, 0.45, 0.25]))


def cloud_puffs(img, rng, items, lit_pal, shade_pal, light_xy):
    """Heaped clouds with flat bases, lit from the side facing (light_xy)."""
    H, W = img.shape[:2]
    for cx, cy, size, n in items:
        cx, cy, size = cx * W, cy * H, size * W
        n = int(n * W * H / 1e6)
        k = 10
        bx = cx + size * rng.normal(0, 0.40, k)
        by = cy - size * 0.16 * np.abs(rng.normal(0, 1, k)) * (1 - np.abs(bx - cx) / (1.4 * size)).clip(0.2, 1)
        br = size * rng.uniform(0.18, 0.36, k)
        which = rng.integers(0, k, n)
        a, r = rng.uniform(0, 2 * np.pi, n), rng.random(n) ** 0.75   # dense cores, airy edges
        x = bx[which] + np.cos(a) * r * br[which]
        y = by[which] + np.sin(a) * r * br[which] * 0.75
        y = np.minimum(y, cy + size * 0.10 + rng.normal(0, size * 0.025, n))   # a soft, flattish base
        d = unit((light_xy[0] - cx, light_xy[1] - cy))
        f = ((x - cx) * d[0] + (y - cy) * d[1]) / size + 0.45 * (cy - y) / size
        col = mix(shades(rng, shade_pal, rng.random(n)), shades(rng, lit_pal, rng.random(n)),
                  np.clip(0.45 + 0.9 * f, 0, 1))
        stamp(img, x, y, col, rng.choice([1.0, 1.5, 2.0, 2.5], n, p=[0.2, 0.35, 0.3, 0.15]))


def hill_layers(img, rng, layers, bottom):
    """Stippled rolling hills, back to front. layers: (base, amplitude, colour, f1, f2, phase),
    base and amplitude as fractions of the height. Returns the top line of every layer."""
    H, W = img.shape[:2]
    xs = np.arange(W, dtype=np.float32)
    yy = np.arange(H, dtype=np.float32)[:, None]
    tops = []
    for base, amp, col, f1, f2, phase in layers:
        top = H * (base - amp * (0.55 + 0.45 * np.sin(xs / W * 2 * np.pi * f1 + phase))
                   - amp * 0.35 * np.sin(xs / W * 2 * np.pi * f2 + 2 * phase))
        tops.append(top.astype(np.float32))
        col = np.asarray(col, np.float32)
        fill = (yy >= top[None, :]) & (yy < bottom)
        img[fill] = col * 0.82
        n = int(W * (bottom - top.min()) * 0.45)
        x = rng.uniform(0, W, n)
        ytop = np.interp(x, xs, top)
        depth = rng.random(n)
        y = ytop + depth * (bottom - ytop)
        c = col * (1.08 - 0.2 * depth)[:, None] * (1 + rng.normal(0, 0.07, (n, 1)))
        stamp(img, x, y, c.astype(np.float32), rng.choice([1.0, 1.5, 2.0], n, p=[0.3, 0.45, 0.25]))
    return tops


class Backdrop:
    """A few full background images at key times, cross-faded per frame."""

    def __init__(self, times, images):
        self.times, self.images = list(times), images

    def at(self, t):
        ts = self.times
        if t <= ts[0]:
            return self.images[0].copy()
        if t >= ts[-1]:
            return self.images[-1].copy()
        i = int(np.searchsorted(ts, t, side="right")) - 1
        w = np.float32(ss(t, ts[i], ts[i + 1]))
        out = self.images[i] * (1 - w)
        out += self.images[i + 1] * w
        return out


class Sun:
    """A pointillist sun: a textured disc and a wide glow, coloured when drawn."""

    def __init__(self, rng, R, reach=5.0):
        reach = R * reach
        size = int(2 * reach) + 3
        c = size / 2
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        d = np.hypot(xx - c, yy - c)
        out = np.maximum(d - R, 0) / R
        halo = (0.75 * np.exp(-out / 0.8) + 0.35 * np.exp(-out / 2.6)) * np.clip(1 - d / reach, 0, 1) ** 0.5
        tmp = np.zeros((size, size, 3), np.float32)
        n = int(7000 * (reach / 300.0) ** 2) + 500
        a, r = rng.uniform(0, 2 * np.pi, n), R + (reach - R) * rng.random(n) ** 2.0
        w = np.clip(1 - (r - R) / (reach - R), 0, 1) ** 2 * 0.45 * ss(r, 1.05 * R, 1.8 * R)
        glow(tmp, c + np.cos(a) * r, c + np.sin(a) * r, np.repeat(w[:, None], 3, 1), 1.5)
        self.halo = (halo + tmp[..., 0]).astype(np.float32)
        tex = np.full((size, size, 3), 0.94, np.float32)
        n = int(np.pi * R * R * 0.9) + 10
        a, rr = rng.uniform(0, 2 * np.pi, n), R * np.sqrt(rng.random(n))
        stamp(tex, c + np.cos(a) * rr, c + np.sin(a) * rr, np.repeat(rng.uniform(0.82, 1.06, (n, 1)), 3, 1),
              rng.choice([1.0, 1.5], n))
        self.tex = tex[..., 0]
        self.alpha = np.clip(R + 0.5 - d, 0, 1).astype(np.float32)
        self.c, self.size, self.r = c, size, int(math.ceil(R))

    def draw(self, img, sx, sy, core, glow_col, disc_mask=None, glow_mask=None):
        H, W = img.shape[:2]
        size, c = self.size, self.c
        x0, y0 = int(round(sx - c)), int(round(sy - c))
        r = self.r + 2
        xa, ya = max(int(round(sx - r)), 0, x0), max(int(round(sy - r)), 0, y0)
        xb, yb = min(int(round(sx + r)) + 1, W, x0 + size), min(int(round(sy + r)) + 1, H, y0 + size)
        if xa < xb and ya < yb:                         # the disc
            dst = (slice(ya, yb), slice(xa, xb))
            src = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
            a = self.alpha[src] * (disc_mask[dst] if disc_mask is not None else 1.0)
            a = a[..., None]
            img[dst] = img[dst] * (1 - a) + np.asarray(core, np.float32) * self.tex[src][..., None] * a
        xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + size, W), min(y0 + size, H)
        if xa < xb and ya < yb:                         # the glow, over everything
            dst = (slice(ya, yb), slice(xa, xb))
            src = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
            h = self.halo[src] * (glow_mask[dst] if glow_mask is not None else 1.0)
            img[dst] += h[..., None] * np.asarray(glow_col, np.float32)


class Moon(Sun):
    """The Sun sprite with grey seas on the disc and a quieter halo."""

    def __init__(self, rng, R, reach=3.0):
        super().__init__(rng, R, reach)
        yy, xx = np.mgrid[0:self.size, 0:self.size].astype(np.float32)
        c = self.c
        seas = sum(w * np.exp(-((xx - c - ox * R) ** 2 + (yy - c - oy * R) ** 2) / (sr * R) ** 2)
                   for ox, oy, sr, w in ((-0.3, -0.2, 0.35, 0.24), (0.25, 0.1, 0.3, 0.2), (0.05, 0.45, 0.22, 0.16),
                                         (-0.15, 0.3, 0.18, 0.12)))
        d = np.clip(np.hypot(xx - c, yy - c) / R, 0, 1)
        self.tex = (self.tex * (1 - seas) * (1 - 0.18 * d ** 3)).astype(np.float32)


class Haze:
    """Soft light gathered on a coarse grid, then smoothed up to full size and added:
    halos round fireflies and windows, beams, glowing water. Collect with add(), finish with apply()."""

    def __init__(self, W, H, k=4):
        self.W, self.H, self.k = W, H, k
        self.buf = np.zeros((H // k + 3, W // k + 3, 3), np.float32)
        self.pending = []

    def add(self, x, y, col, rad=8.0):
        """rad in full-size pixels (up to about 3k)."""
        k = self.k
        x = (np.asarray(x, np.float32) - (k - 1) / 2) / k
        y = (np.asarray(y, np.float32) - (k - 1) / 2) / k
        glow(self.buf, x, y, col, np.clip(np.asarray(rad, np.float32) / k, 0.5, 3.0))

    def splat(self, x, y, col):
        """Drop each point's light into its nearest grid cell - very cheap; apply() with more blur softens it.
        Points are gathered and summed in one go when the layer is applied."""
        k = self.k
        bh, bw = self.buf.shape[:2]
        xi = np.rint((np.asarray(x, np.float32) - (k - 1) / 2) / k).astype(np.int64)
        yi = np.rint((np.asarray(y, np.float32) - (k - 1) / 2) / k).astype(np.int64)
        ok = (xi >= 0) & (xi < bw) & (yi >= 0) & (yi < bh)
        if ok.any():
            self.pending.append((yi[ok] * bw + xi[ok], np.broadcast_to(np.asarray(col, np.float32), (len(xi), 3))[ok]))

    def flush(self):
        if not self.pending:
            return
        bh, bw = self.buf.shape[:2]
        idx = np.concatenate([p[0] for p in self.pending])
        col = np.concatenate([p[1] for p in self.pending])
        self.pending = []
        for c in range(3):
            self.buf[..., c] += np.bincount(idx, col[:, c], bw * bh).reshape(bh, bw)

    def apply(self, img, gain=1.0, blur=1, mask=None, mask_from=0, mask_to=None):
        """Add the light to img and clear the grid. With mask (an (H, W) weight) the light is
        multiplied by it in rows mask_from..mask_to (rows outside are taken as fully open)."""
        self.flush()
        b = self.buf
        if not b.any():
            return
        for _ in range(blur):                           # a [1 2 1] blur along each axis
            c = b * 0.5
            c[1:] += 0.25 * b[:-1]
            c[:-1] += 0.25 * b[1:]
            b = c * 0.5
            b[:, 1:] += 0.25 * c[:, :-1]
            b[:, :-1] += 0.25 * c[:, 1:]
        bh, bw = b.shape[:2]
        H, W, k = self.H, self.W, self.k
        lit = b.max(axis=2) * gain > 0.5
        rows, cols = np.nonzero(lit.any(1))[0], np.nonzero(lit.any(0))[0]
        if len(rows):                                   # only the box that has any light
            r0, r1 = max(int(rows[0]) - 2, 0), min(int(rows[-1]) + 3, bh)
            c0, c1 = max(int(cols[0]) - 2, 0), min(int(cols[-1]) + 3, bw)
            q = np.clip(b[r0:r1, c0:c1] * gain + 0.5, 0, 255).astype(np.uint8)
            up = np.asarray(Image.fromarray(q, "RGB").resize(((c1 - c0) * k, (r1 - r0) * k), Image.BILINEAR))
            ya, yb = r0 * k, min(H, r1 * k)
            xa, xb = c0 * k, min(W, c1 * k)
            if ya < yb and xa < xb:
                up = up[:yb - ya, :xb - xa]
                if mask is None:
                    img[ya:yb, xa:xb] += up
                else:
                    m0 = max(min(mask_from, yb), ya)
                    m1 = max(min(H if mask_to is None else mask_to, yb), m0)
                    if m0 > ya:
                        img[ya:m0, xa:xb] += up[:m0 - ya]
                    if m1 > m0:
                        img[m0:m1, xa:xb] += up[m0 - ya:m1 - ya] * mask[m0:m1, xa:xb, None]
                    if yb > m1:
                        img[m1:yb, xa:xb] += up[m1 - ya:]
        self.buf[:] = 0


# ---- window and exports -----------------------------------------------------------------------
def save_png(make, prefix, stages):
    scene = make(1000, 1000)
    strip = Image.new("RGB", (500 * len(stages), 500))
    for i, t in enumerate(stages):
        frame = Image.fromarray(scene.render(t))
        frame.save(f"{prefix}_{i + 1}.png")
        strip.paste(frame.resize((500, 500), Image.LANCZOS), (500 * i, 0))
    strip.save(f"{prefix}_stages.png")
    print(f"Saved: {prefix}_stages.png and {prefix}_1..{len(stages)}.png")


def save_gif(make, prefix, t_end, fps=12, size=640, wrap=False):
    scene = make(size, size)
    count = int(t_end * fps) + (0 if wrap else 1)                    # a seamless loop must not repeat its first frame
    frames = [Image.fromarray(scene.render(i / fps)).convert("P", palette=Image.ADAPTIVE) for i in range(count)]
    frames[0].save(f"{prefix}.gif", save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=False)
    print(f"Saved: {prefix}.gif")


class Viewer:
    def __init__(self, make, title, prefix, t_end, loop=False, win_frac=0.92, wrap=False):
        import tkinter as tk
        from PIL import ImageTk

        self.prefix, self.t_end, self.loop, self.wrap = prefix, t_end, loop, wrap
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg="black")
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        size = int(min(sh * win_frac, sw * 0.96))
        self.root.geometry(f"{size}x{size}+{(sw - size) // 2}+{max(0, (sh - size) // 2 - 20)}")
        self.canvas = tk.Canvas(self.root, width=size, height=size, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_text(size // 2, size // 2, text="planting...", fill="#777777")
        self.root.update()

        self.scene = make(size, size)
        self.photo = ImageTk.PhotoImage(Image.fromarray(self.scene.render(0.0)))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("<space>", lambda e: self.toggle())
        self.root.bind("<r>", lambda e: self.restart())
        self.root.bind("<s>", lambda e: self.snapshot())
        self.clock, self.last, self.paused = 0.0, time.perf_counter(), False
        self.tick()

    def toggle(self):
        self.paused = not self.paused

    def restart(self):
        self.clock = 0.0

    def now(self):
        return self.clock % self.t_end if self.wrap else min(self.clock, self.t_end)

    def snapshot(self):
        Image.fromarray(self.scene.render(self.now())).save(f"{self.prefix}_frame.png")

    def tick(self):
        now = time.perf_counter()
        if not self.paused:
            self.clock += now - self.last
        self.last = now
        if self.loop and self.clock > self.t_end + 3:
            self.clock = 0.0
        self.photo.paste(Image.fromarray(self.scene.render(self.now())))
        self.root.after(10, self.tick)

    def run(self):
        self.root.mainloop()


def run(make, title, prefix, t_end, stages, argv=None, loop=False, wrap=False):
    """wrap=True: the piece is a seamless loop, time runs round and round."""
    argv = sys.argv[1:] if argv is None else argv
    if "--png" in argv:
        save_png(make, prefix, stages)
    elif "--gif" in argv:
        save_gif(make, prefix, t_end, wrap=wrap)
    else:
        Viewer(make, title, prefix, t_end, loop, wrap=wrap).run()
