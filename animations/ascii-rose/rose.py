#!/usr/bin/env python3
"""
rose.py - a 3D rose that blooms in your terminal, drawn in ASCII.

    python rose.py                       bloom once, then keep turning
    python rose.py --to "Asha"           sign it for someone ("for Asha <3")
    python rose.py --loop                fold back into a bud and re-bloom every ~26 s
    python rose.py --spin 0              hold still
    python rose.py --snapshot 12 --size 110x50 --no-color    print one frame and exit

Ctrl+C quits. Needs numpy. Looks best in a truecolor terminal (Windows Terminal,
iTerm2, most Linux terminals) with the window maximised or the font zoomed out.

How it works: the petals are one long parametric surface wound around the
centre (3.6 petals per turn). Each frame the surface is rebuilt for the current
amount of bloom, rotated, projected onto the character grid with a z-buffer,
and lit; brightness picks both the character and its colour.
"""

import argparse
import math
import os
import shutil
import sys
import time

import numpy as np

TAU = 2 * math.pi

# Materials, and their (shadow, mid, highlight) colours.
PETAL, RIM, SEPAL, STEM, LEAF, VEIN, THORN = range(7)
PALETTE = np.array([
    [(50, 0, 10), (175, 12, 35), (245, 62, 78)],       # petal
    [(95, 4, 22), (215, 40, 62), (255, 122, 132)],     # petal rim
    [(10, 50, 20), (40, 130, 55), (120, 205, 105)],    # sepal
    [(10, 55, 22), (45, 140, 60), (135, 215, 115)],    # stem
    [(6, 42, 16), (32, 135, 52), (135, 225, 115)],     # leaf
    [(30, 95, 40), (95, 195, 100), (190, 248, 170)],   # leaf vein
    [(40, 18, 14), (90, 45, 32), (145, 90, 60)],       # thorn
], dtype=np.float32)

RAMP = np.array(list(".,:;=+*#%@"))       # dark -> bright
BG = "\x1b[48;2;6;6;10m"
DIM = (95, 95, 120)
PINK = (255, 125, 160)
HEART = (235, 30, 60)

# Lighting, in camera space: (right, towards viewer, up).
LIGHT = np.array([-0.45, 0.60, 0.66])
LIGHT /= np.linalg.norm(LIGHT)
HALF = LIGHT + np.array([0.0, 1.0, 0.0])
HALF /= np.linalg.norm(HALF)
CAM_DIST = 7.0
HEAD = 0.56          # size of the flower head relative to the unit rose surface


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def grid_normals(P):
    """Unit normals of a surface sampled on a (rows, cols, 3) grid."""
    n = np.cross(np.gradient(P, axis=0), np.gradient(P, axis=1))
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    return n


def flat(P, N, mat, shade):
    """Flatten a gridded part into point lists: positions, normals, material, shade."""
    grid = P.shape[:-1]
    return (P.reshape(-1, 3),
            np.broadcast_to(N, P.shape).reshape(-1, 3),
            np.broadcast_to(mat, grid).reshape(-1),
            np.broadcast_to(np.float64(shade), grid).reshape(-1))


def bloom(t, loop):
    """0 = tight bud, 1 = open rose. Holds the bud for a beat, then eases open over ~8 s."""
    closing = 0.0
    if loop:
        t = t % 26.0
        closing = float(smoothstep((t - 22.0) / 4.0))
    return float(smoothstep((t - 1.5) / 8.0)) * (1.0 - closing)


class Rose:
    def __init__(self, stem_len=2.4, nx=34, nt=1400):
        self.stem_len = stem_len
        x = np.linspace(0.0, 1.0, nx)
        th = np.linspace(-2 * math.pi, 15 * math.pi, nt)
        X, TH = np.meshgrid(x, th, indexing="ij")
        self.X, self.TH = X, TH
        self.S = (TH - th[0]) / (th[-1] - th[0])            # 0 = outermost petal .. 1 = the heart
        m = np.mod(3.6 * TH, TAU) / math.pi                  # position across one petal, 0..2
        self.OUTLINE = 1 - 0.5 * (1.25 * (1 - m) ** 2 - 0.25) ** 2   # 0.5 at the seams, ~1 mid-petal
        self.PHI_OPEN = 0.88 * (math.pi / 2) * np.exp(-TH / (8 * math.pi))  # outer petals fall open, the heart stays upright
        self.PHI_BUD = 0.12 + 0.32 * (1 - self.S) ** 2
        # the closed bud: a pointed teardrop, outer petals a little wider, the outermost tips just flaring
        self.BUD_R = self.bud_radius(X, self.S) + 0.05 * (1 - self.S) ** 3 * X ** 4
        self.BUD_Z = 1.05 * X * (0.88 + 0.12 * self.OUTLINE)
        crease = (self.OUTLINE - 0.5) / 0.5
        self.head_shade = ((0.42 + 0.58 * X ** 0.6) * (0.65 + 0.35 * crease)).ravel()
        self.head_mat = np.where(X > 0.93, RIM, PETAL).ravel()
        self.static = self._build_static()

    @staticmethod
    def bud_radius(x, s):
        return (0.11 + 0.24 * (1 - s) ** 0.8) * np.sin(math.pi * x ** 0.62)

    # ---- petals: rebuilt every frame for the current bloom -------------------
    def head(self, b, t):
        # outer petals start opening first, the heart follows
        local = smoothstep(b * 1.4 - self.S * 0.4)
        phi = self.PHI_BUD + (self.PHI_OPEN - self.PHI_BUD) * local
        phi = phi + 0.035 * np.sin(1.7 * t + self.TH * 0.6) * local      # a slow breath
        x = self.X
        curl = 1.95653 * x ** 2 * (1.27689 * x - 1) ** 2 * np.sin(phi)   # petal tips roll outward
        r = self.OUTLINE * (x * np.sin(phi) + curl * np.cos(phi))
        z = self.OUTLINE * (x * np.cos(phi) - curl * np.sin(phi))
        r = (self.BUD_R + (r - self.BUD_R) * local) * HEAD                # unfold from the bud shape
        z = (self.BUD_Z + (z - self.BUD_Z) * local) * HEAD
        P = np.stack([r * np.sin(self.TH), r * np.cos(self.TH), z], axis=-1)
        N = grid_normals(P)
        shade = self.head_shade * (1 - 0.3 * (self.S * local).ravel())    # the heart sits in shadow
        return P.reshape(-1, 3), N.reshape(-1, 3), self.head_mat, shade

    def sepals(self, b):
        k = float(smoothstep(b * 1.6))          # sepals let go early
        s = np.linspace(0, 1, 30)[:, None]
        v = np.linspace(-1, 1, 7)[None, :]
        xb = 0.45 * s                            # closed: wrapped around the lower half of the bud
        rho = (self.bud_radius(xb, 0.0) + 0.03) * (1 - k) + (0.06 + 0.45 * s) * k
        z = 1.05 * xb * (1 - k) + (-0.03 - 0.10 * s - 0.22 * s ** 2) * k
        rho, z = rho * HEAD, z * HEAD
        w = (0.085 * np.sin(math.pi * s ** 0.6) * (1 - 0.5 * s) + 0.004) * HEAD
        out = []
        for i in range(5):
            a = i * TAU / 5 + 0.4
            radial = np.array([math.cos(a), math.sin(a), 0.0])
            side = np.array([-math.sin(a), math.cos(a), 0.0])
            P = (rho[..., None] * radial + (v * w)[..., None] * side
                 + z[..., None] * np.array([0.0, 0.0, 1.0]))
            out.append(flat(P, grid_normals(P), SEPAL, 0.85))
        return [np.concatenate(p) for p in zip(*out)]

    # ---- stem, leaves, thorns: built once ------------------------------------
    def stem_center(self, s):
        s = np.asarray(s, dtype=float)
        return np.stack([0.10 * np.sin(1.15 * math.pi * s),
                         0.05 * np.sin(TAU * s),
                         -self.stem_len * s], axis=-1)

    def _stem(self):
        s = np.linspace(0, 1, 260)[:, None]
        a = np.linspace(0, TAU, 14, endpoint=False)[None, :]
        ring = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=-1)
        rad = 0.05 + 0.015 * s
        P = self.stem_center(s[:, 0])[:, None, :] + rad[..., None] * ring
        return flat(P, ring, STEM, 0.9)

    def _receptacle(self):
        u = np.linspace(0, math.pi, 18)[:, None]
        v = np.linspace(0, TAU, 24, endpoint=False)[None, :]
        N = np.stack([np.sin(u) * np.cos(v), np.sin(u) * np.sin(v), np.cos(u) + 0 * v], axis=-1)
        P = (N * np.array([0.10, 0.10, 0.08]) + np.array([0.0, 0.0, -0.06])) * HEAD
        return flat(P, N, SEPAL, 0.9)

    def _leaf(self, s, angle, tilt, length, width, droop):
        base = self.stem_center(s)
        d = np.array([math.cos(angle) * math.cos(tilt), math.sin(angle) * math.cos(tilt), math.sin(tilt)])
        side = np.array([-math.sin(angle), math.cos(angle), 0.0])
        up = np.cross(d, side)
        u = np.linspace(0, 1, 56)[:, None]
        v = np.linspace(-1, 1, 19)[None, :]
        blade = np.clip((u - 0.12) / 0.88, 0, 1)                         # short bare stalk, then the blade
        w = width * np.sin(math.pi * blade ** 0.75) * (1 + 0.06 * np.sin(u * 50)) + 0.012
        lift = 0.30 * np.abs(v) * w - droop * u ** 2                     # folded along the midrib, tip curls down
        P = base + (u * length)[..., None] * d + (v * w)[..., None] * side + lift[..., None] * up
        av = np.abs(v) + 0 * u
        vein = (av < 0.09) | ((np.mod(u * 9 - av * 1.7, 1.0) < 0.12) & (av < 0.8) & (blade > 0.05))
        return flat(P, grid_normals(P), np.where(vein, VEIN, LEAF), 0.95)

    def _thorn(self, s, angle):
        c = self.stem_center(s)
        out = np.array([math.cos(angle), math.sin(angle), 0.0])
        base = c + out * 0.04
        axis = out * 0.12 + np.array([0.0, 0.0, -0.05])                  # hooked slightly downward
        an = axis / np.linalg.norm(axis)
        e1 = np.cross(an, [0.0, 0.0, 1.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(an, e1)
        h = np.linspace(0, 1, 7)[:, None]
        a = np.linspace(0, TAU, 10, endpoint=False)[None, :]
        ring = np.cos(a)[..., None] * e1 + np.sin(a)[..., None] * e2
        P = base + h[..., None] * axis + (0.035 * (1 - h))[..., None] * ring
        return flat(P, ring, THORN, 0.9)

    def _build_static(self):
        parts = [
            self._stem(),
            self._receptacle(),
            self._leaf(s=0.50, angle=0.25, tilt=0.45, length=0.72, width=0.23, droop=0.26),
            self._leaf(s=0.72, angle=math.pi + 0.45, tilt=0.38, length=0.66, width=0.21, droop=0.22),
            self._thorn(0.22, 2.4),
            self._thorn(0.53, 1.2),
            self._thorn(0.82, 4.0),
        ]
        return [np.concatenate(p) for p in zip(*parts)]


# ---- background ---------------------------------------------------------------
def h01(a, b=0.0, c=0.0):
    v = math.sin(a * 12.9898 + b * 78.233 + c * 37.719) * 43758.5453
    return v - math.floor(v)


def pack(rgb):
    r, g, b = (max(0, min(255, int(x))) & 0xF8 for x in rgb)
    return (r << 16) | (g << 8) | b


def put_cell(chars, cols, y, x, ch, rgb):
    H, W = chars.shape
    if 0 <= y < H and 0 <= x < W:
        chars[y, x] = ch
        cols[y, x] = pack(rgb)


def draw_sky(chars, cols, t, blocked):
    """Pollen, twinkles and flares, drawn only into cells the flower doesn't cover."""
    H, W = chars.shape

    def put(y, x, ch, rgb):
        if 0 <= y < H and 0 <= x < W and not blocked[y, x]:
            chars[y, x] = ch
            cols[y, x] = pack(rgb)

    # pollen: faint dots drifting upward
    for i in range(max(12, W * H // 110)):
        x = int(h01(i, 1.3) * W)
        y = int((h01(i, 2.7) * H - t * (0.4 + 0.8 * h01(i, 4.1))) % H)
        a = 0.30 + 0.25 * math.sin(t * (0.8 + 2 * h01(i, 5.5)) + i)
        put(y, x, ".", (90 * a, 90 * a, 120 * a))
    # twinkles: small stars that fade in and out, then reappear somewhere else
    for i in range(max(6, W * H // 260)):
        ph = t / (2.0 + 3.0 * h01(i, 7.7)) + h01(i, 8.8)
        cyc = math.floor(ph)
        a = math.sin(math.pi * (ph - cyc)) ** 2
        if a < 0.1:
            continue
        x, y = int(h01(i, cyc, 1.1) * W), int(h01(i, cyc, 2.2) * (H - 3))
        put(y, x, "*" if a > 0.75 else "+" if a > 0.4 else ".", (255 * a, 235 * a, 210 * a))
    # flares: four-pointed sparkles with long arms
    for i in range(max(3, W * H // 1100)):
        ph = t / (3.5 + 3.0 * h01(i, 9.9)) + h01(i, 3.3)
        cyc = math.floor(ph)
        a = math.sin(math.pi * (ph - cyc)) ** 3
        if a < 0.08:
            continue
        x, y = int(h01(i, cyc, 5.1) * W), int(h01(i, cyc, 6.2) * (H - 3))
        arm = 1 + int(a * 3.5)
        tint = lambda f: (255 * f, 245 * f, 225 * f)
        for k in range(1, 2 * arm + 1):
            f = a * (1 - k / (2 * arm + 1))
            put(y, x - k, "-", tint(f))
            put(y, x + k, "-", tint(f))
        for k in range(1, arm + 1):
            f = a * (1 - k / (arm + 1))
            put(y - k, x, "|", tint(f))
            put(y + k, x, "|", tint(f))
        if a > 0.7:
            for dy, dx, ch in ((-1, -1, "\\"), (1, 1, "\\"), (-1, 1, "/"), (1, -1, "/")):
                put(y + dy, x + dx, ch, tint(a * 0.45))
        put(y, x, "*" if a > 0.55 else "+", tint(a))


def shift(a, dy, dx):
    """Shift a 2D boolean array, filling with False."""
    out = np.zeros_like(a)
    H, W = a.shape
    out[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = \
        a[max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
    return out


def put_text(chars, cols, y, x, text, rgb):
    for i, ch in enumerate(text):
        put_cell(chars, cols, y, x + i, ch, rgb)


# ---- one frame ------------------------------------------------------------------
def render(rose, t, W, H, *, spin=0.35, elev=0.62, aspect=2.0, loop=False,
           name="you", hud=True, fps=None):
    chars = np.full((H, W), " ", dtype="<U1")
    cols = np.zeros((H, W), dtype=np.int64)

    b = bloom(t, loop)
    parts = [rose.head(b, t), rose.sepals(b), rose.static]
    P = np.concatenate([p[0] for p in parts])
    N = np.concatenate([p[1] for p in parts])
    mat = np.concatenate([p[2] for p in parts])
    shade = np.concatenate([p[3] for p in parts])

    yaw = 0.7 + spin * t
    cy, sy, ce, se = math.cos(yaw), math.sin(yaw), math.cos(elev), math.sin(elev)

    def to_cam(V):                       # -> (right, towards viewer, up)
        x1 = V[:, 0] * cy - V[:, 1] * sy
        y1 = V[:, 0] * sy + V[:, 1] * cy
        return x1, -y1 * ce + V[:, 2] * se, y1 * se + V[:, 2] * ce

    px, pd, pu = to_cam(P)

    # fit the whole flower above the caption line, keeping cells ~2x taller than wide
    top = 0.72
    bottom =-rose.stem_len * ce * CAM_DIST / (CAM_DIST + rose.stem_len * se) - 0.1
    usable = H - 4
    ky = usable / (top - bottom)
    kx = ky * aspect
    if kx * 1.9 > W - 2:
        kx = (W - 2) / 1.9
        ky = kx / aspect
    row0 = 1 + (usable - (top - bottom) * ky) / 2 + top * ky
    f = CAM_DIST / (CAM_DIST - pd)
    ci = np.rint(W / 2 + px * f * kx).astype(np.int64)
    ri = np.rint(row0 - pu * f * ky).astype(np.int64)

    # z-buffer: for each cell keep only the point nearest the viewer
    ok = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
    sel = np.nonzero(ok)[0]
    cell = ri[sel] * W + ci[sel]
    order = np.lexsort((-pd[sel], cell))
    cell_sorted = cell[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = cell_sorted[1:] != cell_sorted[:-1]
    win, cell = sel[order[first]], cell_sorted[first]

    # light the visible points (two-sided: petals show both faces)
    nx, nd, nu = to_cam(N[win])
    flip = np.where(nd < 0, -1.0, 1.0)
    nx, nd, nu = nx * flip, nd * flip, nu * flip
    diff = np.clip(nx * LIGHT[0] + nd * LIGHT[1] + nu * LIGHT[2], 0, 1)
    spec = np.clip(nx * HALF[0] + nd * HALF[1] + nu * HALF[2], 0, 1) ** 22
    sh, m = shade[win], mat[win]
    lum = 0.08 + 0.82 * diff * sh + 0.30 * spec * sh
    lum += np.where(m == RIM, 0.10, 0.0) + np.where(m == VEIN, 0.08, 0.0)
    lum = np.clip(lum, 0, 1)

    pal = PALETTE[m]
    lo = np.clip(lum / 0.55, 0, 1)[:, None]
    hi = np.clip((lum - 0.55) / 0.45, 0, 1)[:, None]
    rgb = np.where(lum[:, None] < 0.55,
                   pal[:, 0] + (pal[:, 1] - pal[:, 0]) * lo,
                   pal[:, 1] + (pal[:, 2] - pal[:, 1]) * hi)
    rgb = np.clip(rgb + spec[:, None] * sh[:, None] * 45, 0, 255).astype(np.int64) & 0xF8
    flat_chars, flat_cols = chars.reshape(-1), cols.reshape(-1)
    flat_chars[cell] = RAMP[np.clip((lum * len(RAMP)).astype(int), 0, len(RAMP) - 1)]
    flat_cols[cell] = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]

    # close single-cell pinholes, copying the neighbour to the left (or above)
    occ = np.zeros(H * W, dtype=bool)
    occ[cell] = True
    occ = occ.reshape(H, W)
    left, right, up, down = shift(occ, 0, 1), shift(occ, 0, -1), shift(occ, 1, 0), shift(occ, -1, 0)
    hy, hx = np.nonzero(~occ & ((left & right) | (up & down)))
    from_left = left[hy, hx]
    sy, sx = np.where(from_left, hy, hy - 1), np.where(from_left, hx - 1, hx)
    chars[hy, hx], cols[hy, hx] = chars[sy, sx], cols[sy, sx]
    occ[hy, hx] = True

    # the sky stays behind the flower, with a one-cell margin around it
    blocked = occ | shift(occ, 1, 0) | shift(occ, -1, 0)
    blocked = blocked | shift(blocked, 0, 1) | shift(blocked, 0, -1)
    draw_sky(chars, cols, t, blocked)

    # the dedication types itself out as the last petals open
    if name:
        text = f"for {name}"
        k = float(np.clip((b - 0.8) / 0.2, 0, 1))
        shown = int(round(k * (len(text) + 2)))
        x0 = (W - len(text) - 2) // 2
        y0 = min(H - 3, int(row0 - bottom * ky) + 1)
        put_text(chars, cols, y0, x0, text[:shown], PINK)
        if shown > len(text) + 1:
            put_cell(chars, cols, y0, x0 + len(text) + 1, "♥", HEART)
        elif 0 < k < 1 and int(t * 4) % 2:
            put_cell(chars, cols, y0, x0 + shown, "_", PINK)

    if hud:
        put_text(chars, cols, 0, 1, "rose.py", DIM)
        clock = f"t {t:5.1f}s"
        put_text(chars, cols, 0, W - len(clock) - 1, clock, DIM)
        line = f"bloom {b:4.2f}   yaw {math.degrees(yaw) % 360:5.1f} deg   points {len(P):,}"
        if fps:
            line += f"   {fps:4.1f} fps"
        put_text(chars, cols, H - 1, 1, line, DIM)

    return chars, cols


# ---- output -----------------------------------------------------------------------
def to_ansi(chars, cols, home=True):
    out = ["\x1b[H" if home else "", BG]
    cache = {}
    for crow, krow in zip(chars.tolist(), cols.tolist()):
        last = -1
        buf = []
        for ch, c in zip(crow, krow):
            if ch != " " and c != last:
                esc = cache.get(c)
                if esc is None:
                    esc = cache[c] = f"\x1b[38;2;{c >> 16};{(c >> 8) & 255};{c & 255}m"
                buf.append(esc)
                last = c
            buf.append(ch)
        out.append("".join(buf))
        out.append("\n")
    out[-1] = "" if home else "\x1b[0m\n"
    return "".join(out)


def to_plain(chars, home=True):
    rows = ["".join(r) for r in chars.tolist()]
    if home:
        return "\x1b[H" + "\n".join(rows)
    return "\n".join(r.rstrip() for r in rows) + "\n"


def enable_vt():
    """Turn on ANSI escape handling in the classic Windows console."""
    if os.name != "nt":
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        handle = k32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if k32.GetConsoleMode(handle, ctypes.byref(mode)):
            k32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="A 3D ASCII rose that blooms in your terminal.")
    ap.add_argument("--to", default="you", metavar="NAME", help='who the rose is for (default "you"; "" hides it)')
    ap.add_argument("--loop", action="store_true", help="fold back into a bud and re-bloom every ~26 s")
    ap.add_argument("--spin", type=float, default=0.35, help="turning speed in radians/second (default 0.35)")
    ap.add_argument("--fps", type=float, default=30.0, help="frame-rate cap (default 30)")
    ap.add_argument("--aspect", type=float, default=2.0, help="character cell height/width (default 2.0)")
    ap.add_argument("--no-hud", action="store_true", help="hide the corner readouts")
    ap.add_argument("--no-color", action="store_true", help="plain characters, no colour")
    ap.add_argument("--snapshot", type=float, metavar="SECONDS", help="print the frame at this time and exit")
    ap.add_argument("--size", metavar="WxH", help="frame size for --snapshot (default: terminal size)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    enable_vt()
    rose = Rose()
    opts = dict(spin=args.spin, aspect=args.aspect, loop=args.loop, name=args.to, hud=not args.no_hud)

    if args.snapshot is not None:
        if args.size:
            W, H = (int(v) for v in args.size.lower().split("x"))
        else:
            ts = shutil.get_terminal_size((100, 40))
            W, H = ts.columns - 1, ts.lines - 1
        chars, cols = render(rose, args.snapshot, W, H, **opts)
        sys.stdout.write(to_plain(chars, home=False) if args.no_color else to_ansi(chars, cols, home=False))
        return

    out = sys.stdout
    out.write("\x1b[?1049h\x1b[?25l" + BG + "\x1b[2J")
    last_size, frame_dt = None, None
    t0 = prev = time.perf_counter()
    try:
        while True:
            start = time.perf_counter()
            if start > prev:
                dt = start - prev
                frame_dt = dt if frame_dt is None else frame_dt * 0.9 + dt * 0.1
            prev = start
            fps = 1.0 / frame_dt if frame_dt else None
            ts = shutil.get_terminal_size((100, 40))
            W, H = max(30, ts.columns - 1), max(14, ts.lines - 1)
            if (W, H) != last_size:
                out.write(BG + "\x1b[2J")
                last_size = (W, H)
            chars, cols = render(rose, start - t0, W, H, fps=fps, **opts)
            out.write(to_plain(chars) if args.no_color else to_ansi(chars, cols))
            out.flush()
            wait = 1.0 / args.fps - (time.perf_counter() - start)
            if wait > 0:
                time.sleep(wait)
    except KeyboardInterrupt:
        pass
    finally:
        out.write("\x1b[0m\x1b[?25h\x1b[?1049l")
        out.flush()


if __name__ == "__main__":
    main()
