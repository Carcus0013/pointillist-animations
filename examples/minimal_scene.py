"""
minimal_scene.py - the smallest complete animation built on the engine, as a starting point.

A stippled sky fades from dusk to night while the sun sets behind rolling hills, and
fireflies wake and blink above them. It shows the pattern every scene in this repository
follows: build everything that never changes once in __init__, return one frame for any
time t from render(t), and hand the class to run().

    python examples/minimal_scene.py          interactive window (Space, R, S, Esc)
    python examples/minimal_scene.py --png    save five stages
    python examples/minimal_scene.py --gif    save the whole 20 seconds as a GIF
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))   # the shared engine

from pointillism import Backdrop, Sun, glow, hill_layers, run, sky, ss  # noqa: E402

T_END = 20.0
STAGES = [1.0, 6.0, 10.0, 14.0, 19.0]
LOOKS = [  # the backdrop at two times; render() cross-fades between them
    dict(t=0.0, sky=[(0.0, (40, 70, 140)), (0.45, (200, 150, 150)), (0.70, (255, 190, 120))],
         hills=[(60, 50, 90), (40, 40, 60)]),
    dict(t=T_END, sky=[(0.0, (8, 10, 34)), (0.45, (30, 30, 70)), (0.70, (90, 50, 80))],
         hills=[(24, 22, 44), (14, 14, 26)]),
]


class Minimal:
    def __init__(self, W, H):
        self.W, self.H = W, H
        images = []
        for L in LOOKS:
            rng = np.random.default_rng(1)          # the same seed for every look, so the dots line up as they fade
            img = sky(W, H, rng, L["sky"], H)
            far, near = L["hills"]
            tops = hill_layers(img, rng, [(0.74, 0.05, far, 1.3, 3.1, 0.4), (0.84, 0.05, near, 0.9, 2.3, 2.0)], H)
            images.append(img)
        self.backdrop = Backdrop([L["t"] for L in LOOKS], images)
        rows = np.arange(H, dtype=np.float32)[:, None]
        self.sky_mask = (rows < tops[0][None, :]).astype(np.float32)   # the sun's disc only shows above the far hills
        self.sun = Sun(np.random.default_rng(2), 0.05 * min(W, H))
        rng = np.random.default_rng(3)
        n = 160                                     # fireflies: where they hover and when they wake
        self.fx, self.fy = rng.uniform(0, W, n), rng.uniform(0.70, 0.95, n) * H
        self.ph, self.wake = rng.uniform(0, 2 * np.pi, n), rng.uniform(8, 15, n)

    def render(self, t):
        img = self.backdrop.at(t)                   # a float image to paint on
        sun_y = self.H * (0.45 + 0.40 * t / T_END)
        self.sun.draw(img, 0.5 * self.W, sun_y, (255, 225, 170), (120, 70, 30), self.sky_mask)
        b = ss(t, self.wake, self.wake + 1.0) * (0.5 + 0.5 * np.sin(3 * t + self.ph)) ** 4
        x = self.fx + 8 * np.sin(0.7 * t + self.ph)
        glow(img, x, self.fy, np.array([200, 255, 110])[None] * (b * 1.6)[:, None], 2.0)   # light that adds up
        return np.clip(img, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    run(Minimal, "Minimal scene", "minimal", T_END, STAGES)
