"""
benchmark.py - how long each animation takes to draw a frame.

    python tools/benchmark.py              every animation at 1000 x 1000
    python tools/benchmark.py --size 640   at another size
    python tools/benchmark.py fuji kelp    just these

Each scene is built once, then timed over frames spread evenly across its minute (the first
frame is drawn beforehand and not counted). Results depend heavily on the machine.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_previews import CATALOG, load  # noqa: E402


def main(argv):
    size = int(argv[argv.index("--size") + 1]) if "--size" in argv else 1000
    skip = {argv[argv.index("--size") + 1]} if "--size" in argv else set()
    want = [a for a in argv if not a.startswith("-") and a not in skip] or list(CATALOG)
    print(f"{'animation':18s} {'build s':>8s} {'ms/frame':>9s} {'fps':>6s}   ({size} x {size})")
    for key in want:
        rel, cls, args, t0, stages = CATALOG[key]
        mod = load(rel)
        t_end = getattr(mod, "T_END", 54.0)
        a = time.perf_counter()
        scene = getattr(mod, cls)(size, size, *args)
        build = time.perf_counter() - a
        times = [t_end * (k + 0.5) / 12 for k in range(12)]
        scene.render(times[0])
        a = time.perf_counter()
        for t in times:
            scene.render(t)
        ms = (time.perf_counter() - a) * 1000 / len(times)
        print(f"{key:18s} {build:8.2f} {ms:9.1f} {1000 / ms:6.1f}")


if __name__ == "__main__":
    main(sys.argv[1:])
