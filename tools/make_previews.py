"""
make_previews.py - render the README previews: a short GIF of each animation's signature moment
(assets/previews/) and a five-stage strip of its whole minute (assets/images/).

    python tools/make_previews.py                 everything
    python tools/make_previews.py fuji diwali     just these
    python tools/make_previews.py banner          the banner at the top of the README
    python tools/make_previews.py --list          show what can be built

GIFs are encoded with ffmpeg (palettegen/paletteuse) when it is on PATH, otherwise with Pillow.
The originals are never touched: this only reads the animations and writes preview files.
"""
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PREVIEWS = ROOT / "assets" / "previews"
IMAGES = ROOT / "assets" / "images"
SIZE, FPS, CLIP = 360, 10, 5.0          # preview GIFs: 360 x 360, 10 frames a second, 5 seconds

# key: (script, class, extra args, clip start (s), five stage times)
CATALOG = {
    "sunflower":        ("garden/sunflower.py", "Sunflower", (), 21.0, None),
    "lotus":            ("garden/lotus.py", "Lotus", (), 30.0, None),
    "sakura":           ("garden/sakura.py", "Sakura", (), 46.0, None),
    "koi_pond":         ("garden/koi_pond.py", "KoiPond", (), 19.0, None),
    "fireflies":        ("garden/fireflies.py", "Fireflies", (), 41.0, None),
    "glow_beach":       ("garden/glow_beach.py", "GlowBeach", (), 8.0, None),
    "aurora":           ("garden/aurora.py", "Aurora", (), 47.0, None),
    "kelp":             ("garden/kelp.py", "Kelp", (), 28.0, None),
    "night_bloom":      ("garden/night_bloom.py", "NightBloom", (), 31.0, None),
    "black_hole":       ("garden/black_hole.py", "BlackHole", (), 8.0, None),
    "golden_hour":      ("garden/golden_hour.py", "GoldenHour", (), 45.0, None),
    "fuji":             ("garden/fuji.py", "Fuji", (), 41.5, None),
    "diwali":           ("garden/diwali.py", "Diwali", (), 45.5, None),
    "neon_alley":       ("neon/neon_alley.py", "NeonAlley", (), 29.5, None),
    "holo_sakura":      ("neon/holo_sakura.py", "HoloSakura", (), 44.0, None),
    "synthwave_drive":  ("neon/synthwave_drive.py", "SynthwaveDrive", (), 10.0, None),
    "zeppelin_harbor":  ("neon/zeppelin_harbor.py", "ZeppelinHarbor", (), 44.0, None),
    "iron_streamliner": ("neon/iron_streamliner.py", "IronStreamliner", (), 8.0, None),
    "dandelion":        ("dandelion/dandelion.py", "Dandelion", ("night",), 40.0, [5.5, 14.5, 25.0, 34.5, 41.0]),
    "dandelion_sunset": ("dandelion/dandelion.py", "Dandelion", ("sunset",), 40.0, [5.5, 14.5, 25.0, 34.5, 41.0]),
    "dandelion_sunrise": ("dandelion/dandelion.py", "Dandelion", ("sunrise",), 40.0, [5.5, 14.5, 25.0, 34.5, 41.0]),
}


def load(rel):
    path = ROOT / "animations" / rel
    sys.path.insert(0, str(path.parent))
    name = f"preview_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_gif(frames, out, fps):
    """Frames (a list of H x W x 3 uint8 arrays) to an optimised looping GIF."""
    if shutil.which("ffmpeg"):
        with tempfile.TemporaryDirectory() as tmp:
            for i, f in enumerate(frames):
                Image.fromarray(f).save(Path(tmp) / f"{i:04d}.png")
            # ordered (Bayer) dithering keeps the dots crisp and compresses far better than error diffusion
            vf = "split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=3"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(Path(tmp) / "%04d.png"),
                            "-vf", vf, "-loop", "0", str(out)], check=True)
    else:
        imgs = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE) for f in frames]
        imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)


def strip(frames, out, tile=320, gap=4):
    w = tile * len(frames) + gap * (len(frames) - 1)
    canvas = Image.new("RGB", (w, tile), (18, 18, 22))
    for i, f in enumerate(frames):
        canvas.paste(Image.fromarray(f).resize((tile, tile), Image.LANCZOS), (i * (tile + gap), 0))
    canvas.save(out, quality=88, optimize=True)


def build(key):
    rel, cls, args, t0, stages = CATALOG[key]
    mod = load(rel)
    stages = stages or list(mod.STAGES)
    scene = getattr(mod, cls)(SIZE, SIZE, *args)
    n = int(CLIP * FPS)
    frames = [scene.render(t0 + i / FPS) for i in range(n)]
    write_gif(frames, PREVIEWS / f"{key}.gif", FPS)
    big = getattr(mod, cls)(640, 640, *args)
    strip([big.render(t) for t in stages], IMAGES / f"{key}_stages.jpg")


# ---- the ASCII rose: render its coloured characters to images ---------------------------------------
def mono_font(px):
    for name in ("consola.ttf", "DejaVuSansMono.ttf", "Menlo.ttc", "LiberationMono-Regular.ttf", "cour.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()


def build_rose():
    mod = load("ascii-rose/rose.py")
    rose = mod.Rose()
    cols, rows, cw, ch = 96, 44, 7, 14
    font = mono_font(12)
    frames = []
    for i in range(80):                                              # the bloom, 0-16 s, at double speed
        t = i / 5.0
        chars, colours = mod.render(rose, t, cols, rows, name="you")
        img = Image.new("RGB", (cols * cw, rows * ch), (6, 6, 10))
        d = ImageDraw.Draw(img)
        for y in range(rows):
            for x in range(cols):
                c = chars[y, x]
                if c != " ":
                    v = int(colours[y, x])
                    d.text((x * cw, y * ch), c, fill=((v >> 16) & 255, (v >> 8) & 255, v & 255), font=font)
        frames.append(np.asarray(img))
    write_gif(frames, PREVIEWS / "ascii_rose.gif", 10)
    picks = [frames[k] for k in (5, 25, 45, 62, 79)]
    h, w = picks[0].shape[:2]
    canvas = Image.new("RGB", (w * 5 + 16, h), (18, 18, 22))
    for i, f in enumerate(picks):
        canvas.paste(Image.fromarray(f), (i * (w + 4), 0))
    canvas.save(IMAGES / "ascii_rose_stages.jpg", quality=88, optimize=True)


BANNER = [("fuji", 44.6), ("diwali", 51.3), ("neon_alley", 52.0), ("black_hole", 42.0), ("koi_pond", 23.0), ("aurora", 50.6)]


def build_banner(tile=300, gap=4):
    """One wide strip of six signature frames for the top of the README."""
    canvas = Image.new("RGB", (tile * len(BANNER) + gap * (len(BANNER) - 1), tile), (18, 18, 22))
    for i, (key, t) in enumerate(BANNER):
        rel, cls, args, _, _ = CATALOG[key]
        frame = getattr(load(rel), cls)(900, 900, *args).render(t)
        canvas.paste(Image.fromarray(frame).resize((tile, tile), Image.LANCZOS), (i * (tile + gap), 0))
    canvas.save(IMAGES / "banner.jpg", quality=90, optimize=True)


def main(argv):
    keys = list(CATALOG) + ["ascii_rose"]
    if "banner" in argv:
        IMAGES.mkdir(parents=True, exist_ok=True)
        build_banner()
        print("banner.jpg")
        return
    if "--list" in argv:
        print("\n".join(keys))
        return
    want = [a for a in argv if not a.startswith("-")] or keys
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    for key in want:
        if key not in keys:
            sys.exit(f"unknown animation {key!r}; try --list")
        build_rose() if key == "ascii_rose" else build(key)
        gif = PREVIEWS / f"{key}.gif"
        print(f"{key:18s} {gif.stat().st_size / 1e6:5.2f} MB gif")


if __name__ == "__main__":
    main(sys.argv[1:])
