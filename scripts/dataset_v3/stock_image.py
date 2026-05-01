"""Génère des images sources procédurales variées pour SVD-XT.

Pas de SDXL/SD-Turbo nécessaire — le world-model V3 doit apprendre des
dynamiques latentes depuis des observations photoréalistes. Pour le
bootstrap, on injecte des images structurées (gradients + textures +
shapes) qui ont assez de variété visuelle pour que SVD-XT produise des
videos non-dégénérées.

L'optical flow comme action proxy capture la dynamique injectée par SVD.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _seeded(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def gradient_two_color(rng: np.random.Generator, size: int = 256) -> Image.Image:
    """Diagonal gradient between two random colors + noise."""
    c1 = rng.integers(0, 256, size=3)
    c2 = rng.integers(0, 256, size=3)
    angle = rng.uniform(0, 2 * np.pi)
    yy, xx = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size), indexing="ij")
    t = (np.cos(angle) * xx + np.sin(angle) * yy + 1) / 2
    arr = (c1[None, None, :] * (1 - t)[..., None] + c2[None, None, :] * t[..., None]).astype(np.uint8)
    arr = arr + rng.integers(-15, 15, size=arr.shape, dtype=np.int16).astype(np.int16)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def shapes_on_color(rng: np.random.Generator, size: int = 256) -> Image.Image:
    """Random colored shapes on a softly-colored background."""
    bg = rng.integers(60, 200, size=3)
    img = Image.new("RGB", (size, size), tuple(int(c) for c in bg))
    drw = ImageDraw.Draw(img)
    n = int(rng.integers(3, 7))
    for _ in range(n):
        kind = rng.choice(["rect", "ellipse", "line"])
        col = tuple(int(c) for c in rng.integers(0, 256, size=3))
        x0, y0 = int(rng.integers(0, size)), int(rng.integers(0, size))
        x1, y1 = int(rng.integers(x0, size)), int(rng.integers(y0, size))
        if kind == "rect":
            drw.rectangle([x0, y0, x1, y1], fill=col)
        elif kind == "ellipse":
            drw.ellipse([x0, y0, x1, y1], fill=col)
        else:
            drw.line([x0, y0, x1, y1], fill=col, width=int(rng.integers(2, 10)))
    img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0, 2.5))))
    return img


def stripes(rng: np.random.Generator, size: int = 256) -> Image.Image:
    n_stripes = int(rng.integers(4, 12))
    angle = rng.uniform(0, np.pi)
    yy, xx = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size), indexing="ij")
    t = np.cos(angle) * xx + np.sin(angle) * yy
    pattern = ((np.sin(t * n_stripes * np.pi) + 1) * 127).astype(np.uint8)
    c = rng.integers(0, 256, size=3)
    rgb = np.stack([pattern * (c[i] / 255.0) for i in range(3)], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb)


def checker(rng: np.random.Generator, size: int = 256) -> Image.Image:
    n = int(rng.integers(4, 10))
    cell = size // n
    img = Image.new("RGB", (size, size), (255, 255, 255))
    drw = ImageDraw.Draw(img)
    c1 = tuple(int(c) for c in rng.integers(20, 220, size=3))
    c2 = tuple(int(c) for c in rng.integers(20, 220, size=3))
    for i in range(n + 1):
        for j in range(n + 1):
            col = c1 if (i + j) % 2 == 0 else c2
            drw.rectangle([i * cell, j * cell, (i + 1) * cell, (j + 1) * cell], fill=col)
    return img


GENERATORS = [gradient_two_color, shapes_on_color, stripes, checker]


def make_for_class(class_name: str, seed: int, size: int = 256) -> Image.Image:
    """Choose generator deterministically per class to keep classes visually distinct.

    indoor_manipulation → shapes_on_color (objects on bg)
    navigation_pov     → gradient_two_color (perspective-like)
    outdoor_slow       → stripes (horizon-like)
    human_gesture      → checker (focused background)
    """
    mapping = {
        "indoor_manipulation": shapes_on_color,
        "navigation_pov": gradient_two_color,
        "outdoor_slow": stripes,
        "human_gesture": checker,
    }
    fn = mapping.get(class_name, GENERATORS[seed % len(GENERATORS)])
    return fn(_seeded(seed), size=size)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--class", dest="cls", default="indoor_manipulation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=256)
    args = p.parse_args()
    img = make_for_class(args.cls, args.seed, args.size)
    img.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
