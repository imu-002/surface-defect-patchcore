"""Stand-in dataset generator in the MVTec AD folder layout.

PURPOSE: a pipeline check that runs offline. It is NOT a substitute for MVTec AD
and numbers measured on it must never be reported as MVTec results.

Defect-free images come from 3 real textures bundled with scikit-image (brick,
grass, gravel; tinted to RGB) and 3 procedural textures (fabric, wood, leather).
Defects (scratch, crack, hole, stain, color) are painted on test images with
pixel-accurate ground-truth masks. Train and test crops of the real textures come
from disjoint regions of the source photo so no patch is shared.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree

SIZE = 256
DEFECTS = ("scratch", "crack", "hole", "stain", "color")
REAL = {"brick": (1.0, 0.55, 0.42), "grass": (0.48, 0.82, 0.32), "gravel": (0.78, 0.72, 0.62)}
CATEGORIES = ("brick", "grass", "gravel", "fabric", "wood", "leather")


# ------------------------------------------------------------------ textures --
def _smooth_noise(rng, shape, sigma):
    n = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), sigma)
    return n / (n.std() + 1e-6)


def _fabric(rng):
    n, p = SIZE, 8
    ii, jj = np.mgrid[0:n, 0:n]
    warp = 1 + 0.12 * rng.standard_normal(n // p * 2 + 2)
    weft = 1 + 0.12 * rng.standard_normal(n // p * 2 + 2)
    over = ((ii // (p // 2) + jj // (p // 2)) % 2) == 0
    shade = np.where(over, warp[jj // (p // 2)], weft[ii // (p // 2)])
    yarn = 0.75 + 0.25 * np.sin(2 * np.pi * ii / (p / 2)) * np.sin(2 * np.pi * jj / (p / 2))
    v = np.clip(shade * yarn + 0.05 * _smooth_noise(rng, (n, n), 1.0), 0, 2) * 0.5
    return v[..., None] * np.array([0.55, 0.7, 1.0])[None, None]


def _wood(rng):
    n = SIZE
    ii, jj = np.mgrid[0:n, 0:n].astype(np.float32)
    cx, cy = rng.uniform(-n, 2 * n), rng.uniform(-3 * n, -n)
    r = np.sqrt((jj - cx) ** 2 + ((ii - cy) * 0.35) ** 2)
    v = 0.5 + 0.5 * np.sin(2 * np.pi * (r / 22 + 0.15 * _smooth_noise(rng, (n, n), 20)))
    grain = cv2.GaussianBlur(rng.standard_normal((n, n)).astype(np.float32), (0, 0), sigmaX=14, sigmaY=0.8)
    v = np.clip(0.8 * v + 0.35 * grain / grain.std() * 0.3, 0, 1)
    dark, light = np.array([0.25, 0.14, 0.07]), np.array([0.72, 0.5, 0.3])
    return dark + (light - dark) * v[..., None]


def _leather(rng):
    n = SIZE
    pts = rng.uniform(0, n, size=(260, 2))
    tree = cKDTree(np.concatenate([pts + o for o in ([0, 0], [n, 0], [-n, 0], [0, n], [0, -n])]))
    yy, xx = np.mgrid[0:n, 0:n]
    d, _ = tree.query(np.stack([xx.ravel(), yy.ravel()], 1), k=2)
    cell = (d[:, 1] - d[:, 0]).reshape(n, n).astype(np.float32)
    cell = cv2.GaussianBlur(cell / cell.max(), (0, 0), 1.2)
    v = np.clip(0.55 + 0.9 * (cell - cell.mean()) + 0.06 * _smooth_noise(rng, (n, n), 0.8), 0, 1)
    return v[..., None] * np.array([0.62, 0.38, 0.22])[None, None] * 1.3


def _real(name, split, rng):
    import skimage.data as sd
    src = getattr(sd, name)().astype(np.float32) / 255.0
    rows = (0, 240) if split == "train" else (272, 512)
    for _ in range(50):
        c = int(rng.integers(150, 225))
        top = int(rng.integers(rows[0], rows[1] - c + 1))
        left = int(rng.integers(0, 512 - c + 1))
        crop = src[top:top + c, left:left + c]
        if crop.std() > 0.02:
            break
    crop = np.rot90(crop, int(rng.integers(4)))
    if rng.random() < 0.5:
        crop = crop[:, ::-1]
    crop = cv2.resize(np.ascontiguousarray(crop), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    crop = crop * rng.uniform(0.95, 1.05)
    return crop[..., None] * np.array(REAL[name])[None, None] * 1.1


def make_good(category: str, split: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if category in REAL:
        img = _real(category, split, rng)
    else:
        img = {"fabric": _fabric, "wood": _wood, "leather": _leather}[category](rng)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


# ------------------------------------------------------------------- defects --
def _blob(rng, radius, n_vert=14, jitter=0.35):
    c = rng.uniform(radius + 8, SIZE - radius - 8, size=2)
    ang = np.sort(rng.uniform(0, 2 * np.pi, n_vert))
    rad = radius * (1 + jitter * rng.uniform(-1, 1, n_vert))
    poly = np.stack([c[0] + rad * np.cos(ang), c[1] + rad * np.sin(ang)], 1).astype(np.int32)
    m = np.zeros((SIZE, SIZE), np.uint8)
    cv2.fillPoly(m, [poly], 255, lineType=cv2.LINE_AA)
    return m


def _walk(rng, n_steps, step, start, angle, wobble):
    pts, p, a = [start], np.array(start, float), angle
    for _ in range(n_steps):
        a += rng.normal(0, wobble)
        p = p + step * np.array([np.cos(a), np.sin(a)])
        pts.append(p.copy())
    return np.array(pts)


def add_defect(img: np.ndarray, kind: str, rng: np.random.Generator,
               strength: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Paint a defect. strength < 1 scales the change in pixel values (low-contrast / subtle defects);
    the ground-truth mask stays the same."""
    f = img.astype(np.float32) / 255.0
    f0 = f.copy()
    canvas = np.zeros((SIZE, SIZE), np.uint8)
    if kind == "scratch":
        start = rng.uniform(40, SIZE - 40, 2)
        pts = _walk(rng, 14, rng.uniform(5, 9), start, rng.uniform(0, 2 * np.pi), 0.05)
        cv2.polylines(canvas, [pts.astype(np.int32)], False, 255, int(rng.integers(1, 3)), cv2.LINE_AA)
        a = cv2.GaussianBlur(canvas, (0, 0), 0.7).astype(np.float32) / 255
        f = f * (1 - 0.85 * a[..., None]) + 0.85 * a[..., None] * rng.uniform(0.85, 1.0)  # light, smooth
    elif kind == "crack":
        start = rng.uniform(40, SIZE - 40, 2)
        ang = rng.uniform(0, 2 * np.pi)
        main = _walk(rng, 18, rng.uniform(4, 7), start, ang, 0.45)
        cv2.polylines(canvas, [main.astype(np.int32)], False, 255, 2, cv2.LINE_AA)
        for _ in range(2):  # short branches
            k = int(rng.integers(4, 14))
            br = _walk(rng, 5, 4, main[k], ang + rng.choice([-1, 1]) * rng.uniform(0.5, 1.2), 0.4)
            cv2.polylines(canvas, [br.astype(np.int32)], False, 255, 1, cv2.LINE_AA)
        a = cv2.GaussianBlur(canvas, (0, 0), 0.6).astype(np.float32) / 255
        f = f * (1 - 0.92 * a[..., None])  # dark, jagged
    elif kind == "hole":
        canvas = _blob(rng, rng.uniform(6, 13), 10, 0.3)
        a = cv2.GaussianBlur(canvas, (0, 0), 1.0).astype(np.float32) / 255
        f = f * (1 - a[..., None]) + 0.03 * a[..., None]
    elif kind == "stain":
        canvas = _blob(rng, rng.uniform(14, 30), 12, 0.4)
        a = cv2.GaussianBlur(canvas, (0, 0), 4.0).astype(np.float32) / 255
        tint = np.array(rng.choice([[0.35, 0.25, 0.1], [0.15, 0.2, 0.3], [0.45, 0.45, 0.2]]), np.float32)
        f = f * (1 - 0.65 * a[..., None]) + 0.65 * a[..., None] * (tint * (0.5 + f.mean()))
    elif kind == "color":
        canvas = _blob(rng, rng.uniform(14, 28), 16, 0.15)
        a = cv2.GaussianBlur(canvas, (0, 0), 2.0).astype(np.float32) / 255
        gain = rng.choice([[1.5, 0.8, 0.8], [0.8, 1.4, 0.9], [0.8, 0.9, 1.6], [1.4, 1.3, 0.7]])
        f = f * (1 - a[..., None]) + np.clip(f * np.array(gain, np.float32), 0, 1) * a[..., None]
    else:
        raise ValueError(kind)
    mask = (cv2.GaussianBlur(canvas, (0, 0), 1.0) > 60).astype(np.uint8) * 255
    f = f0 + strength * (f - f0)
    return (np.clip(f, 0, 1) * 255).astype(np.uint8), mask


# --------------------------------------------------------------------- dataset --
def write_dataset(root: str, categories=CATEGORIES, n_train=100, n_test_good=40, n_per_defect=16, seed=0, strength=1.0) -> Dict:
    def save(path, arr):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    counts = {}
    for ci, cat in enumerate(categories):
        base = seed * 1_000_003 + ci * 100_003
        for i in range(n_train):
            save(f"{root}/{cat}/train/good/{i:03d}.png", make_good(cat, "train", base + i))
        for i in range(n_test_good):
            save(f"{root}/{cat}/test/good/{i:03d}.png", make_good(cat, "test", base + 50_000 + i))
        for di, d in enumerate(DEFECTS):
            for i in range(n_per_defect):
                s = base + 60_000 + di * 1000 + i
                img, mask = add_defect(make_good(cat, "test", s), d, np.random.default_rng(s + 7), strength)
                save(f"{root}/{cat}/test/{d}/{i:03d}.png", img)
                save(f"{root}/{cat}/ground_truth/{d}/{i:03d}_mask.png", mask)
        counts[cat] = dict(train=n_train, test_good=n_test_good, test_defect=n_per_defect * len(DEFECTS))
    return counts
