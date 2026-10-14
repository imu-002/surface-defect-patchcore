"""Dataset access for MVTec AD and for our own local-material photos.

Both use the MVTec AD folder layout:

    <root>/<category>/train/good/*.png            defect-free training images
    <root>/<category>/test/good/*.png             defect-free test images
    <root>/<category>/test/<defect_type>/*.png    defective test images
    <root>/<category>/ground_truth/<defect_type>/<stem>_mask.png   optional masks

For local materials (fabric / leather / wood photos) masks are optional; without
masks only image-level evaluation is possible.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass
class Sample:
    path: str
    label: str                    # "good" or the defect type
    mask_path: Optional[str] = None


@dataclass
class Category:
    name: str
    train: List[str] = field(default_factory=list)
    test: List[Sample] = field(default_factory=list)

    @property
    def defect_types(self) -> List[str]:
        return sorted({s.label for s in self.test if s.label != "good"})


def _images(d: str) -> List[str]:
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXT))


def list_categories(root: str) -> List[str]:
    return sorted(c for c in os.listdir(root) if os.path.isdir(os.path.join(root, c, "train")))


def load_category(root: str, name: str) -> Category:
    base = os.path.join(root, name)
    cat = Category(name, train=_images(os.path.join(base, "train", "good")))
    test_dir = os.path.join(base, "test")
    for label in sorted(os.listdir(test_dir)):
        for p in _images(os.path.join(test_dir, label)):
            mp = None
            if label != "good":
                stem = os.path.splitext(os.path.basename(p))[0]
                for cand in (f"{stem}_mask.png", f"{stem}.png"):
                    q = os.path.join(base, "ground_truth", label, cand)
                    if os.path.exists(q):
                        mp = q
                        break
            cat.test.append(Sample(p, label, mp))
    return cat


def read_image(path: str, size: int) -> np.ndarray:
    """RGB float32 in [0, 1], resized to size x size."""
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    return im.astype(np.float32) / 255.0


def read_mask(path: Optional[str], size: int) -> np.ndarray:
    if path is None:
        return np.zeros((size, size), bool)
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return m > 127


def read_batch(paths: List[str], size: int) -> np.ndarray:
    return np.stack([read_image(p, size) for p in paths])
