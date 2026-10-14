"""Figures: heat-map overlays and qualitative result grids."""
from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def overlay(img: np.ndarray, heat: np.ndarray, lo: float, hi: float, alpha: float = 0.5) -> np.ndarray:
    """img RGB [0,1]; heat (S,S) -> RGB [0,1] with a jet heat-map blended on top."""
    h = np.clip((heat - lo) / max(hi - lo, 1e-6), 0, 1)
    jet = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1].astype(np.float32) / 255
    return np.clip((1 - alpha) * img + alpha * jet, 0, 1)


def _contour(img: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    out = (np.clip(img, 0, 1) * 255).astype(np.uint8).copy()
    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnt, -1, color, 1)
    return out


def qualitative_figure(title: str, picks: List[Dict], lo: float, hi: float, path: str, rel: float = 0.5):
    """picks: list of dict(img, heat, mask, label, score, pred). 3 rows: input+GT, heat-map, predicted region."""
    n = len(picks)
    fig, ax = plt.subplots(3, n, figsize=(2.1 * n, 6.6))
    ax = np.atleast_2d(ax).reshape(3, n)
    for j, p in enumerate(picks):
        ax[0, j].imshow(_contour(p["img"], p["mask"], (0, 255, 0)) if p["mask"].any() else p["img"])
        ax[0, j].set_title(p["label"], fontsize=10)
        ax[1, j].imshow(overlay(p["img"], p["heat"], lo, hi))
        ax[1, j].set_xlabel(f"score {p['score']:.1f}", fontsize=8)
        from .classify import region_mask
        pm = region_mask(p["heat"], rel)
        ax[2, j].imshow(_contour(p["img"], pm, (255, 0, 0)))
        if p.get("pred"):
            ax[2, j].set_xlabel(f"predicted: {p['pred']}", fontsize=8)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    ax[0, 0].set_ylabel("input + ground truth", fontsize=9)
    ax[1, 0].set_ylabel("anomaly heat-map", fontsize=9)
    ax[2, 0].set_ylabel("localised region", fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
