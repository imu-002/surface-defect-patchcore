"""PatchCore (Roth et al., CVPR 2022) for CPU: memory bank of normal patch embeddings,
greedy k-center coreset, nearest-neighbour anomaly scoring, localisation heat-maps.

Training uses ONLY defect-free images. Nothing about defect appearance is learned,
so previously unseen defect types are detected the same way as known ones.
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .features import FeatureExtractor


def greedy_coreset(X: np.ndarray, n_select: int, proj_dim: int = 128, seed: int = 0) -> np.ndarray:
    """Greedy k-center selection (indices), run on a random low-dim projection for speed."""
    rng = np.random.default_rng(seed)
    n = len(X)
    n_select = min(n_select, n)
    if X.shape[1] > proj_dim:
        P = (rng.standard_normal((X.shape[1], proj_dim)) / np.sqrt(proj_dim)).astype(np.float32)
        Z = np.concatenate([X[i:i + 50_000] @ P for i in range(0, n, 50_000)])
    else:
        Z = np.ascontiguousarray(X, dtype=np.float32)
    sq = np.einsum("ij,ij->i", Z, Z)
    sel = np.empty(n_select, np.int64)
    sel[0] = int(rng.integers(n))
    mind = sq - 2 * (Z @ Z[sel[0]]) + sq[sel[0]]
    tmp = np.empty_like(mind)
    for i in range(1, n_select):
        j = int(np.argmax(mind))
        sel[i] = j
        np.matmul(Z, Z[j], out=tmp)
        tmp *= -2
        tmp += sq
        tmp += sq[j]
        np.minimum(mind, tmp, out=mind)
    return sel


@dataclass
class Calibration:
    image_thr: float          # decision threshold on the image score (max patch distance)
    mean: float
    std: float
    n: int


class PatchCore:
    def __init__(self, extractor: FeatureExtractor, size: int = 256, coreset_ratio: float = 0.1,
                 max_bank: int = 40_000, pca_dim: Optional[int] = None, k: int = 1,
                 calib_frac: float = 0.15, sigma: float = 4.0, seed: int = 0):
        self.ex, self.size = extractor, size
        self.coreset_ratio, self.max_bank, self.pca_dim, self.k = coreset_ratio, max_bank, pca_dim, k
        self.calib_frac, self.sigma, self.seed = calib_frac, sigma, seed
        self.bank = self.bank_sq = self.pca_mean = self.pca_W = None
        self.cal: Optional[Calibration] = None
        self.fit_seconds = 0.0

    # ------------------------------------------------------------ embeddings --
    def _project(self, F: np.ndarray) -> np.ndarray:
        if self.pca_W is None:
            return F
        return (F - self.pca_mean) @ self.pca_W

    def embed(self, images: np.ndarray) -> np.ndarray:
        """(N,H,W,3) -> (N,h,w,D) patch embeddings (after optional PCA)."""
        F = self.ex(images)
        return self._project(F) if self.pca_W is not None else F

    # ------------------------------------------------------------------- fit --
    def split_train(self, n: int):
        """Indices of (fit images, calibration images). Calibration images are held out of the bank
        and only used to set the decision threshold."""
        order = np.random.default_rng(self.seed).permutation(n)
        n_cal = max(int(round(n * self.calib_frac)), min(8, n // 4)) if n >= 8 else 0
        return order[n_cal:], order[:n_cal]

    def fit_raw(self, fit_feats: np.ndarray, cal_feats: Optional[np.ndarray] = None, verbose: bool = False):
        """fit_feats: (P, 1536) raw backbone patches of the fit images; cal_feats: (n, h, w, 1536) raw
        patches of the held-out calibration images."""
        t0 = time.time()
        rng = np.random.default_rng(self.seed)
        feats = fit_feats
        if self.pca_dim:
            sub = feats[rng.choice(len(feats), min(len(feats), 100_000), replace=False)]
            self.pca_mean = sub.mean(0)
            w, V = np.linalg.eigh(np.cov((sub - self.pca_mean).T))
            self.pca_W = V[:, np.argsort(w)[::-1][: self.pca_dim]].astype(np.float32)
            feats = self._project(feats)
        n_keep = int(min(self.max_bank, max(1, round(len(feats) * self.coreset_ratio))))
        if verbose:
            print(f"  patches {len(feats):,} -> coreset {n_keep:,}", flush=True)
        idx = greedy_coreset(feats, n_keep, seed=self.seed)
        self.bank = np.ascontiguousarray(feats[idx], dtype=np.float32)
        self.bank_sq = np.einsum("ij,ij->i", self.bank, self.bank)
        self.cal = None
        if cal_feats is not None and len(cal_feats):
            s = self.patch_scores(self._project(cal_feats)).reshape(len(cal_feats), -1).max(1)
            std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
            self.cal = Calibration(float(s.mean() + 3 * std), float(s.mean()), std, len(s))
        self.fit_seconds = time.time() - t0
        return self

    def fit(self, images: np.ndarray, verbose: bool = False) -> "PatchCore":
        t0 = time.time()
        fit_idx, cal_idx = self.split_train(len(images))
        feats = np.concatenate([f.reshape(-1, f.shape[-1]) for f in self.ex.iter_features(images[fit_idx])])
        cal = self.ex(images[cal_idx]) if len(cal_idx) else None
        self.fit_raw(feats, cal, verbose)
        self.fit_seconds = time.time() - t0
        return self

    # --------------------------------------------------------------- scoring --
    def nn_distance(self, q: np.ndarray) -> np.ndarray:
        """q: (M, D) -> distance to the k nearest bank patches (mean over k)."""
        d = q @ self.bank.T
        d *= -2
        d += self.bank_sq[None]
        d += np.einsum("ij,ij->i", q, q)[:, None]
        np.maximum(d, 0, out=d)
        if self.k == 1:
            m = d.min(1)
        else:
            m = np.partition(d, self.k - 1, axis=1)[:, : self.k].mean(1)
        return np.sqrt(m)

    def patch_scores(self, emb: np.ndarray) -> np.ndarray:
        """(N,h,w,D) -> (N,h,w)"""
        N, h, w, D = emb.shape
        return np.stack([self.nn_distance(e.reshape(-1, D)).reshape(h, w) for e in emb])

    def score_images(self, images: np.ndarray):
        """returns (image_scores (N,), patch_score_maps (N,h,w))"""
        P = np.concatenate([self.patch_scores(self.embed(images[i:i + 8])) for i in range(0, len(images), 8)])
        return P.reshape(len(P), -1).max(1), P

    def to_heatmaps(self, P: np.ndarray, out_size: Optional[int] = None) -> np.ndarray:
        """Upsample patch score maps to full resolution and smooth (Gaussian), (N,S,S)."""
        S = out_size or self.size
        out = np.stack([cv2.resize(p, (S, S), interpolation=cv2.INTER_LINEAR) for p in P])
        if self.sigma > 0:
            out = np.stack([cv2.GaussianBlur(o, (0, 0), self.sigma * S / self.size) for o in out])
        return out

    def predict(self, images: np.ndarray):
        """-> dict(score, is_defect, patch_scores, heatmap)"""
        s, P = self.score_images(images)
        thr = self.cal.image_thr if self.cal else None
        return dict(score=s, patch_scores=P, heatmap=self.to_heatmaps(P),
                    is_defect=(s > thr) if thr is not None else None)

    # ----------------------------------------------------------- persistence --
    def save(self, path: str):
        state = {k: getattr(self, k) for k in ("size", "coreset_ratio", "max_bank", "pca_dim", "k", "calib_frac",
                                                "sigma", "seed", "bank", "bank_sq", "pca_mean", "pca_W", "cal")}
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=4)

    @classmethod
    def load(cls, path: str, extractor: FeatureExtractor) -> "PatchCore":
        with open(path, "rb") as f:
            st = pickle.load(f)
        m = cls(extractor, st["size"], st["coreset_ratio"], st["max_bank"], st["pca_dim"], st["k"],
                st["calib_frac"], st["sigma"], st["seed"])
        for k in ("bank", "bank_sq", "pca_mean", "pca_W", "cal"):
            setattr(m, k, st[k])
        return m
