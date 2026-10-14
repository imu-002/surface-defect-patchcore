"""Evaluation metrics: image AUROC, pixel AUROC, AU-PRO (MVTec definition), threshold metrics."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def image_auroc(scores, labels) -> float:
    labels = np.asarray(labels).astype(int)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def pixel_auroc(maps: np.ndarray, masks: np.ndarray) -> float:
    m = np.asarray(masks, bool).ravel()
    if not m.any() or m.all():
        return float("nan")
    return float(roc_auc_score(m, np.asarray(maps, np.float32).ravel()))


def aupro(maps: np.ndarray, masks: np.ndarray, fpr_max: float = 0.3, n_bins: int = 1000) -> float:
    """Area under the per-region-overlap curve for FPR in [0, fpr_max], normalised to [0, 1].

    PRO(t) = mean over ground-truth connected components of the fraction of the component
    covered by pixels with score >= t; FPR(t) is measured on pixels outside every mask.
    """
    maps = np.asarray(maps, np.float32)
    masks = np.asarray(masks, bool)
    if not masks.any():
        return float("nan")
    edges = np.linspace(maps.min(), maps.max() + 1e-9, n_bins + 1)

    def surv(vals):  # fraction of values >= each lower edge
        h, _ = np.histogram(vals, edges)
        return np.cumsum(h[::-1])[::-1] / max(len(vals), 1)

    fpr = surv(maps[~masks])
    pro, n_comp = np.zeros(n_bins), 0
    for m, mp in zip(masks, maps):
        if not m.any():
            continue
        lab, n = ndimage.label(m, structure=np.ones((3, 3)))
        for i in range(1, n + 1):
            pro += surv(mp[lab == i])
            n_comp += 1
    pro /= n_comp
    order = np.argsort(fpr)
    x = np.linspace(0, fpr_max, 2000)
    y = np.interp(x, fpr[order], pro[order])
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz(y, x) / fpr_max)


def threshold_metrics(scores, labels, thr: float) -> dict:
    y = np.asarray(labels).astype(int)
    p = (np.asarray(scores) > thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    return dict(accuracy=float((p == y).mean()), precision=float(pr), recall=float(rc), f1=float(f1),
                false_positive_rate=float(p[y == 0].mean()) if (y == 0).any() else float("nan"))
