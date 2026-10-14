"""Defect-type identification and classical-ML comparisons.

Everything here sits ON TOP of the unsupervised PatchCore detector:

  * `analyze`         one forward pass -> anomaly score/heat-map + per-image descriptors
  * hand-crafted region descriptors (shape + colour/contrast of the localised defect)
  * `cv_classify`     defect-TYPE classifiers (kNN / SVM / Random Forest), cross-validated
  * `GlobalBaselines` unsupervised image-level baselines (global kNN, one-class SVM, isolation forest)
  * `supervised_detector_auroc`  supervised good-vs-defect SVM / RF, to show the unseen-defect gap
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import cv2
import numpy as np
from skimage import measure
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM

from .metrics import image_auroc
from .patchcore import PatchCore

HAND_NAMES = ["area_frac", "n_components", "major_axis", "minor_axis", "aspect", "eccentricity", "solidity",
              "extent", "compactness", "peak_score", "mean_score_in_region", "dL", "da", "db", "dE",
              "texture_ratio", "mean_L_in", "chroma_in"]


# ------------------------------------------------------------ descriptors --
def region_mask(score_map: np.ndarray, rel: float = 0.5) -> np.ndarray:
    """Defect region = pixels above the midpoint between the background level (median) and the peak.
    Relative to the background, not to zero: normal texture already has a non-zero score."""
    bg = float(np.median(score_map))
    peak = float(score_map.max())
    m = score_map >= bg + rel * (peak - bg)
    if m.all() or peak <= bg:  # flat map (nothing stands out): fall back to the top 1 % of pixels
        m = score_map >= np.quantile(score_map, 0.99)
    return m


def hand_features(img: np.ndarray, heat: np.ndarray, rel: float = 0.5) -> np.ndarray:
    """Shape + colour descriptors of the localised defect.  img: RGB float [0,1], heat: (S,S)."""
    S = heat.shape[0]
    m = region_mask(heat, rel)
    lab = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2LAB)
    lab_in = lab[m]
    wide = cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    near = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ring = wide & ~near
    ring_px = lab[ring] if ring.sum() > 20 else lab[~m]
    d = lab_in.mean(0) - ring_px.mean(0)
    L = lab_in[:, 0]
    tex = (L.std() + 1e-3) / (ring_px[:, 0].std() + 1e-3)
    comps = measure.regionprops(measure.label(m, connectivity=2))
    comps = [c for c in comps if c.area >= 10] or comps
    big = max(comps, key=lambda c: c.area)
    major, minor = big.axis_major_length, max(big.axis_minor_length, 1.0)
    compact = big.perimeter ** 2 / (4 * np.pi * big.area) if big.area > 0 and big.perimeter > 0 else 1.0
    chroma = float(np.hypot(lab_in[:, 1], lab_in[:, 2]).mean())
    return np.array([m.mean(), len(comps), major / S, minor / S, major / minor, big.eccentricity, big.solidity,
                     big.extent, compact, heat.max(), heat[m].mean(), d[0], d[1], d[2], np.linalg.norm(d),
                     tex, L.mean(), chroma], np.float32)


def analyze(pc: PatchCore, images: np.ndarray, rel: float = 0.5, chunk: int = 8) -> Dict[str, np.ndarray]:
    """One forward pass per chunk.  Returns score, heat, deep (region-mean embedding),
    gmean/gmax (global embeddings) and hand (hand-crafted descriptors)."""
    out: Dict[str, List] = {k: [] for k in ("score", "heat", "patch", "deep", "gmean", "gmax", "hand")}
    for i in range(0, len(images), chunk):
        batch = images[i:i + chunk]
        emb = pc.embed(batch)                      # (B,h,w,D)
        P = pc.patch_scores(emb)                   # (B,h,w)
        heat = pc.to_heatmaps(P)
        for b in range(len(batch)):
            region = region_mask(P[b], rel).reshape(-1)
            flat = emb[b].reshape(-1, emb.shape[-1])
            out["deep"].append(flat[region].mean(0))
            out["gmean"].append(flat.mean(0))
            out["gmax"].append(flat.max(0))
            out["hand"].append(hand_features(batch[b], heat[b], rel))
        out["score"].append(P.reshape(len(P), -1).max(1))
        out["heat"].append(heat)
        out["patch"].append(P)
    return {k: np.concatenate(v) if k in ("score", "heat", "patch") else np.stack(v) for k, v in out.items()}


def global_descriptor(a: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([a["gmean"], a["gmax"]], axis=1)


# ------------------------------------------------- defect-type classification --
def _clf(name: str):
    return {"kNN": KNeighborsClassifier(n_neighbors=3, weights="distance"),
            "SVM": SVC(kernel="rbf", C=10.0, gamma="scale"),
            "RandomForest": RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=1)}[name]


def _pipeline(name: str, feature_set: str, n_deep: int, n_hand: int, pca_dim: int):
    deep = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=pca_dim, random_state=0))])
    if feature_set == "deep":
        pre = ColumnTransformer([("deep", deep, slice(0, n_deep))])
    elif feature_set == "hand":
        pre = ColumnTransformer([("hand", StandardScaler(), slice(n_deep, n_deep + n_hand))])
    else:
        pre = ColumnTransformer([("deep", deep, slice(0, n_deep)),
                                 ("hand", StandardScaler(), slice(n_deep, n_deep + n_hand))])
    return Pipeline([("pre", pre), ("clf", _clf(name))])


def cv_classify(deep: np.ndarray, hand: np.ndarray, y: Sequence[str], model: str, feature_set: str,
                n_splits: int = 5, seed: int = 0) -> Dict:
    """Stratified k-fold cross-validated defect-type classification (defective images only)."""
    y = np.asarray(y)
    labels = sorted(set(y))
    k = min(n_splits, min(int((y == c).sum()) for c in labels))
    if k < 2:
        return dict(accuracy=float("nan"), macro_f1=float("nan"), labels=labels, confusion=None)
    X = np.concatenate([deep, hand], axis=1)
    pca_dim = int(min(20, (len(y) * (k - 1) // k) - 1))
    pipe = _pipeline(model, feature_set, deep.shape[1], hand.shape[1], pca_dim)
    pred = cross_val_predict(pipe, X, y, cv=StratifiedKFold(k, shuffle=True, random_state=seed))
    return dict(accuracy=float(accuracy_score(y, pred)), macro_f1=float(f1_score(y, pred, average="macro")),
                labels=labels, confusion=confusion_matrix(y, pred, labels=labels).tolist(), folds=k)


def fit_type_classifier(deep: np.ndarray, hand: np.ndarray, y: Sequence[str], model: str = "SVM",
                        feature_set: str = "both"):
    """Final classifier trained on all labelled defects (used by infer.py)."""
    X = np.concatenate([deep, hand], axis=1)
    pca_dim = int(min(20, len(y) - 1))
    return _pipeline(model, feature_set, deep.shape[1], hand.shape[1], pca_dim).fit(X, np.asarray(y))


# --------------------------------------------------------- image-level baselines --
class GlobalBaselines:
    """Unsupervised image-level detectors on global descriptors; trained on defect-free images only."""

    def fit(self, G: np.ndarray) -> "GlobalBaselines":
        self.sc = StandardScaler().fit(G)
        Z = self.sc.transform(G)
        self.pca = PCA(n_components=min(50, len(G) - 1), random_state=0).fit(Z)
        Z = self.pca.transform(Z)
        self.nn = NearestNeighbors(n_neighbors=1).fit(Z)
        self.oc = OneClassSVM(nu=0.05, gamma="scale").fit(Z)
        self.iso = IsolationForest(n_estimators=300, random_state=0).fit(Z)
        return self

    def scores(self, G: np.ndarray) -> Dict[str, np.ndarray]:
        Z = self.pca.transform(self.sc.transform(G))
        return {"Global-feature kNN": self.nn.kneighbors(Z)[0][:, 0],
                "One-class SVM": -self.oc.decision_function(Z),
                "Isolation Forest": -self.iso.score_samples(Z)}


def supervised_detector_auroc(G_good_train: np.ndarray, G_def_train: np.ndarray, G_test: np.ndarray,
                              y_test: np.ndarray, model: str) -> float:
    """Supervised good-vs-defect classifier on global descriptors -> image AUROC on (G_test, y_test)."""
    X = np.concatenate([G_good_train, G_def_train])
    y = np.r_[np.zeros(len(G_good_train)), np.ones(len(G_def_train))].astype(int)
    steps = [("sc", StandardScaler()), ("pca", PCA(n_components=min(50, len(X) - 1), random_state=0))]
    if model == "SVM":
        pipe = Pipeline(steps + [("clf", SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced"))]).fit(X, y)
        s = pipe.decision_function(G_test)
    else:
        pipe = Pipeline(steps + [("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                               random_state=0, n_jobs=1))]).fit(X, y)
        s = pipe.predict_proba(G_test)[:, 1]
    return image_auroc(s, y_test)


def global_descriptors(pc: PatchCore, images: np.ndarray, chunk: int = 8) -> np.ndarray:
    """mean- and max-pooled patch embeddings per image (no anomaly scoring needed)."""
    out = []
    for i in range(0, len(images), chunk):
        emb = pc.embed(images[i:i + chunk])
        flat = emb.reshape(len(emb), -1, emb.shape[-1])
        out.append(np.concatenate([flat.mean(1), flat.max(1)], axis=1))
    return np.concatenate(out)
