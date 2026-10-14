"""Speed/accuracy trade-off: embedding dimension (PCA) x memory-bank size (coreset ratio).

Backbone features are extracted once and re-used, so each variant only pays for coreset + search.

    python scripts/ablation.py --data data/standin_subtle --categories gravel fabric --out results/ablation_subtle.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sdd import data
from sdd.features import FeatureExtractor
from sdd.metrics import aupro, image_auroc, pixel_auroc
from sdd.patchcore import PatchCore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--categories", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--pca", nargs="*", type=int, default=[0, 128])
ap.add_argument("--ratios", nargs="*", type=float, default=[0.01, 0.05, 0.1])
ap.add_argument("--onnx", default=os.path.join(ROOT, "weights", "wrn50_features.onnx"))
a = ap.parse_args()
S = 256
ex = FeatureExtractor(a.onnx)
rows = []
for name in a.categories:
    cat = data.load_category(a.data, name)
    train = data.read_batch(cat.train, S)
    test = data.read_batch([s.path for s in cat.test], S)
    y = np.array([s.label != "good" for s in cat.test]).astype(int)
    masks = np.stack([data.read_mask(s.mask_path, S) for s in cat.test])
    base = PatchCore(ex, S)
    fit_i, cal_i = base.split_train(len(train))
    feats = np.concatenate([f.reshape(-1, f.shape[-1]) for f in ex.iter_features(train[fit_i])])
    cal = ex(train[cal_i])
    Ftest = ex(test)
    t = []
    for i in range(13):  # backbone latency at batch 1 (first 3 = warm-up)
        t0 = time.perf_counter(); ex(test[i:i + 1]); t.append((time.perf_counter() - t0) * 1000)
    backbone_ms = float(np.mean(t[3:]))
    for pca in a.pca:
        for ratio in a.ratios:
            pc = PatchCore(ex, S, coreset_ratio=ratio, max_bank=10**9, pca_dim=pca or None)
            pc.fit_raw(feats, cal)
            emb = pc._project(Ftest) if pca else Ftest
            P = np.concatenate([pc.patch_scores(emb[i:i + 8]) for i in range(0, len(emb), 8)])
            heat = pc.to_heatmaps(P)
            sc = P.reshape(len(P), -1).max(1)
            t = []
            for i in range(13):
                t0 = time.perf_counter(); pc.patch_scores(emb[i:i + 1]); t.append((time.perf_counter() - t0) * 1000)
            search_ms = float(np.mean(t[3:]))
            r = dict(category=name, pca_dim=pca or 1536, coreset_ratio=ratio, bank=int(len(pc.bank)),
                     image_auroc=image_auroc(sc, y), pixel_auroc=pixel_auroc(heat, masks), aupro=aupro(heat, masks),
                     search_ms=search_ms, backbone_ms=backbone_ms, est_fps=1000 / (backbone_ms + search_ms + 3.5),
                     fit_seconds_coreset=pc.fit_seconds)
            rows.append(r)
            print(f"{name:8s} dim {r['pca_dim']:4d} ratio {ratio:<5} bank {r['bank']:6d}  imgAUROC {r['image_auroc']:.3f} "
                  f"pixAUROC {r['pixel_auroc']:.3f} AUPRO {r['aupro']:.3f}  search {search_ms:5.1f} ms  ~{r['est_fps']:.1f} FPS", flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)
