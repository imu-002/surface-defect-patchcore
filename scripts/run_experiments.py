"""Full evaluation on a MVTec-AD-layout dataset (MVTec AD itself, local photos, or the stand-in set).

    python scripts/run_experiments.py --data /path/to/mvtec_anomaly_detection --out results/mvtec
    python scripts/run_experiments.py --data data/standin --out results/standin      # offline pipeline check

Per category it writes <out>/<category>.json, a qualitative figure and the fitted model.
"""
import argparse, json, os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sdd import data, classify as C, viz
from sdd.features import FeatureExtractor
from sdd.metrics import aupro, image_auroc, pixel_auroc, threshold_metrics
from sdd.patchcore import PatchCore


def best_f1(scores, y):
    """Oracle (threshold tuned on the test set -> optimistic; reported separately)."""
    best = 0.0
    for t in np.unique(scores):
        best = max(best, threshold_metrics(scores, y, t - 1e-9)["f1"])
    return best


def latency(pc, paths, size, n=25):
    """Per-image latency at batch size 1 (ms): read+resize / backbone / NN search / heat-map."""
    paths = (paths * (n // len(paths) + 1))[: n + 3]
    t = {k: [] for k in ("preprocess", "backbone", "search", "heatmap")}
    for i, p in enumerate(paths):
        a = time.perf_counter(); x = data.read_batch([p], size)
        b = time.perf_counter(); e = pc.embed(x)
        c = time.perf_counter(); P = pc.patch_scores(e)
        d = time.perf_counter(); pc.to_heatmaps(P)
        f = time.perf_counter()
        if i >= 3:  # 3 warm-up runs
            for k, v in zip(t, (b - a, c - b, d - c, f - d)):
                t[k].append(v * 1000)
    out = {k: float(np.mean(v)) for k, v in t.items()}
    out["total_ms"] = sum(out.values())
    out["fps"] = 1000.0 / out["total_ms"]
    return out


def run_category(args, ex, name):
    t_cat = time.time()
    cat = data.load_category(args.data, name)
    S = args.size
    print(f"[{name}] train {len(cat.train)}  test {len(cat.test)}  types {cat.defect_types}", flush=True)
    train = data.read_batch(cat.train, S)
    pc = PatchCore(ex, S, args.coreset_ratio, args.max_bank, args.pca_dim, args.k, seed=args.seed)
    pc.fit(train, verbose=True)
    print(f"[{name}] fitted in {pc.fit_seconds:.0f}s, bank {pc.bank.shape}", flush=True)

    test = data.read_batch([s.path for s in cat.test], S)
    types = np.array([s.label for s in cat.test])
    y = (types != "good").astype(int)
    have_masks = all(s.mask_path for s in cat.test if s.label != "good")
    masks = np.stack([data.read_mask(s.mask_path, S) for s in cat.test])
    A = C.analyze(pc, test)
    res = dict(category=name, n_train=len(cat.train), n_test_good=int((y == 0).sum()), n_test_defect=int(y.sum()),
               defect_types=cat.defect_types, bank_size=int(len(pc.bank)), fit_seconds=pc.fit_seconds,
               image_thr=pc.cal.image_thr if pc.cal else None)

    # ---------------- detection + localisation (PatchCore) ----------------
    res["patchcore"] = dict(
        image_auroc=image_auroc(A["score"], y),
        pixel_auroc=pixel_auroc(A["heat"], masks) if have_masks else None,
        aupro=aupro(A["heat"], masks) if have_masks else None,
        best_f1_oracle=best_f1(A["score"], y))
    if pc.cal:
        res["patchcore"]["at_calibrated_threshold"] = threshold_metrics(A["score"], y, pc.cal.image_thr)
    res["per_type"] = {}
    for d in cat.defect_types:
        sel = (types == "good") | (types == d)
        res["per_type"][d] = dict(image_auroc=image_auroc(A["score"][sel], y[sel]),
                                  detected=float((A["score"][types == d] > pc.cal.image_thr).mean()) if pc.cal else None)

    # ---------------- unsupervised image-level baselines ----------------
    G_train = C.global_descriptors(pc, train)
    G_test = np.concatenate([A["gmean"], A["gmax"]], axis=1)
    base = C.GlobalBaselines().fit(G_train)
    res["baselines_image_auroc"] = {k: image_auroc(v, y) for k, v in base.scores(G_test).items()}
    res["baselines_image_auroc"]["PatchCore (ours)"] = res["patchcore"]["image_auroc"]

    # ---------------- defect TYPE identification (defective images only) ----------------
    dsel = y == 1
    res["type_classification"] = {}
    if len(cat.defect_types) >= 2:
        for model in ("kNN", "SVM", "RandomForest"):
            for fs in ("hand", "deep", "both"):
                r = C.cv_classify(A["deep"][dsel], A["hand"][dsel], types[dsel], model, fs, seed=args.seed)
                res["type_classification"][f"{model}/{fs}"] = r

    # ---------------- supervised detectors vs unseen defect types ----------------
    rng = np.random.default_rng(args.seed)
    good_idx = rng.permutation(np.where(y == 0)[0])
    g_tr, g_te = good_idx[: len(good_idx) // 2], good_idx[len(good_idx) // 2:]
    sup = {"seen": {}, "unseen": {}}
    pc_auc = {"seen": [], "unseen": []}
    # seen: half of every defect type is available for training, the other half is tested
    d_tr, d_te = [], []
    for d in cat.defect_types:
        ids = rng.permutation(np.where(types == d)[0])
        d_tr += list(ids[: len(ids) // 2]); d_te += list(ids[len(ids) // 2:])
    te = np.r_[d_te, g_te]
    for m in ("SVM", "RandomForest"):
        sup["seen"][m] = C.supervised_detector_auroc(np.concatenate([G_train, G_test[g_tr]]), G_test[d_tr], G_test[te], y[te], m)
    pc_auc["seen"] = image_auroc(A["score"][te], y[te])
    # unseen: leave one defect type out entirely
    for m in ("SVM", "RandomForest"):
        vals = []
        for d in cat.defect_types:
            tr_def = np.where((types != "good") & (types != d))[0]
            te = np.r_[np.where(types == d)[0], g_te]
            vals.append(C.supervised_detector_auroc(np.concatenate([G_train, G_test[g_tr]]), G_test[tr_def], G_test[te], y[te], m))
        sup["unseen"][m] = float(np.mean(vals))
    pc_auc["unseen"] = float(np.mean([image_auroc(A["score"][np.r_[np.where(types == d)[0], g_te]],
                                                  y[np.r_[np.where(types == d)[0], g_te]]) for d in cat.defect_types]))
    res["supervised_vs_unseen"] = dict(supervised=sup, patchcore=pc_auc)

    # ---------------- speed ----------------
    res["latency_ms"] = latency(pc, [s.path for s in cat.test], S)

    # ---------------- artefacts: model, type classifier, figure ----------------
    os.makedirs(os.path.join(args.out, "models"), exist_ok=True)
    pc.save(os.path.join(args.out, "models", f"{name}.pkl"))
    typeclf = None
    if len(cat.defect_types) >= 2 and dsel.sum() > 5:
        typeclf = C.fit_type_classifier(A["deep"][dsel], A["hand"][dsel], types[dsel], "SVM", "both")
        with open(os.path.join(args.out, "models", f"{name}_typeclf.pkl"), "wb") as f:
            pickle.dump(typeclf, f)
    picks = []
    for lab in ["good"] + cat.defect_types:
        i = int(np.where(types == lab)[0][0])
        pred = None
        if typeclf is not None and lab != "good":
            pred = str(typeclf.predict(np.concatenate([A["deep"][i], A["hand"][i]])[None])[0])
        picks.append(dict(img=test[i], heat=A["heat"][i], mask=masks[i], label=lab, score=float(A["score"][i]), pred=pred))
    lo = float(pc.cal.mean * 0.6) if pc.cal else float(A["heat"].min())
    viz.qualitative_figure(f"{name}: PatchCore detection and localisation", picks, lo, float(np.percentile(A["heat"], 99.9)),
                           os.path.join(args.out, f"qualitative_{name}.png"))
    res["seconds"] = time.time() - t_cat
    with open(os.path.join(args.out, f"{name}.json"), "w") as f:
        json.dump(res, f, indent=1)
    p = res["patchcore"]
    print(f"[{name}] imgAUROC {p['image_auroc']:.3f}  pixAUROC {p['pixel_auroc']}  AUPRO {p['aupro']}  "
          f"{res['latency_ms']['fps']:.1f} FPS  ({res['seconds']:.0f}s)", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--categories", nargs="*")
    ap.add_argument("--onnx", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "wrn50_features.onnx"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--coreset-ratio", type=float, default=0.1)
    ap.add_argument("--max-bank", type=int, default=40000)
    ap.add_argument("--pca-dim", type=int, default=None)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    args.data = data.resolve_root(args.data)
    print("dataset root:", args.data, "| categories:", data.list_categories(args.data), flush=True)
    ex = FeatureExtractor(args.onnx)
    missing = [c for c in (args.categories or []) if not os.path.isdir(os.path.join(args.data, c, "test"))]
    if missing:
        sys.exit(f"categories not found in {args.data}: {missing}. Available: {data.list_categories(args.data)}")
    for name in args.categories or data.list_categories(args.data):
        if os.path.exists(os.path.join(args.out, f"{name}.json")):
            print(f"[{name}] already done, skipping", flush=True)
            continue
        run_category(args, ex, name)
