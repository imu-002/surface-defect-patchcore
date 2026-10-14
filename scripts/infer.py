"""Inspect images: defect yes/no, where, and (optionally) which type.

    python scripts/infer.py --model models/fabric.pkl --input photos/new_batch --out inspected/
    python scripts/infer.py --model m.pkl --typeclf clf.pkl --input one_image.jpg --out inspected/

Writes <out>/<name>_result.png (heat-map + localised region + verdict) and <out>/results.jsonl.
"""
import argparse, json, os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from sdd import data, classify as C, viz
from sdd.features import FeatureExtractor
from sdd.patchcore import PatchCore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--input", required=True, help="image file or folder")
ap.add_argument("--out", required=True)
ap.add_argument("--typeclf")
ap.add_argument("--threshold", type=float, help="override the calibrated decision threshold")
ap.add_argument("--onnx", default=os.path.join(ROOT, "weights", "wrn50_features.onnx"))
a = ap.parse_args()

pc = PatchCore.load(a.model, FeatureExtractor(a.onnx))
thr = a.threshold if a.threshold is not None else (pc.cal.image_thr if pc.cal else None)
if thr is None:
    sys.exit("model has no calibrated threshold; pass --threshold")
clf = pickle.load(open(a.typeclf, "rb")) if a.typeclf else None
paths = data._images(a.input) if os.path.isdir(a.input) else [a.input]
os.makedirs(a.out, exist_ok=True)
lo, rows = (pc.cal.mean * 0.6 if pc.cal else 0.0), []
with open(os.path.join(a.out, "results.jsonl"), "w") as jl:
    for p in paths:
        t0 = time.perf_counter()
        x = data.read_image(p, pc.size)
        r = C.analyze(pc, x[None])
        score = float(r["score"][0])
        defect = score > thr
        ptype = None
        if defect and clf is not None:
            ptype = str(clf.predict(np.concatenate([r["deep"][0], r["hand"][0]])[None])[0])
        ms = (time.perf_counter() - t0) * 1000
        heat = r["heat"][0]
        shown = viz.overlay(x, heat, lo, max(heat.max(), thr * 1.2))
        if defect:
            m = C.region_mask(heat).astype(np.uint8)
            cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            shown = (shown * 255).astype(np.uint8).copy()
            cv2.drawContours(shown, cnt, -1, (255, 255, 255), 1)
        else:
            shown = (shown * 255).astype(np.uint8)
        verdict = ("DEFECT" + (f": {ptype}" if ptype else "")) if defect else "OK"
        shown = cv2.resize(np.ascontiguousarray(shown), (512, 512), interpolation=cv2.INTER_LINEAR)
        banner = (200, 40, 40) if defect else (30, 130, 70)
        cv2.rectangle(shown, (0, 0), (512, 26), banner, -1)
        cv2.putText(shown, f"{verdict}   score {score:.1f} (threshold {thr:.1f})", (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        name = os.path.splitext(os.path.basename(p))[0]
        cv2.imwrite(os.path.join(a.out, f"{name}_result.png"), shown[..., ::-1])
        rec = dict(image=p, score=round(score, 3), threshold=round(thr, 3), defect=bool(defect), type=ptype,
                   ms=round(ms, 1))
        jl.write(json.dumps(rec) + "\n")
        print(f"{os.path.basename(p):28s} {verdict:22s} score {score:6.1f}  {ms:5.0f} ms")
