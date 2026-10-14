"""Fit a PatchCore model from a folder of DEFECT-FREE photos only (works with a few dozen images).

    python scripts/fit_model.py --good-dir photos/fabric_good --out models/fabric.pkl
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sdd import data
from sdd.features import FeatureExtractor
from sdd.patchcore import PatchCore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--good-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--onnx", default=os.path.join(ROOT, "weights", "wrn50_features.onnx"))
ap.add_argument("--size", type=int, default=256)
ap.add_argument("--coreset-ratio", type=float, default=0.1)
ap.add_argument("--max-bank", type=int, default=40000)
ap.add_argument("--pca-dim", type=int, default=None, help="e.g. 128 for faster inference")
a = ap.parse_args()

paths = data._images(a.good_dir)
if len(paths) < 8:
    sys.exit(f"need at least 8 defect-free images in {a.good_dir}, found {len(paths)}")
pc = PatchCore(FeatureExtractor(a.onnx), a.size, a.coreset_ratio, a.max_bank, a.pca_dim)
pc.fit(data.read_batch(paths, a.size), verbose=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
pc.save(a.out)
print(f"saved {a.out}: {len(paths)} images, bank {pc.bank.shape}, decision threshold {pc.cal.image_thr:.2f} "
      f"(held-out good images: mean {pc.cal.mean:.2f}, std {pc.cal.std:.2f}, n={pc.cal.n}), {pc.fit_seconds:.0f}s")
