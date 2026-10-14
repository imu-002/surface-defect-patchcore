"""Train the defect-TYPE classifier from a few labelled defect photos.

    <defect-root>/<type_name>/*.png   (one sub-folder per defect type)

    python scripts/fit_type_classifier.py --model models/fabric.pkl --defect-root photos/fabric_defects --out models/fabric_typeclf.pkl
"""
import argparse, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sdd import data, classify as C
from sdd.features import FeatureExtractor
from sdd.patchcore import PatchCore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--defect-root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--onnx", default=os.path.join(ROOT, "weights", "wrn50_features.onnx"))
a = ap.parse_args()

pc = PatchCore.load(a.model, FeatureExtractor(a.onnx))
paths, labels = [], []
for t in sorted(os.listdir(a.defect_root)):
    for p in data._images(os.path.join(a.defect_root, t)):
        paths.append(p); labels.append(t)
labels = np.array(labels)
if len(set(labels)) < 2:
    sys.exit("need at least two defect-type sub-folders")
A = C.analyze(pc, data.read_batch(paths, pc.size))
for m in ("kNN", "SVM", "RandomForest"):
    r = C.cv_classify(A["deep"], A["hand"], labels, m, "both")
    print(f"{m:13s} cross-validated accuracy {r['accuracy']:.2f}  macro-F1 {r['macro_f1']:.2f}")
clf = C.fit_type_classifier(A["deep"], A["hand"], labels, "SVM", "both")
with open(a.out, "wb") as f:
    pickle.dump(clf, f)
print("saved", a.out, "classes:", sorted(str(x) for x in set(labels)))
