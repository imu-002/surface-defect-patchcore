"""Generate the offline stand-in dataset (MVTec AD folder layout). NOT MVTec data."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sdd import synth

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/standin")
ap.add_argument("--train", type=int, default=100)
ap.add_argument("--test-good", type=int, default=40)
ap.add_argument("--per-defect", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--strength", type=float, default=1.0, help="defect contrast; <1 = subtle defects")
a = ap.parse_args()
counts = synth.write_dataset(a.out, n_train=a.train, n_test_good=a.test_good, n_per_defect=a.per_defect, seed=a.seed, strength=a.strength)
json.dump(counts, open(os.path.join(a.out, "counts.json"), "w"), indent=1)
open(os.path.join(a.out, "README.txt"), "w").write(
    "SYNTHETIC stand-in dataset (pipeline check only). Not MVTec AD. Layout mirrors MVTec AD.\n")
print(json.dumps(counts, indent=1))
