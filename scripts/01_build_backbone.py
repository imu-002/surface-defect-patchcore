"""Download ImageNet-pretrained WideResNet50-2 weights, convert them to ONNX (no PyTorch needed) and verify.

    python scripts/01_build_backbone.py

Produces weights/wrn50_features.onnx (PatchCore embedding) and weights/wrn50_logits.onnx (check only).
The check classifies two photos bundled with scikit-image; if the converted network is wrong this fails.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import onnxruntime as ort
from sdd import backbone as B

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "wrn50_racm.pth"))
a = ap.parse_args()
os.makedirs(os.path.dirname(a.weights), exist_ok=True)
if not os.path.exists(a.weights):
    print("downloading", B.WEIGHTS_URL)
    B.download_weights(a.weights)
out_f = B.export(a.weights, os.path.join(ROOT, "weights", "wrn50_features.onnx"), "features")
out_l = B.export(a.weights, os.path.join(ROOT, "weights", "wrn50_logits.onnx"), "logits")
print("wrote", out_f, "and", out_l)

import skimage.data as sd
sess = ort.InferenceSession(out_l, providers=["CPUExecutionProvider"])
def top1(im):
    h, w = im.shape[:2]; s = 256 / min(h, w)
    im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    h, w = im.shape[:2]; t, l = (h - 224) // 2, (w - 224) // 2
    x = (im[t:t + 224, l:l + 224].astype(np.float32) / 255).transpose(2, 0, 1)[None]
    lg = sess.run(None, {"image": x})[0][0]
    return int(lg.argmax())
cat, cup = top1(sd.chelsea()), top1(sd.coffee())
print(f"chelsea (a cat) -> ImageNet class {cat}  (expected 281-285);  coffee -> class {cup}  (expected 967 espresso / 968 cup / 504 mug)")
assert 281 <= cat <= 285 and cup in (967, 968, 504), "converted weights failed the sanity check"
print("sanity check passed")
