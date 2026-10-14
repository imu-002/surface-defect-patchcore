"""ImageNet-pretrained WideResNet50-2 as an ONNX feature extractor, built without PyTorch.

Why: the target is cheap CPU-only inspection stations. ONNX Runtime gives fast,
dependency-light inference, and building the graph ourselves keeps the whole
project reproducible with numpy + onnxruntime only.

Steps: (1) read the legacy-format PyTorch checkpoint with a *restricted*
unpickler (only a few harmless globals are allowed, so a hostile file cannot
execute code); (2) fold BatchNorm into the convolutions; (3) emit an ONNX graph.

Two graphs can be exported:
  * "features": PatchCore embedding = concat(avgpool3x3(layer2), upsample(avgpool3x3(layer3)))
                -> (N, 1536, H/8, W/8). Input is RGB in [0, 1]; ImageNet normalisation is inside.
  * "logits":   the full ImageNet classifier, used only to sanity-check the converted weights.
"""
from __future__ import annotations

import collections
import os
import pickle
import struct
import urllib.request
from typing import Dict

import numpy as np

from .onnx_min import GraphBuilder

WEIGHTS_URL = ("https://github.com/rwightman/pytorch-image-models/releases/download/"
               "v0.1-weights/wide_resnet50_racm-8234f177.pth")
WEIGHTS_SHA_PREFIX = "8234f177"  # timm names files by the first 8 hex chars of their SHA256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BN_EPS = 1e-5
STAGES = (("layer1", 3, 1), ("layer2", 4, 2), ("layer3", 6, 2), ("layer4", 3, 2))

_DTYPES = {"FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64,
           "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16,
           "ByteStorage": np.uint8, "CharStorage": np.int8, "BoolStorage": np.bool_}


# ------------------------------------------------- restricted checkpoint reader --
class _StorageType:
    def __init__(self, name):
        self.dtype = np.dtype(_DTYPES[name])


class _Storage:
    def __init__(self, dtype, numel):
        self.dtype, self.numel, self.data = dtype, numel, None


class _LazyTensor:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)

    def materialize(self) -> np.ndarray:
        item = self.storage.dtype.itemsize
        base = self.storage.data[self.offset:]
        view = np.lib.stride_tricks.as_strided(base, self.size, tuple(s * item for s in self.stride))
        return np.array(view)  # copy out of the shared buffer


def _rebuild_tensor(storage, offset, size, stride, *_unused):
    return _LazyTensor(storage, offset, size, stride)


class _RestrictedUnpickler(pickle.Unpickler):
    def __init__(self, f):
        super().__init__(f)
        self.storages: Dict[str, _Storage] = {}

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor
        if module == "torch" and name in _DTYPES:
            return _StorageType(name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")

    def persistent_load(self, pid):
        if pid[0] != "storage":
            raise pickle.UnpicklingError(f"unknown persistent id {pid[0]!r}")
        _, stype, key, _loc, numel, _view = pid
        if key not in self.storages:
            self.storages[key] = _Storage(stype.dtype, numel)
        return self.storages[key]


def read_torch_checkpoint(path: str) -> Dict[str, np.ndarray]:
    """Read a legacy (pre-1.6, non-zip) PyTorch state dict into numpy arrays."""
    with open(path, "rb") as f:
        if f.read(2) == b"PK":
            raise ValueError("zip-format checkpoint: use the legacy timm file from WEIGHTS_URL")
        f.seek(0)
        for _ in range(3):  # magic number, protocol version, system info
            _RestrictedUnpickler(f).load()
        up = _RestrictedUnpickler(f)
        state = up.load()
        keys = _RestrictedUnpickler(f).load()  # storage keys in file order
        for key in keys:
            st = up.storages[key]
            (numel,) = struct.unpack("<q", f.read(8))
            if numel != st.numel:
                raise ValueError(f"storage {key}: size mismatch {numel} != {st.numel}")
            st.data = np.frombuffer(f.read(numel * st.dtype.itemsize), dtype=st.dtype)
        return {k: (v.materialize() if isinstance(v, _LazyTensor) else v) for k, v in state.items()}


def download_weights(dest: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    urllib.request.urlretrieve(WEIGHTS_URL, dest)
    import hashlib
    h = hashlib.sha256()
    with open(dest, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if not h.hexdigest().startswith(WEIGHTS_SHA_PREFIX):
        os.remove(dest)
        raise ValueError("downloaded weights failed the SHA-256 check")
    return dest


# ------------------------------------------------------------------- the graph --
def _conv_bn(g: GraphBuilder, sd, x, conv_key, bn_key, stride, pad, relu):
    w = sd[conv_key + ".weight"].astype(np.float64)
    gamma, beta = sd[bn_key + ".weight"], sd[bn_key + ".bias"]
    mean, var = sd[bn_key + ".running_mean"], sd[bn_key + ".running_var"]
    scale = gamma / np.sqrt(var + BN_EPS)
    w_f = (w * scale[:, None, None, None]).astype(np.float32)
    b_f = (beta - mean * scale).astype(np.float32)
    k = w.shape[-1]
    y = g.op("Conv", [x, g.const(w_f, "w"), g.const(b_f, "b")], kernel_shape=[k, k],
             strides=[stride, stride], pads=[pad] * 4)
    return g.op("Relu", [y]) if relu else y


def _bottleneck(g, sd, x, prefix, stride):
    out = _conv_bn(g, sd, x, f"{prefix}.conv1", f"{prefix}.bn1", 1, 0, True)
    out = _conv_bn(g, sd, out, f"{prefix}.conv2", f"{prefix}.bn2", stride, 1, True)
    out = _conv_bn(g, sd, out, f"{prefix}.conv3", f"{prefix}.bn3", 1, 0, False)
    if f"{prefix}.downsample.0.weight" in sd:
        x = _conv_bn(g, sd, x, f"{prefix}.downsample.0", f"{prefix}.downsample.1", stride, 0, False)
    return g.op("Relu", [g.op("Add", [out, x])])


def build_onnx(sd: Dict[str, np.ndarray], mode: str = "features") -> bytes:
    if mode not in ("features", "logits"):
        raise ValueError(mode)
    g = GraphBuilder("wide_resnet50_2_" + mode)
    mean = g.const(np.array(IMAGENET_MEAN, np.float32).reshape(1, 3, 1, 1), "mean")
    std = g.const(np.array(IMAGENET_STD, np.float32).reshape(1, 3, 1, 1), "std")
    x = g.op("Div", [g.op("Sub", ["image", mean]), std])
    x = _conv_bn(g, sd, x, "conv1", "bn1", 2, 3, True)
    x = g.op("MaxPool", [x], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
    taps = {}
    for name, n_blocks, stride in STAGES:
        if mode == "features" and name == "layer4":
            break
        for i in range(n_blocks):
            x = _bottleneck(g, sd, x, f"{name}.{i}", stride if i == 0 else 1)
        taps[name] = x
    dims_in = {"image": ["N", 3, "H", "W"]}
    if mode == "logits":
        pooled = g.op("Flatten", [g.op("GlobalAveragePool", [x])], axis=1)
        out = g.op("Gemm", [pooled, g.const(sd["fc.weight"].astype(np.float32), "fcw"),
                            g.const(sd["fc.bias"].astype(np.float32), "fcb")], transB=1)
        return g.serialize(dims_in, {g.alias(out, "logits"): ["N", 1000]})
    pool = dict(kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1], count_include_pad=1)
    a2 = g.op("AveragePool", [taps["layer2"]], **pool)
    a3 = g.op("AveragePool", [taps["layer3"]], **pool)
    up3 = g.op("Resize", [a3, "", g.const(np.array([1, 1, 2, 2], np.float32), "scales")],
               mode="linear", coordinate_transformation_mode="half_pixel")
    feat = g.op("Concat", [a2, up3], axis=1)
    return g.serialize(dims_in, {g.alias(feat, "features"): ["N", 1536, "h", "w"]})


def export(weights_path: str, out_path: str, mode: str = "features") -> str:
    sd = read_torch_checkpoint(weights_path)
    with open(out_path, "wb") as f:
        f.write(build_onnx(sd, mode))
    return out_path
