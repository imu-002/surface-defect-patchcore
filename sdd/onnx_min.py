"""Minimal ONNX writer (no `onnx` or `torch` package needed).

Only what is required to export a CNN feature extractor: Conv, Relu, Add,
MaxPool, AveragePool, Resize, Concat, Sub, Div, GlobalAveragePool, Flatten, Gemm.
The protobuf wire format is simple enough to emit by hand; onnxruntime validates
the result on load, so a malformed graph fails loudly instead of silently.
"""
from __future__ import annotations

import struct
from typing import Dict, List, Optional, Sequence

import numpy as np

FLOAT, INT64 = 1, 7  # TensorProto.DataType


# ----------------------------------------------------------------- protobuf --
def _varint(n: int) -> bytes:
    if n < 0:
        n += 1 << 64
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _key(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _ld(field: int, payload: bytes) -> bytes:
    return _key(field, 2) + _varint(len(payload)) + payload


def _int(field: int, n: int) -> bytes:
    return _key(field, 0) + _varint(n)


def _str(field: int, s: str) -> bytes:
    return _ld(field, s.encode("utf-8"))


# ------------------------------------------------------------------ messages --
def tensor_proto(name: str, arr: np.ndarray) -> bytes:
    if arr.dtype == np.float32:
        dt = FLOAT
    elif arr.dtype == np.int64:
        dt = INT64
    else:
        raise TypeError(f"unsupported dtype {arr.dtype}")
    arr = np.ascontiguousarray(arr)
    out = b"".join(_int(1, int(d)) for d in arr.shape)
    out += _int(2, dt) + _str(8, name) + _ld(9, arr.tobytes())
    return out


def attribute(name: str, value) -> bytes:
    out = _str(1, name)
    if isinstance(value, bool):
        raise TypeError("use int, not bool")
    if isinstance(value, int):
        out += _int(3, value) + _int(20, 2)
    elif isinstance(value, float):
        out += _key(2, 5) + struct.pack("<f", value) + _int(20, 1)
    elif isinstance(value, str):
        out += _ld(4, value.encode()) + _int(20, 3)
    elif isinstance(value, (list, tuple)) and all(isinstance(v, int) for v in value):
        out += b"".join(_int(8, v) for v in value) + _int(20, 7)
    else:
        raise TypeError(f"unsupported attribute {name}={value!r}")
    return out


def node_proto(op: str, inputs: Sequence[str], outputs: Sequence[str], name: str, **attrs) -> bytes:
    out = b"".join(_str(1, i) for i in inputs) + b"".join(_str(2, o) for o in outputs)
    out += _str(3, name) + _str(4, op)
    for k, v in attrs.items():
        out += _ld(5, attribute(k, v))
    return out


def value_info(name: str, dims: Sequence) -> bytes:
    shape = b""
    for d in dims:
        shape += _ld(1, _int(1, d) if isinstance(d, int) else _str(2, d))
    tensor_type = _int(1, FLOAT) + _ld(2, shape)
    return _str(1, name) + _ld(2, _ld(1, tensor_type))


class GraphBuilder:
    def __init__(self, name: str = "graph"):
        self.name = name
        self.nodes: List[bytes] = []
        self.inits: List[bytes] = []
        self._n = 0

    def uid(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}_{self._n}"

    def const(self, arr: np.ndarray, prefix: str = "c") -> str:
        name = self.uid(prefix)
        self.inits.append(tensor_proto(name, arr))
        return name

    def op(self, op: str, inputs: Sequence[str], n_out: int = 1, **attrs) -> str:
        outs = [self.uid(op.lower()) for _ in range(n_out)]
        self.nodes.append(node_proto(op, inputs, outs, self.uid("n"), **attrs))
        return outs[0]

    def alias(self, src: str, out_name: str) -> str:
        """Expose a tensor under a stable public name (via Identity)."""
        self.nodes.append(node_proto("Identity", [src], [out_name], self.uid("n")))
        return out_name

    def serialize(self, inputs: Dict[str, Sequence], outputs: Dict[str, Sequence], opset: int = 13) -> bytes:
        g = b"".join(_ld(1, n) for n in self.nodes) + _str(2, self.name)
        g += b"".join(_ld(5, t) for t in self.inits)
        g += b"".join(_ld(11, value_info(k, v)) for k, v in inputs.items())
        g += b"".join(_ld(12, value_info(k, v)) for k, v in outputs.items())
        model = _int(1, 8) + _str(2, "sdd-onnx-min") + _ld(7, g) + _ld(8, _str(1, "") + _int(2, opset))
        return model
