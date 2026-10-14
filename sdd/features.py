"""ONNX Runtime feature extractor (CPU)."""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import onnxruntime as ort


class FeatureExtractor:
    """RGB float32 images in [0,1], NHWC  ->  patch embeddings (N, h, w, 1536), h = H/8."""

    def __init__(self, onnx_path: str, threads: int | None = None, batch: int = 8):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or os.cpu_count() or 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.batch = batch

    def __call__(self, images: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(images), self.batch):
            x = np.ascontiguousarray(images[i:i + self.batch].transpose(0, 3, 1, 2), dtype=np.float32)
            f = self.session.run(["features"], {"image": x})[0]
            out.append(f.transpose(0, 2, 3, 1))
        return np.concatenate(out) if out else np.zeros((0,))

    def iter_features(self, images: np.ndarray) -> Iterable[np.ndarray]:
        for i in range(0, len(images), self.batch):
            yield self(images[i:i + self.batch])
