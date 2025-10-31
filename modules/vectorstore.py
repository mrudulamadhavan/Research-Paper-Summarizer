# modules
import faiss
import numpy as np
from typing import List, Dict, Any

class FaissStore:
    """
    Minimal FAISS wrapper that stores metadata in parallel list.
    Note: not persistent in this simple prototype.
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # use inner product with normalized vectors => cosine
        self.metadatas: List[Dict[str, Any]] = []

    def add(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]):
        """
        vectors: (n, dim) normalized
        metadatas: list of dicts (length n)
        """
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError("vectors shape mismatch")
        self.index.add(vectors)
        self.metadatas.extend(metadatas)

    def search(self, qvec: np.ndarray, k: int = 5):
        """
        qvec: (dim,) or (1,dim) normalized
        returns list of {"score","metadata","index"}
        """
        if qvec.ndim == 1:
            qvec = qvec.reshape(1, -1)
        D, I = self.index.search(qvec, k)
        out = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            out.append({"score": float(score), "metadata": self.metadatas[idx], "index": int(idx)})
        return out
