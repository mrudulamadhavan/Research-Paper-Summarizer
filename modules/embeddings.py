
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional

try:
    import openai
except Exception:
    openai = None

class EmbeddingBackend:
    """
    Embedding backend supporting:
      - mode="openai": use OpenAI embeddings (requires openai package & key)
      - mode="local": use sentence-transformers local model
    """
    def __init__(self, mode: str = "local", openai_api_key: Optional[str] = None, local_model_name: str = "all-mpnet-base-v2"):
        self.mode = mode
        if mode == "openai":
            if openai is None:
                raise RuntimeError("openai package not installed.")
            if not openai_api_key:
                raise ValueError("openai_api_key required for openai mode.")
            openai.api_key = openai_api_key
        else:
            self.model = SentenceTransformer(local_model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Returns normalized float32 numpy array (n, dim).
        """
        if self.mode == "openai":
            # batch requests for safety (16 per batch)
            all_vecs = []
            batch = []
            for t in texts:
                batch.append(t)
                if len(batch) >= 16:
                    res = openai.Embedding.create(input=batch, model="text-embedding-3-small")
                    vecs = [d["embedding"] for d in res["data"]]
                    all_vecs.extend(vecs)
                    batch = []
            if batch:
                res = openai.Embedding.create(input=batch, model="text-embedding-3-small")
                vecs = [d["embedding"] for d in res["data"]]
                all_vecs.extend(vecs)
            arr = np.array(all_vecs, dtype="float32")
        else:
            arr = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            arr = arr.astype("float32")
        # normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms
