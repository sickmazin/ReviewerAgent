"""
rag_guideline_checker.py
========================
RAG system for validating reviews against platform guidelines 
(Amazon, eBay, Trustpilot, Google, etc.).

Architecture
------------
  1. RAG Retriever  → identifies the correct guidelines document 
                      based on input (platform name or review)
  2. GuidelineStore → repository of guidelines documents read from text files
  3. ComplianceChecker → first general heuristic check, then LLM evaluation
  4. InsightBridge  → (optional) integrates InsightReviewScorer for the 
                      insightfulness score on the validated review
"""

from __future__ import annotations
import re
import json
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from ml_models.rag_and_compliance.config import FALLBACK_MODELS,GUIDELINES_PATH


# ============================================================
#  GuidelineStore — loads files from the guidelines/ folder
# ============================================================

class GuidelineStore:
    """
    Loads .txt files from the `guidelines_path` folder.
    The filename (without extension) becomes the platform_id.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GuidelineStore, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, guidelines_path: str = "guidelines"):
        if getattr(self, "_initialized", False):
            return
        self._store: dict[str, dict] = {}

        path = Path(guidelines_path)
        if not path.exists():
             raise FileNotFoundError(
                f"[GuidelineStore] Directory '{path.resolve()}' not found."
            )

        # loads TXT files and converts them
        for f in sorted(path.glob("*.txt")):
            try:
                platform_id = f.stem.lower()
                text_content = f.read_text(encoding="utf-8")
                # Creates JSON structure from TXT file
                self._store[platform_id] = {
                    "display_name": platform_id.capitalize(),
                    "keywords": [platform_id],  # Minimal keyword
                    "text": text_content,
                }
                print(f"[GuidelineStore] Loaded platform '{platform_id}' from {f.name}")
            except Exception as e:
                print(f"[GuidelineStore] ⚠️  Error in {f.name}: {e}")

        print(f"[GuidelineStore] Total: {len(self._store)} platforms.")
        self._initialized = True

    def reload(self):
        self._store.clear()
        self.__init__()

    def add(self, platform_id: str, keywords: list[str], text: str):
        self._store[platform_id.lower()] = {
            "display_name": platform_id.capitalize(),
            "keywords": keywords,
            "text": text,
        }

    @property
    def platforms(self) -> list[str]:
        return list(self._store.keys())

    def get(self, platform_id: str) -> Optional[dict]:
        return self._store.get(platform_id.lower())

    def all_texts(self) -> list[tuple[str, str]]:
        return [(k, v["text"]) for k, v in self._store.items()]


# ============================================================
#  Light embedder for semantic retrieval
# ============================================================

class LightEmbedder:


    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or FALLBACK_MODELS[0]
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        for name in ([self.model_name] + FALLBACK_MODELS):
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(name)
                self._model = AutoModel.from_pretrained(name).float().eval()
                self.model_name = name
                print(f"[Embedder] Model loaded: {name}")
                return
            except Exception as e:
                print(f"[Embedder] {name} not available: {e}")
        raise RuntimeError("No embedding model available.")

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        self._load()
        if not texts:
            # Returns an empty array with the correct shape
            return np.empty((0, 384), dtype=np.float32)

        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            )
            out = self._model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_embs.append(emb.cpu().numpy())
        return np.vstack(all_embs)


# ============================================================
#  RAG Retriever
# ============================================================

class GuidelineRetriever:
    """
    Retrieves the most relevant guidelines document given a text input.

    Two-level strategy:
      1. Keyword match  — searches for 'keywords' from the JSON in the input (fast, precise)
      2. Semantic match — embedding cosine similarity (fallback)
    """

    def __init__(
            self,
            store: Optional[GuidelineStore] = None,
            embedder: Optional[LightEmbedder] = None,
    ):
        self.store    = store or GuidelineStore(guidelines_path=GUIDELINES_PATH)
        self.embedder = embedder or LightEmbedder()
        self._index_built = False
        self._platform_ids: list[str] = []
        self._embeddings: Optional[np.ndarray] = None

    def _keyword_match(self, query: str) -> Optional[str]:
        """
        Searches for 'keywords' from the JSON in the input (whole word).
        """
        query_l = query.lower()
        for pid, data in self.store._store.items():
            for kw in data.get("keywords", []):
                if re.search(r'\b' + re.escape(kw.lower()) + r'\b', query_l):
                    return pid
        return None

    def _build_index(self):
        if self._index_built:
            return
        pairs = self.store.all_texts()
        self._platform_ids = [p for p, _ in pairs]
        texts = [t for _, t in pairs]
        print("[Retriever] Building semantic index...")
        self._embeddings = self.embedder.encode(texts)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        self._embeddings = self._embeddings / np.maximum(norms, 1e-9)
        self._index_built = True

    def _semantic_match(self, query: str, top_k: int = 1) -> list[tuple[str, float]]:
        self._build_index()
        if self._embeddings is None or len(self._embeddings) == 0:
            return []
        q_emb = self.embedder.encode([query])
        q_norm = np.linalg.norm(q_emb)
        q_emb = q_emb / max(q_norm, 1e-9)
        sims = (self._embeddings @ q_emb.T).squeeze()
        if sims.ndim == 0:
            sims = sims.reshape(1)
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self._platform_ids[i], float(sims[i])) for i in top_idx]

    def retrieve(
            self,
            query: str,
            semantic_threshold: float = 0.50,
    ) -> tuple[Optional[str], Optional[dict], str]:
        """
        Retrieves the most relevant guideline.
        FIX: always returns 3 values → (platform_id, guideline_dict, method)
             so ReviewRAGSystem.check can do: pid, guideline, method = ...
        """
        platform_id = self._keyword_match(query)
        if platform_id:
            return platform_id, self.store.get(platform_id), "keyword"

        results = self._semantic_match(query)
        if results:
            platform_id, score = results[0]
            if score >= semantic_threshold:
                return platform_id, self.store.get(platform_id), f"semantic({score:.2f})"

        return None, None, "none"

