"""A tiny learned scorer for the MIRA retrieval score.

One neural unit: 9 inputs (the retrieval component scores), sigmoid output,
trained with the delta rule (Widrow-Hoff 1960) — the simplest supervised
learning rule, locally computable like a synapse, no backprop-through-layers.
It learns the component weights from corpus QA data instead of hand-tuning.

Honesty (spec §49): this is a linear scoring model with a nonlinearity —
it is interpretable (9 learned weights), inspectable, and its contribution
is ablatable ("learned" vs hand weights). It is NOT a deep network and makes
no claim of biological fidelity beyond the delta-rule analogy.

Off by default. Enable with `retrieval_score.learned: true` in config.yaml
after running scripts/train_neural.py.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger("mira.neural")

FEATURES = ("semantic", "structural", "radial", "graph",
            "importance", "confidence", "recency", "path", "activation")


class DeltaRuleScorer:
    def __init__(self, lr: float = 0.05, epochs: int = 60, seed: int = 42):
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.w = np.zeros(len(FEATURES), dtype=np.float64)
        self.b = 0.0
        self.trained = False

    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """X: (n, 9) component scores in [0,1]; y: 1=relevant, 0=not."""
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.w = rng.normal(0, 0.01, d)
        self.b = 0.0
        hist: List[float] = []
        for _ in range(self.epochs):
            order = rng.permutation(n)
            loss = 0.0
            for i in order:
                p = float(self._sigmoid(self.w @ X[i] + self.b))
                err = p - float(y[i])                    # delta rule
                self.w -= self.lr * err * X[i]
                self.b -= self.lr * err
                loss += -(y[i] * np.log(p + 1e-9) + (1 - y[i]) * np.log(1 - p + 1e-9))
            hist.append(loss / n)
        self.trained = True
        return {"final_loss": round(hist[-1], 4), "first_loss": round(hist[0], 4)}

    # ------------------------------------------------------------------
    def weights(self) -> Dict[str, float]:
        """Learned weights, floored at 0 and normalized to sum 1 so they
        drop into the existing weighted-sum scorer unchanged."""
        w = np.clip(self.w, 0.0, None)
        s = w.sum()
        if s <= 1e-9:  # degenerate fit — equal weights fallback
            return {f: 1.0 / len(FEATURES) for f in FEATURES}
        v = w / s
        return {f: float(x) for f, x in zip(FEATURES, v)}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"features": list(FEATURES), "w": self.w.tolist(),
                       "b": float(self.b), "lr": self.lr, "epochs": self.epochs}, fh)

    @classmethod
    def load(cls, path: str) -> Optional["DeltaRuleScorer"]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            if list(d.get("features", [])) != list(FEATURES):
                logger.warning("neural scorer feature mismatch — ignoring file")
                return None
            s = cls(lr=d.get("lr", 0.05), epochs=d.get("epochs", 60))
            s.w = np.asarray(d["w"], dtype=np.float64)
            s.b = float(d.get("b", 0.0))
            s.trained = True
            return s
        except FileNotFoundError:
            return None
        except Exception as exc:
            logger.warning("neural scorer load failed: %s", exc)
            return None
