"""Inference: load pre-trained models and predict epoch labels."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def list_available_models() -> list[str]:
    """Return the friendly names of all .pkl files in ``models/``."""
    if not MODELS_DIR.exists():
        return []
    return sorted(p.stem for p in MODELS_DIR.glob("*.pkl"))


def load_model(name: str):
    """Load a joblib-pickled model by its stem name (e.g. 'random_forest')."""
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Model '{name}' not found at {path}. "
            "Please run `python train_models.py` to generate the models first."
        )
    return joblib.load(path)


def load_metadata() -> dict[str, Any]:
    """Load model_metadata.json (training summary, evaluation results)."""
    path = MODELS_DIR / "model_metadata.json"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def predict_epochs(model, features: np.ndarray) -> np.ndarray:
    """Predict class labels for a feature matrix of shape (n_epochs, n_features)."""
    return model.predict(features).astype(int)
