import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

DEFAULT_FEATURE_COLS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "is_transfer",
]
DEFAULT_LOG_COLS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
DEFAULT_COLS_TO_SCALE = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


class FraudAutoencoderService:
    def __init__(self, models_dir: Path | str = "models") -> None:
        self.models_dir = Path(models_dir)
        self.config = self._load_config()
        self.model = self._load_model()
        self.scaler = self._load_optional_scaler()

        self.threshold_tau = float(self.config["threshold_tau"])
        self.input_features = int(self.config.get("input_features", 7))
        self.feature_cols: list[str] = list(
            self.config.get("feature_cols", DEFAULT_FEATURE_COLS)
        )
        self.log_cols: list[str] = list(
            self.config.get("log_cols", DEFAULT_LOG_COLS)
        )
        self.cols_to_scale: list[str] = list(
            self.config.get("cols_to_scale", DEFAULT_COLS_TO_SCALE)
        )

    def _load_config(self) -> dict[str, Any]:
        config_path = self.models_dir / "model_config.json"
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)

    def _load_model(self) -> torch.jit.ScriptModule:
        scaler_or_model_file = self.config["model_file"]
        model_path = self.models_dir / scaler_or_model_file
        model = torch.jit.load(str(model_path), map_location="cpu")
        model.eval()
        return model

    def _load_optional_scaler(self):
        scaler_filename = self.config.get("scaler_file", "scaler.joblib")
        scaler_path = self.models_dir / scaler_filename
        if scaler_path.exists():
            return joblib.load(scaler_path)
        return None

    def preprocess(self, features: list[list[float]], apply_log1p: bool = True) -> torch.Tensor:
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.input_features:
            raise ValueError(
                f"Attendu un tableau 2D de forme (N, {self.input_features}), reçu {arr.shape}"
            )

        col_to_idx = {name: idx for idx, name in enumerate(self.feature_cols)}

        # 1. log1p sur les colonnes monétaires/soldes si demandé
        if apply_log1p:
            for col in self.log_cols:
                if col in col_to_idx:
                    idx = col_to_idx[col]
                    arr[:, idx] = np.log1p(np.clip(arr[:, idx], a_min=0.0, a_max=None))

        # 2. StandardScaler sur les 6 colonnes (step + log_cols)
        if self.scaler is not None:
            scale_indices = [col_to_idx[col] for col in self.cols_to_scale if col in col_to_idx]
            scaled_part = self.scaler.transform(arr[:, scale_indices]).astype(np.float32)
            for pos, col_idx in enumerate(scale_indices):
                arr[:, col_idx] = scaled_part[:, pos]

        return torch.from_numpy(arr)

    def predict(
        self, features: list[list[float]], apply_log1p: bool = True
    ) -> list[dict[str, Any]]:
        x_tensor = self.preprocess(features, apply_log1p=apply_log1p)

        with torch.no_grad():
            recon = self.model(x_tensor)
            errors = torch.mean((x_tensor - recon) ** 2, dim=1).numpy()

        results: list[dict[str, Any]] = []
        for err in errors:
            score = float(err)
            results.append(
                {
                    "reconstruction_error": score,
                    "threshold_tau": self.threshold_tau,
                    "is_fraud": bool(score > self.threshold_tau),
                }
            )
        return results