from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException

from app.inference import FraudAutoencoderService
from app.schema import BatchPredictRequest, BatchPredictResponse

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(
    title="Fraud Autoencoder API",
    description=(
        "API d'inférence pour la détection de fraude financière par"
        " autoencodeur."
    ),
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
service = FraudAutoencoderService(models_dir=MODELS_DIR)


@app.get("/")
def root() -> dict:
  return {
      "project": "Détection de fraude financière par Autoencodeur",
      "author": "Jérémie MPEMBELE",
      "version": "1.0.0",
      "model_info": {
          "type": "PyTorch Autoencoder (TorchScript traced)",
          "input_features": service.input_features,
          "threshold_tau": service.threshold_tau,
          "scaler_loaded": service.scaler is not None,
      },
      "endpoints": {
          "root": "GET /",
          "health": "GET /health",
          "predict": "POST /predict",
          "docs": "GET /docs",
      },
  }


@app.get("/health")
def health() -> dict:
  return {
      "status": "ok",
      "input_features": service.input_features,
      "threshold_tau": service.threshold_tau,
      "scaler_loaded": service.scaler is not None,
  }


@app.post("/predict", response_model=BatchPredictResponse)
def predict(payload: BatchPredictRequest) -> BatchPredictResponse:
  raw_features = [item.features for item in payload.transactions]

  try:
    predictions = service.predict(raw_features, apply_log1p=True)
    return BatchPredictResponse(
        count=len(predictions),
        predictions=predictions,
    )
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc