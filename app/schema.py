from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
  features: list[float] = Field(
      ...,
      min_length=7,
      max_length=7,
      description=(
          "Vecteur de exactement 7 features numériques alignées sur"
          " l'entraînement."
      ),
  )


class BatchPredictRequest(BaseModel):
  transactions: list[TransactionInput] = Field(
      ...,
      min_length=1,
      description="Liste d'au moins une transaction à évaluer.",
  )


class PredictionResult(BaseModel):
  reconstruction_error: float = Field(
      ...,
      description=(
          "Erreur quadratique moyenne (MSE) de reconstruction de la"
          " transaction."
      ),
  )
  threshold_tau: float = Field(
      ...,
      description="Seuil de décision tau utilisé pour classer l'anomalie.",
  )
  is_fraud: bool = Field(
      ...,
      description="True si reconstruction_error > threshold_tau, sinon False.",
  )


class BatchPredictResponse(BaseModel):
  count: int = Field(..., ge=1, description="Nombre de transactions évaluées.")
  predictions: list[PredictionResult]