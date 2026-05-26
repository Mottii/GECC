from pydantic import BaseModel, Field


class ExpressionInput(BaseModel):
    gene_values: list[float] = Field(..., description="Expression values in feature_names order.")
    sample_id: str = "sample_001"


class TopFeature(BaseModel):
    gene: str
    expression: float


class PredictionResponse(BaseModel):
    sample_id: str
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    top_features: list[TopFeature]
    low_confidence: bool
    warning: str | None = None
    model_version: str | None = None
