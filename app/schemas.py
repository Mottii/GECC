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
    pca_coords: dict[str, float] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's query.")
    history: list[dict] = Field(default_factory=list, description="Conversation history.")
    prediction: dict | None = Field(None, description="Current prediction context.")


class ChatResponse(BaseModel):
    reply: str
