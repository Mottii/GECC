from typing import Any
from pydantic import BaseModel, Field


class ExpressionInput(BaseModel):
    gene_values: list[float] | dict[str, float] = Field(
        ..., description="Expression values in feature_names order, or raw 20,531 vector, or gene name dictionary."
    )
    sample_id: str = "sample_001"


class TopFeature(BaseModel):
    gene: str
    expression: float
    attribution: float | None = None


class PredictionResponse(BaseModel):
    sample_id: str
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    top_features: list[TopFeature]
    suppressed_features: list[TopFeature] = Field(default_factory=list)
    low_confidence: bool
    warning: str | None = None
    model_version: str | None = None
    pca_coords: dict[str, float] = Field(default_factory=dict)
    ood_score: float | None = None
    is_ood: bool = False


class ProviderConfig(BaseModel):
    provider_name: str = Field("auto", description="Provider identifier (gemini, openai, anthropic, openai_compatible, offline, auto).")
    api_key: str | None = Field(None, description="Provider API key (optional if set on server).")
    model_name: str | None = Field(None, description="Model identifier override.")
    base_url: str | None = Field(None, description="Custom base URL for OpenAI-compatible providers.")


class AttachedFile(BaseModel):
    filename: str = Field(..., description="Original filename.")
    content: str = Field(..., description="File text content or base64 encoded data.")
    file_type: str | None = Field(None, description="MIME type or file extension.")


class FileUploadResponse(BaseModel):
    filename: str
    file_type: str
    file_size_bytes: int
    summary_text: str
    detected_entities: dict[str, Any] = Field(default_factory=dict)
    tabular_stats: dict[str, Any] | None = None
    matched_genes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's query.")
    history: list[dict[str, Any]] = Field(default_factory=list, description="Conversation history.")
    prediction: dict[str, Any] | None = Field(None, description="Current prediction context.")
    attached_files: list[AttachedFile] | None = Field(None, description="Attached files for clinical interpretation.")
    provider_config: ProviderConfig | None = Field(None, description="AI provider configuration.")


class ChatResponse(BaseModel):
    reply: str
    provider_used: str = Field("Offline Clinical Bio-Agent", description="The AI provider that generated the response.")
    file_summaries: list[dict[str, Any]] | None = Field(None, description="Summaries of attached files.")
    structured_insights: dict[str, Any] | None = Field(None, description="Structured clinical genomic insights.")
