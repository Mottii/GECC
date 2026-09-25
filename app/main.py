import json
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.inference import ArtifactNotReadyError, CancerInferenceEngine, InvalidInputError
from app.schemas import (
    ExpressionInput,
    PredictionResponse,
    ChatRequest,
    ChatResponse,
    FileUploadResponse,
)
from app.ai.providers import OfflineBioAgentProvider, registry
from app.ai.pipeline import ClinicalInterpretationAgent
from app.ai.file_parser import parse_attached_file
from src.utils.config import PCA_REFERENCE_PATH

app = FastAPI(
    title="Gene Expression Cancer Classifier API",
    description="Predict cancer type from RNA-Seq gene expression data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/frontend", StaticFiles(directory="app/frontend"), name="frontend")

engine = CancerInferenceEngine()


@app.get("/")
def root():
    return {"message": "Gene Expression Cancer Classifier API", "docs": "/docs", "frontend": "/frontend/index.html"}


@app.get("/health")
def health():
    return {
        "status": "ok" if engine.model_loaded else "setup_required",
        "model_loaded": engine.model_loaded,
        "detail": engine.readiness_message(),
        "model_metadata": engine.model_metadata(),
    }


@app.post("/reload")
def reload_artifacts():
    engine.reload()
    return health()


@app.get("/classes")
def get_classes():
    return {"classes": engine.class_names}


@app.get("/features")
def get_features():
    return {"feature_names": engine.feature_names, "count": len(engine.feature_names)}


@app.get("/metadata")
def get_metadata():
    return {"model_metadata": engine.model_metadata()}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: ExpressionInput):
    if not engine.model_loaded:
        raise HTTPException(status_code=503, detail=engine.readiness_message())
    
    val = payload.gene_values
    if isinstance(val, list):
        if len(val) != len(engine.feature_names) and len(val) != 20531:
            raise HTTPException(
                status_code=422,
                detail=f"Expected {len(engine.feature_names)} features (or 20,531 raw features), got {len(val)}",
            )
    elif isinstance(val, dict):
        if not val:
            raise HTTPException(status_code=422, detail="gene_values dictionary cannot be empty.")
    else:
        raise HTTPException(status_code=422, detail="gene_values must be a list of floats or a dictionary.")

    try:
        result = engine.predict(payload.gene_values)
    except ArtifactNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InvalidInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictionResponse(sample_id=payload.sample_id, **result)


@app.get("/pca_reference")
def get_pca_reference():
    if not PCA_REFERENCE_PATH.exists():
        return []
    try:
        with PCA_REFERENCE_PATH.open(encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load PCA references: {exc}")


@app.get("/demo_profiles")
def get_demo_profiles():
    from src.utils.config import DEMO_PROFILES_PATH
    if not DEMO_PROFILES_PATH.exists():
        return {}
    try:
        with DEMO_PROFILES_PATH.open(encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load demo profiles: {exc}")


# -------------------------------------------------------------------------
# AI Architecture & Pipeline Endpoints
# -------------------------------------------------------------------------

@app.get("/ai/providers")
def get_ai_providers():
    """List available AI model providers and default resolution status."""
    return {
        "providers": registry.list_providers(),
        "active_default": registry.resolve_provider_name(),
    }


@app.post("/chat/upload", response_model=FileUploadResponse)
async def upload_chat_file(request: Request):
    """Upload and profile an attached clinical note or expression table for chat."""
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        uploaded_file = form.get("file")
        if not uploaded_file or not hasattr(uploaded_file, "filename"):
            raise HTTPException(status_code=400, detail="No file field uploaded in form data.")
        filename = uploaded_file.filename
        raw_bytes = await uploaded_file.read()
        if len(raw_bytes) > 5 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Uploaded file exceeds 5 MB size limit.")
        content = raw_bytes.decode("utf-8", errors="replace")
    elif "application/json" in content_type:
        data = await request.json()
        filename = data.get("filename", "uploaded_file.txt")
        content = data.get("content", "")
    else:
        filename = "attached_file.txt"
        body = await request.body()
        if len(body) > 5 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Uploaded file exceeds 5 MB size limit.")
        content = body.decode("utf-8", errors="replace")

    res = parse_attached_file(
        filename=filename,
        content=content,
        feature_names=engine.feature_names if engine.model_loaded else None,
    )

    return FileUploadResponse(
        filename=res.filename,
        file_type=res.file_type,
        file_size_bytes=res.file_size_bytes,
        summary_text=res.summary_text,
        detected_entities=res.detected_entities,
        tabular_stats=res.tabular_stats,
        matched_genes=res.matched_genes,
        warnings=res.warnings,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    """Multi-provider AI interpretation of predictions and attached files."""
    agent = ClinicalInterpretationAgent(
        provider_registry=registry,
        feature_names=engine.feature_names if engine.model_loaded else None,
    )
    return agent.run(payload)


def generate_local_fallback_response(message: str, prediction: dict | None) -> str:
    """Backward-compatible fallback clinical reasoning engine."""
    agent_provider = OfflineBioAgentProvider()
    return agent_provider.interpret(message, prediction)
