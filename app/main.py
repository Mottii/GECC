from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.inference import ArtifactNotReadyError, CancerInferenceEngine, InvalidInputError
from app.schemas import ExpressionInput, PredictionResponse

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
    if len(payload.gene_values) != len(engine.feature_names):
        raise HTTPException(
            status_code=422,
            detail=f"Expected {len(engine.feature_names)} features, got {len(payload.gene_values)}",
        )
    try:
        result = engine.predict(payload.gene_values)
    except ArtifactNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InvalidInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictionResponse(sample_id=payload.sample_id, **result)
