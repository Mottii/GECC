import json
import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.inference import ArtifactNotReadyError, CancerInferenceEngine, InvalidInputError
from app.schemas import ExpressionInput, PredictionResponse, ChatRequest, ChatResponse
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


@app.get("/pca_reference")
def get_pca_reference():
    if not PCA_REFERENCE_PATH.exists():
        return []
    try:
        with PCA_REFERENCE_PATH.open(encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load PCA references: {exc}")


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        reply = generate_local_fallback_response(payload.message, payload.prediction)
        return ChatResponse(reply=reply)

    # Call Gemini API
    prediction_context = payload.prediction or {}
    system_instruction = (
        "You are a clinical bioinformatics AI assistant for the Gene Expression Cancer Classifier (DetectAI).\n"
        "Your goal is to explain and interpret predictions made by the deep learning model to the user.\n"
        "Context:\n"
        f"Model Prediction: {prediction_context.get('predicted_class', 'N/A')}\n"
        f"Confidence: {prediction_context.get('confidence', 0.0)}\n"
        f"Class Probabilities: {prediction_context.get('class_probabilities', {})}\n"
        f"Top contributing genes: {prediction_context.get('top_features', [])}\n\n"
        "Respond directly, concisely, and professionally in markdown format. Do not use generic explanations; "
        "always link the user's questions back to this patient's specific expression levels and cohorts."
    )

    contents = []
    for msg in payload.history:
        role = "user" if msg.get("role") == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": msg.get("content", "")}]
        })

    contents.append({
        "role": "user",
        "parts": [{"text": f"System Context:\n{system_instruction}\n\nUser Message: {payload.message}"}]
    })

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.2
        }
    }

    try:
        response = requests.post(url, headers=headers, json=body, timeout=10)
        response.raise_for_status()
        res_data = response.json()
        reply = res_data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:
        reply = f"Error communicating with Gemini API: {exc}. Falling back to offline assistant.\n\n" + generate_local_fallback_response(payload.message, payload.prediction)

    return ChatResponse(reply=reply)


def generate_local_fallback_response(message: str, prediction: dict | None) -> str:
    if not prediction:
        return (
            "Hello! I am your AI clinical assistant. Please load a sample and predict its cancer type "
            "so I can analyze the expression data and provide specific insights."
        )
    
    predicted_class = prediction.get("predicted_class", "Unknown")
    confidence = prediction.get("confidence", 0.0)
    top_features = prediction.get("top_features", [])
    
    msg_lower = message.lower()
    
    if "gene" in msg_lower or "feature" in msg_lower or "expression" in msg_lower:
        genes_str = ", ".join([f"{f['gene']} ({f['expression']})" for f in top_features])
        return (
            f"For this sample, the top contributing genes detected are: {genes_str}. "
            f"These highly-expressed genes are characteristic markers that heavily influenced the model "
            f"to predict **{predicted_class}** with a confidence score of {confidence * 100:.1f}%."
        )
    
    if "confidence" in msg_lower or "certain" in msg_lower or "probabilit" in msg_lower:
        prob_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in prediction.get("class_probabilities", {}).items()])
        return (
            f"The model predicted **{predicted_class}** with {confidence * 100:.1f}% confidence. "
            f"The full cohort probability distribution is: {prob_str}. "
            + ("This is a high-confidence call." if confidence >= 0.75 else "Note: the confidence is low; proceed with caution.")
        )
        
    return (
        f"I've analyzed the prediction of **{predicted_class}** (confidence: {confidence * 100:.1f}%). "
        f"The top driving features are {', '.join([f['gene'] for f in top_features[:3]])}. "
        f"Configure your `GEMINI_API_KEY` in the environment to enable full, dynamic conversational interpretations."
    )
