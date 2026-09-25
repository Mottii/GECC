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
        f"Top contributing genes: {prediction_context.get('top_features', [])}\n"
        f"Suppressed genes: {prediction_context.get('suppressed_features', [])}\n"
        f"Warning: {prediction_context.get('warning', 'None')}\n\n"
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
    suppressed_features = prediction.get("suppressed_features", [])
    low_confidence = prediction.get("low_confidence", False)
    warning = prediction.get("warning")
    probs = prediction.get("class_probabilities", {})

    msg_lower = message.lower()

    if "suppress" in msg_lower or "negative" in msg_lower or "tumor suppressor" in msg_lower:
        if suppressed_features:
            genes_str = ", ".join([f"{f['gene']} (attr: {f.get('attribution', 'N/A')}, expr: {f['expression']})" for f in suppressed_features])
            return (
                f"**Suppressed Biomarkers & Tumor Suppressors:**\n"
                f"For this {predicted_class} sample, key negatively attributing biomarkers include: {genes_str}.\n"
                f"In oncogenomics, down-regulation of protective tumor suppressors removes critical cell-cycle checkpoints, "
                f"significantly contributing to malignant phenotype classification."
            )
        return f"No strongly suppressed biomarkers were detected for this {predicted_class} profile."

    if "gene" in msg_lower or "feature" in msg_lower or "expression" in msg_lower or "driver" in msg_lower or "biomarker" in msg_lower:
        pos_str = ", ".join([f"{f['gene']} (expr: {f['expression']}, attr: {f.get('attribution', 'N/A')})" for f in top_features[:3]])
        neg_str = ", ".join([f"{f['gene']} (expr: {f['expression']}, attr: {f.get('attribution', 'N/A')})" for f in suppressed_features[:2]]) if suppressed_features else "None"
        return (
            f"**Genomic Attribution Analysis for {predicted_class}** (Confidence: {confidence * 100:.1f}%):\n\n"
            f"- **Top Activating Drivers (Positive Input x Gradient):** {pos_str}\n"
            f"- **Suppressed Biomarkers (Negative Input x Gradient):** {neg_str}\n\n"
            f"These biomarkers reflect characteristic oncogenic dysregulations. Attributions combine normalized expression with model gradients to identify true functional drivers."
        )

    if "confiden" in msg_lower or "certain" in msg_lower or "probabilit" in msg_lower or "warning" in msg_lower:
        sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        prob_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in sorted_probs])
        diff_str = ""
        if len(sorted_probs) >= 2:
            margin = (sorted_probs[0][1] - sorted_probs[1][1]) * 100
            diff_str = f"\n- Primary vs Secondary Margin: {margin:.1f}% ({sorted_probs[0][0]} vs {sorted_probs[1][0]})."

        status_note = (
            "⚠️ **Clinician Alert: Low Confidence / Borderline Call.** Proceed with caution and verify via immunohistochemistry (IHC) or histological biopsy."
            if low_confidence or warning
            else "✅ **High Confidence Call:** Cohort separation is statistically distinct."
        )
        return (
            f"**Model Calibration & Cohort Probabilities:**\n"
            f"- Predicted Cohort: **{predicted_class}** ({confidence * 100:.1f}% confidence){diff_str}\n"
            f"- Distribution: {prob_str}\n\n"
            f"{status_note}"
        )

    return (
        f"**Clinical Differential Summary ({predicted_class}):**\n"
        f"- Confidence: {confidence * 100:.1f}% ({'Uncertain / Borderline' if low_confidence else 'Statistically Robust'})\n"
        f"- Primary Driver: {top_features[0]['gene'] if top_features else 'N/A'}\n"
        f"- Key Suppressed Gene: {suppressed_features[0]['gene'] if suppressed_features else 'N/A'}\n"
        f"Ask about specific driving biomarkers, suppressed tumor suppressors, or cohort probability margins for deeper clinical interpretation."
    )
