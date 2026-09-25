"""Comprehensive Unit & Integration Test Suite for DetectAI AI Agent Architecture.

Tests:
1. All AI Providers (Gemini, OpenAI, Anthropic, OpenAI-Compatible, Offline Bio-Agent).
2. Provider Registry resolution, auto-detection, and graceful fallbacks.
3. File Parser (CSV/TSV tabular, clinical text notes, genomic JSON, edge cases).
4. ClinicalInterpretationAgent 4-stage interpretation pipeline.
5. Endpoints: GET /ai/providers, POST /chat/upload, POST /chat (backward-compatibility).
"""

import io
import json
import pytest
from fastapi.testclient import TestClient

from app.ai.file_parser import FileParseResult, parse_attached_file
from app.ai.pipeline import ClinicalInterpretationAgent, GenomicCaseDossier
from app.ai.providers import (
    AnthropicProvider,
    BaseAIProvider,
    GeminiProvider,
    OfflineBioAgentProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    ProviderRegistry,
)
from app.main import app
from app.schemas import AttachedFile, ChatRequest, ProviderConfig


# =============================================================================
# 1. AI PROVIDER UNIT TESTS
# =============================================================================

class MockResponse:
    """Mock requests.Response for testing HTTP-based providers."""
    def __init__(self, json_data: dict, status_code: int = 200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP Error {self.status_code}")


def test_gemini_provider_success(monkeypatch):
    """GeminiProvider formats contents and parses response."""
    captured = {}

    def mock_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return MockResponse({
            "candidates": [
                {"content": {"parts": [{"text": "Gemini clinical response"}]}}
            ]
        })

    monkeypatch.setattr("requests.post", mock_post)

    provider = GeminiProvider(api_key="fake-gemini-key", model_name="gemini-1.5-flash")
    assert provider.is_configured() is True

    reply = provider.generate_response(
        messages=[{"role": "user", "content": "Explain BRCA"}],
        system_prompt="Clinical assistant prompt",
    )
    assert reply == "Gemini clinical response"
    assert "fake-gemini-key" in captured["url"]
    assert "gemini-1.5-flash" in captured["url"]
    assert captured["json"]["systemInstruction"]["parts"][0]["text"] == "Clinical assistant prompt"


def test_gemini_provider_missing_key():
    """GeminiProvider raises ValueError if no key is configured."""
    provider = GeminiProvider(api_key="")
    with pytest.raises(ValueError, match="Gemini API key is not configured"):
        provider.generate_response([{"role": "user", "content": "Hi"}])


def test_openai_provider_success(monkeypatch):
    """OpenAIProvider formats messages and parses chat completion."""
    captured = {}

    def mock_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return MockResponse({
            "choices": [
                {"message": {"content": "OpenAI clinical response"}}
            ]
        })

    monkeypatch.setattr("requests.post", mock_post)

    provider = OpenAIProvider(api_key="fake-openai-key", model_name="gpt-4o-mini")
    assert provider.is_configured() is True

    reply = provider.generate_response(
        messages=[{"role": "user", "content": "Explain KIRC"}],
        system_prompt="Expert oncologist prompt",
    )
    assert reply == "OpenAI clinical response"
    assert captured["headers"]["Authorization"] == "Bearer fake-openai-key"
    assert captured["json"]["model"] == "gpt-4o-mini"
    assert captured["json"]["messages"][0]["role"] == "system"


def test_anthropic_provider_success(monkeypatch):
    """AnthropicProvider formats messages with alternating roles and system parameter."""
    captured = {}

    def mock_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return MockResponse({
            "content": [
                {"text": "Claude clinical response"}
            ]
        })

    monkeypatch.setattr("requests.post", mock_post)

    provider = AnthropicProvider(api_key="fake-claude-key", model_name="claude-3-5-sonnet-20241022")
    assert provider.is_configured() is True

    reply = provider.generate_response(
        messages=[
            {"role": "user", "content": "Explain LUAD"},
            {"role": "user", "content": "Add IHC details"}
        ],
        system_prompt="Anthropic system instructions",
    )
    assert reply == "Claude clinical response"
    assert captured["headers"]["x-api-key"] == "fake-claude-key"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["json"]["system"] == "Anthropic system instructions"
    # Consecutive user messages should be merged
    assert len(captured["json"]["messages"]) == 1
    assert "Explain LUAD\n\nAdd IHC details" in captured["json"]["messages"][0]["content"]


def test_openai_compatible_provider_custom_base_url(monkeypatch):
    """OpenAICompatibleProvider connects to custom endpoints like Ollama."""
    captured = {}

    def mock_post(url, headers, json, timeout):
        captured["url"] = url
        return MockResponse({
            "choices": [{"message": {"content": "Ollama local model response"}}]
        })

    monkeypatch.setattr("requests.post", mock_post)

    provider = OpenAICompatibleProvider(
        base_url="http://localhost:11434/v1",
        model_name="llama3.2",
    )
    assert provider.is_configured() is True

    reply = provider.generate_response([{"role": "user", "content": "Local model query"}])
    assert reply == "Ollama local model response"
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"


def test_offline_bio_agent_provider_richness():
    """OfflineBioAgentProvider handles domain intents with zero keys."""
    provider = OfflineBioAgentProvider()
    assert provider.is_configured() is True
    assert provider.requires_api_key is False

    mock_prediction = {
        "predicted_class": "BRCA",
        "confidence": 0.85,
        "class_probabilities": {"BRCA": 0.85, "LUAD": 0.10, "KIRC": 0.05},
        "top_features": [
            {"gene": "ESR1", "expression": 3.4, "attribution": 0.42},
            {"gene": "GATA3", "expression": 2.8, "attribution": 0.38},
        ],
        "suppressed_features": [
            {"gene": "TP53", "expression": -1.9, "attribution": -0.31},
        ],
        "low_confidence": False,
    }

    # Query 1: Suppressed genes
    resp_sup = provider.interpret("What tumor suppressors are lost?", mock_prediction)
    assert "Suppressed Biomarkers" in resp_sup
    assert "TP53" in resp_sup

    # Query 2: Drivers
    resp_drv = provider.interpret("What are the driving biomarkers?", mock_prediction)
    assert "Activating Drivers" in resp_drv
    assert "ESR1" in resp_drv

    # Query 3: Confidence
    resp_cnf = provider.interpret("Check model confidence and warnings.", mock_prediction)
    assert "85.0%" in resp_cnf
    assert "High Confidence Call" in resp_cnf

    # Query 4: IHC Recommendations
    resp_ihc = provider.interpret("Recommend confirmatory IHC panels.", mock_prediction)
    assert "ER (ESR1)" in resp_ihc
    assert "HER2" in resp_ihc
    assert "GATA3" in resp_ihc

    # Query 5: Attached file evidence synthesis
    file_summaries = [
        {"filename": "pathology.txt", "summary_text": "Invasive ductal carcinoma, T2N0M0, ER+ PR+."}
    ]
    resp_file = provider.interpret("Synthesize attached pathology file.", mock_prediction, file_summaries)
    assert "pathology.txt" in resp_file
    assert "Invasive ductal carcinoma" in resp_file

    # Query 6: No prediction loaded
    resp_no_pred = provider.interpret("Hello", None, file_summaries)
    assert "pathology.txt" in resp_no_pred
    assert "Please load a sample" in resp_no_pred


# =============================================================================
# 2. PROVIDER REGISTRY & FALLBACK TESTS
# =============================================================================

def test_registry_provider_listing():
    """Registry lists all providers and configuration status."""
    registry = ProviderRegistry()
    providers = registry.list_providers()
    provider_ids = [p["id"] for p in providers]
    assert "offline" in provider_ids
    assert "gemini" in provider_ids
    assert "openai" in provider_ids
    assert "anthropic" in provider_ids
    assert "openai_compatible" in provider_ids


def test_registry_resolution_precedence(monkeypatch):
    """Test resolution order: Request Config > AI_PROVIDER env > Key presence > Offline default."""
    registry = ProviderRegistry()

    # 1. Default when no env and no config -> offline
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert registry.resolve_provider_name() == "offline"

    # 2. Key presence auto-detection
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert registry.resolve_provider_name() == "openai"

    # 3. AI_PROVIDER env overrides key presence
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    assert registry.resolve_provider_name() == "gemini"

    # 4. Request config overrides AI_PROVIDER env
    req_config = ProviderConfig(provider_name="anthropic", api_key="sk-ant")
    assert registry.resolve_provider_name(req_config) == "anthropic"


def test_registry_graceful_fallback_when_key_missing():
    """When an external provider without keys is requested, fallback to OfflineBioAgent."""
    registry = ProviderRegistry()
    config = ProviderConfig(provider_name="gemini", api_key="")  # No key
    provider = registry.get_provider(config)
    assert isinstance(provider, OfflineBioAgentProvider)


# =============================================================================
# 3. FILE PARSER UNIT TESTS
# =============================================================================

def test_file_parser_csv_header_orientation():
    """Parse CSV where genes are in columns."""
    csv_content = "gene_1,gene_2,gene_3\n2.5,4.1,0.0\n"
    res = parse_attached_file("expression.csv", csv_content, feature_names=["gene_1", "gene_2", "gene_3"])
    assert res.file_type == "tabular"
    assert res.tabular_stats["num_genes"] == 3
    assert res.tabular_stats["matched_features_count"] == 3
    assert res.extracted_vector == [2.5, 4.1, 0.0]
    assert "3 features analyzed" in res.summary_text


def test_file_parser_tsv_row_orientation():
    """Parse TSV where genes are in rows."""
    tsv_content = "Gene\tSample_01\ngene_A\t1.5\ngene_B\t3.0\ngene_C\t0.0\n"
    res = parse_attached_file("matrix.tsv", tsv_content)
    assert res.file_type == "tabular"
    assert res.tabular_stats["num_genes"] == 3
    assert res.tabular_stats["min_expression"] == 0.0
    assert res.tabular_stats["max_expression"] == 3.0


def test_file_parser_clinical_text_note():
    """Extract TNM stage, histology, IHC markers, and age from clinical text."""
    note = (
        "Patient is a 62yo woman with biopsy-proven invasive ductal carcinoma. "
        "Clinical stage IIA (T2N0M0). "
        "Immunohistochemistry: ER positive (90%), PR positive (80%), HER2 negative (1+), Ki-67 22%. "
        "Patient previously underwent lumpectomy and is planned for chemotherapy."
    )
    res = parse_attached_file("patient_note.txt", note)
    assert res.file_type == "clinical_note"
    entities = res.detected_entities
    assert entities["age"] == 62
    assert entities["sex"] == "Female"
    assert entities["tnm_stage"] == "T2N0M0"
    assert "Invasive Ductal Carcinoma" in entities["histology"]
    assert "ER" in entities["ihc_markers"]
    assert "HER2" in entities["ihc_markers"]
    assert "Chemotherapy" in entities["prior_treatments"]


def test_file_parser_kidney_and_lung_markers():
    """Extract non-breast markers such as CD10, Pax-8, TTF-1."""
    kidney_note = "Clear cell renal cell carcinoma, CD10 positive, Pax-8 positive, CA9 positive, T1bN0M0."
    res = parse_attached_file("kirc.txt", kidney_note)
    assert "Clear Cell Renal Cell Carcinoma" in res.detected_entities["histology"]
    assert res.detected_entities["ihc_markers"]["CD10"] == "positive"
    assert res.detected_entities["ihc_markers"]["PAX-8"] == "positive"

    lung_note = "Lung adenocarcinoma, TTF-1 positive, Napsin-A positive, CK7 positive."
    res_lung = parse_attached_file("luad.txt", lung_note)
    assert "Lung Adenocarcinoma" in res_lung.detected_entities["histology"]
    assert res_lung.detected_entities["ihc_markers"]["TTF-1"] == "positive"


def test_file_parser_genomic_json():
    """Parse JSON with gene expressions and variant lists."""
    json_data = {
        "sample_id": "PT-99",
        "stage": "Stage III",
        "mutations": ["TP53 R175H", "KRAS G12D"],
        "ihc": {"ER": "negative", "HER2": "3+"}
    }
    res = parse_attached_file("genomics.json", json.dumps(json_data))
    assert res.file_type == "genomic_json"
    assert res.detected_entities["sample_id"] == "PT-99"
    assert "TP53 R175H" in res.detected_entities["variants"]


def test_file_parser_empty_and_corrupt_edge_cases():
    """Graceful handling of empty or corrupt inputs."""
    # Empty string
    res_empty = parse_attached_file("empty.txt", "")
    assert res_empty.file_type == "unknown"
    assert "empty" in res_empty.summary_text.lower()

    # Corrupt JSON
    res_bad_json = parse_attached_file("bad.json", "{not-valid-json")
    assert res_bad_json.file_type == "genomic_json"
    assert "failed to parse json" in res_bad_json.summary_text.lower()

    # Oversized file simulation
    big_text = "x" * (6 * 1024 * 1024)
    res_big = parse_attached_file("big.txt", big_text)
    assert "exceeds maximum allowed size" in res_big.summary_text


# =============================================================================
# 4. CLINICAL INTERPRETATION AGENT PIPELINE TESTS
# =============================================================================

def test_clinical_interpretation_agent_4stage_run():
    """Agent orchestrates all 4 stages: parse file, assemble dossier, dispatch, synthesize insights."""
    agent = ClinicalInterpretationAgent()

    req = ChatRequest(
        message="Provide clinical differential and confirmatory IHC recommendations.",
        history=[],
        prediction={
            "predicted_class": "KIRC",
            "confidence": 0.92,
            "class_probabilities": {"KIRC": 0.92, "PRAD": 0.05, "BRCA": 0.03},
            "top_features": [{"gene": "VHL", "expression": 3.1, "attribution": 0.38}],
            "suppressed_features": [{"gene": "CA9", "expression": -2.1, "attribution": -0.25}],
            "low_confidence": False,
        },
        attached_files=[
            AttachedFile(
                filename="biopsy.txt",
                content="Clear cell renal cell carcinoma, CD10 positive, Pax-8 positive."
            )
        ],
        provider_config=ProviderConfig(provider_name="offline")
    )

    response = agent.run(req)
    assert response.provider_used == "Offline Clinical Bio-Agent"
    assert "KIRC" in response.reply
    assert len(response.file_summaries) == 1
    assert response.file_summaries[0]["filename"] == "biopsy.txt"

    insights = response.structured_insights
    assert insights["predicted_class"] == "KIRC"
    assert insights["confidence"] == 0.92
    assert "CD10 (MME)" in insights["ihc_recommendations"] or "Pax-8" in insights["ihc_recommendations"]


def test_clinical_interpretation_agent_graceful_provider_fallback(monkeypatch):
    """When a remote provider crashes, agent falls back to OfflineBioAgent with notice."""
    class CrashingProvider(BaseAIProvider):
        provider_id = "crashing"
        display_name = "Crashing Provider"
        def _detect_env_key(self): return "key"
        def generate_response(self, messages, system_prompt="", **kwargs):
            raise ConnectionError("Upstream API is down.")

    registry = ProviderRegistry()
    registry.register("crashing", CrashingProvider)

    agent = ClinicalInterpretationAgent(provider_registry=registry)

    req = ChatRequest(
        message="What are the top biomarkers?",
        prediction={
            "predicted_class": "LUAD",
            "confidence": 0.81,
            "top_features": [{"gene": "EGFR", "expression": 2.5, "attribution": 0.3}],
            "suppressed_features": [],
            "class_probabilities": {"LUAD": 0.81},
        },
        provider_config=ProviderConfig(provider_name="crashing", api_key="dummy")
    )

    resp = agent.run(req)
    assert "Notice: Crashing Provider encountered an error" in resp.reply
    assert "Offline Clinical Bio-Agent" in resp.provider_used
    assert "EGFR" in resp.reply


# =============================================================================
# 5. INTEGRATION ENDPOINTS (FastAPI TestClient)
# =============================================================================

def test_api_get_ai_providers():
    """GET /ai/providers returns catalog of providers."""
    client = TestClient(app)
    res = client.get("/ai/providers")
    assert res.status_code == 200
    data = res.json()
    assert "providers" in data
    assert "active_default" in data
    assert any(p["id"] == "offline" for p in data["providers"])
    assert any(p["id"] == "gemini" for p in data["providers"])


def test_api_chat_upload_multipart():
    """POST /chat/upload accepts multipart form file."""
    client = TestClient(app)
    file_bytes = b"Sample_A,Sample_B\ngene_1,2.5,1.2\ngene_2,3.1,0.0\n"
    files = {"file": ("test_matrix.csv", io.BytesIO(file_bytes), "text/csv")}

    res = client.post("/chat/upload", files=files)
    assert res.status_code == 200
    body = res.json()
    assert body["filename"] == "test_matrix.csv"
    assert body["file_type"] == "tabular"
    assert "features analyzed" in body["summary_text"]


def test_api_chat_upload_json():
    """POST /chat/upload accepts JSON AttachedFile payload."""
    client = TestClient(app)
    payload = {
        "filename": "clinical.txt",
        "content": "Patient has Stage IIA (T2N0M0) breast invasive ductal carcinoma. ER+ PR+ HER2-."
    }
    res = client.post("/chat/upload", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["file_type"] == "clinical_note"
    assert body["detected_entities"]["tnm_stage"] == "T2N0M0"


def test_api_chat_with_attachments_and_provider_config():
    """POST /chat with attached files and custom provider config returns rich synthesis."""
    client = TestClient(app)
    chat_payload = {
        "message": "Synthesize this patient case with IHC recommendations.",
        "history": [],
        "prediction": {
            "predicted_class": "COAD",
            "confidence": 0.89,
            "class_probabilities": {"COAD": 0.89, "LUAD": 0.08, "BRCA": 0.03},
            "top_features": [{"gene": "APC", "expression": 2.8, "attribution": 0.35}],
            "suppressed_features": [{"gene": "MLH1", "expression": -1.9, "attribution": -0.22}],
            "low_confidence": False,
        },
        "attached_files": [
            {
                "filename": "colon_pathology.txt",
                "content": "Colon adenocarcinoma, moderately differentiated, CDX2 positive, CK20 positive, CK7 negative."
            }
        ],
        "provider_config": {
            "provider_name": "offline"
        }
    }

    res = client.post("/chat", json=chat_payload)
    assert res.status_code == 200
    data = res.json()
    assert "reply" in data
    assert "colon_pathology.txt" in data["reply"]
    assert data["provider_used"] == "Offline Clinical Bio-Agent"
    assert data["structured_insights"]["predicted_class"] == "COAD"
    assert "CDX2" in data["structured_insights"]["ihc_recommendations"] or "CK20 (KRT20)" in data["structured_insights"]["ihc_recommendations"]


def test_api_chat_backward_compatibility():
    """POST /chat works with legacy payload without attached_files or provider_config."""
    client = TestClient(app)
    legacy_payload = {
        "message": "Which genes drive this prediction?",
        "history": [],
        "prediction": {
            "predicted_class": "BRCA",
            "confidence": 0.80,
            "class_probabilities": {"BRCA": 0.80, "KIRC": 0.20},
            "top_features": [{"gene": "gene_11457", "expression": 1.5, "attribution": 0.2}],
            "suppressed_features": [],
            "low_confidence": False,
        }
    }

    res = client.post("/chat", json=legacy_payload)
    assert res.status_code == 200
    data = res.json()
    assert "reply" in data
    assert "gene_11457" in data["reply"]
    assert "Activating Drivers" in data["reply"]


def test_file_parser_base64_and_whitespace():
    """Test base64-encoded file decoding and whitespace handling."""
    import base64
    raw = "TP53,BRCA1\n2.1,3.4\n"
    b64 = "data:text/csv;base64," + base64.b64encode(raw.encode("utf-8")).decode("utf-8")
    res = parse_attached_file("encoded.csv", b64)
    assert res.file_type == "tabular"
    assert res.tabular_stats["num_genes"] == 2

    # Whitespace only
    res_ws = parse_attached_file("spaces.txt", "   \n\t  \n  ")
    assert res_ws.file_type == "unknown"
    assert "empty" in res_ws.summary_text.lower()


def test_chat_with_none_prediction_and_attached_file():
    """POST /chat works when prediction is None but file is attached."""
    client = TestClient(app)
    payload = {
        "message": "What is in this attached file?",
        "history": [],
        "prediction": None,
        "attached_files": [
            {
                "filename": "clinical.txt",
                "content": "Patient 45yo female with Stage I breast cancer."
            }
        ],
        "provider_config": {"provider_name": "offline"}
    }
    res = client.post("/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "clinical.txt" in data["reply"]
    assert "clinical.txt" in data["file_summaries"][0]["filename"]
    assert data["structured_insights"]["confidence_status"] == "No Prediction Loaded"


def test_tabular_small_panels_and_headerless_parsing():
    """Verify small gene panels and headerless tables are accurately parsed without inverted orientation."""
    # 1. 2 genes x 3 columns
    panel = "Gene,Sample1,Sample2\nTP53,1.2,2.3\nBRCA1,0.5,1.1\n"
    res1 = parse_attached_file("panel.csv", panel)
    assert res1.tabular_stats["num_genes"] == 2
    genes1 = [item["gene"] for item in res1.tabular_stats["top_expressed"]]
    assert "TP53" in genes1 and "BRCA1" in genes1
    assert "Sample1" not in genes1

    # 2. 1 gene in a 2-column key-value file
    single = "Gene,Expression\nTP53,1.2\n"
    res2 = parse_attached_file("single.csv", single)
    assert res2.tabular_stats["num_genes"] == 1
    assert res2.tabular_stats["top_expressed"][0]["gene"] == "TP53"
    assert res2.tabular_stats["top_expressed"][0]["value"] == 1.2

    # 3. Headerless 2-column table
    no_hdr = "TP53,1.2\nBRCA1,0.5\n"
    res3 = parse_attached_file("no_hdr.csv", no_hdr)
    assert res3.tabular_stats["num_genes"] == 2
    genes3 = [item["gene"] for item in res3.tabular_stats["top_expressed"]]
    assert "TP53" in genes3 and "BRCA1" in genes3

    # 4. Semicolon delimited European CSV
    semi = "Gene;Expression\nPTEN;3.4\nEGFR;2.1\n"
    res4 = parse_attached_file("semi.csv", semi)
    assert res4.tabular_stats["num_genes"] == 2
    genes4 = [item["gene"] for item in res4.tabular_stats["top_expressed"]]
    assert "PTEN" in genes4 and "EGFR" in genes4


def test_clinical_text_hyphenated_markers_and_pathological_staging():
    """Verify hyphenated IHC markers (e.g. ER-positive) are NOT inverted to negative, and pathological TNM is captured."""
    note = (
        "Patient is a 58yo woman. Age: 58. Gender: Female. "
        "Pathological staging shows pT2aN0M0 invasive ductal carcinoma. "
        "Biomarkers: ER-positive, PR-positive, HER2-negative, Ki-67 15%, Pax-8-positive, TTF-1-positive."
    )
    res = parse_attached_file("pathology_hyphens.txt", note)
    entities = res.detected_entities
    assert entities["age"] == 58
    assert entities["sex"] == "Female"
    assert entities["tnm_stage"] == "PT2AN0M0"
    assert entities["ihc_markers"]["ER"] == "positive"
    assert entities["ihc_markers"]["PR"] == "positive"
    assert entities["ihc_markers"]["HER2"] == "negative"
    assert entities["ihc_markers"]["PAX-8"] == "positive"
    assert entities["ihc_markers"]["TTF-1"] == "positive"


def test_genomic_json_nested_expression():
    """Verify JSON files with nested gene_values dict/list are extracted with rich stats."""
    json_dict = {
        "sample_id": "PT-001",
        "cohort": "BRCA",
        "gene_values": {"TP53": 1.5, "BRCA1": 2.5, "ESR1": 3.5}
    }
    res_dict = parse_attached_file("nested_dict.json", json.dumps(json_dict))
    assert res_dict.detected_entities["sample_id"] == "PT-001"
    assert res_dict.tabular_stats is not None
    assert res_dict.tabular_stats["num_genes"] == 3
    assert res_dict.tabular_stats["min_expression"] == 1.5
    assert res_dict.tabular_stats["max_expression"] == 3.5

    json_list = {
        "sample_id": "PT-002",
        "gene_values": [1.0, 2.0, 3.0, 4.0, 5.0]
    }
    res_list = parse_attached_file("nested_list.json", json.dumps(json_list))
    assert res_list.detected_entities["sample_id"] == "PT-002"
    assert res_list.tabular_stats is not None
    assert res_list.tabular_stats["num_genes"] == 5
    assert res_list.tabular_stats["mean_expression"] == 3.0


def test_chat_message_deduplication():
    """Verify that when history already ends with the user message, it is not duplicated."""
    class EchoProvider(BaseAIProvider):
        provider_id = "echo"
        display_name = "Echo"
        requires_api_key = False
        def _detect_env_key(self): return ""
        def generate_response(self, messages, system_prompt="", **kwargs):
            user_msgs = [m["content"] for m in messages if m["role"] == "user"]
            return f"User messages count: {len(user_msgs)}"

    registry = ProviderRegistry()
    registry.register("echo", EchoProvider)
    agent = ClinicalInterpretationAgent(provider_registry=registry)

    # History already ends with user query
    req = ChatRequest(
        message="What is BRCA?",
        history=[{"role": "user", "content": "What is BRCA?"}],
        provider_config=ProviderConfig(provider_name="echo")
    )
    res = agent.run(req)
    assert res.reply == "User messages count: 1"


def test_registry_auto_provider_resolution(monkeypatch):
    """Verify ProviderConfig with provider_name='auto' detects server environment keys."""
    registry = ProviderRegistry()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-test")
    monkeypatch.delenv("AI_PROVIDER", raising=False)

    config = ProviderConfig(provider_name="auto")
    resolved = registry.resolve_provider_name(config)
    assert resolved == "openai"


def test_api_chat_upload_oversized_file():
    """Verify POST /chat/upload rejects files exceeding 5MB with 413 or size warning."""
    client = TestClient(app)
    large_bytes = b"x" * (6 * 1024 * 1024)
    files = {"file": ("huge.txt", io.BytesIO(large_bytes), "text/plain")}
    res = client.post("/chat/upload", files=files)
    assert res.status_code == 413


def test_registry_invalid_provider_fallback():
    """Verify unknown provider name gracefully falls back to offline bio-agent."""
    registry = ProviderRegistry()
    config = ProviderConfig(provider_name="nonexistent_llm")
    provider = registry.get_provider(config)
    assert isinstance(provider, OfflineBioAgentProvider)

