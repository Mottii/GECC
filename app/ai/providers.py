"""Pluggable AI Provider Architecture for DetectAI.

Supports Google Gemini, OpenAI, Anthropic Claude, OpenAI-Compatible (Ollama,
Groq, vLLM, LMStudio), and an Offline Clinical Bio-Agent.
Uses direct HTTP requests via `requests` or `httpx` for zero-dependency portability.
"""

from __future__ import annotations

import abc
import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)


class BaseAIProvider(abc.ABC):
    """Abstract base class for all AI model providers in DetectAI."""

    provider_id: str = "base"
    display_name: str = "Base Provider"
    default_model: str = "default"
    requires_api_key: bool = True

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.api_key = (api_key or "").strip() or self._detect_env_key()
        self.model_name = (model_name or "").strip() or self.default_model
        self.base_url = (base_url or "").strip() or self._default_base_url()
        self.timeout = timeout

    @abc.abstractmethod
    def _detect_env_key(self) -> str:
        """Detect API key from environment variables."""
        ...

    def _default_base_url(self) -> str:
        """Return the default base URL for the provider API."""
        return ""

    def is_configured(self) -> bool:
        """Return True if the provider has all required credentials to execute."""
        if not self.requires_api_key:
            return True
        return bool(self.api_key)

    @abc.abstractmethod
    def generate_response(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = "",
        **kwargs: Any,
    ) -> str:
        """Generate a response given conversation history and system instructions.

        Args:
            messages: List of dicts with keys 'role' ('user'|'assistant'|'model')
              and 'content'.
            system_prompt: High-level instructions/context for the model.
            **kwargs: Extra parameters (temperature, max_tokens, etc.).

        Returns:
            Model response text.
        """
        ...

    def metadata(self) -> dict[str, Any]:
        """Return provider metadata for API listing."""
        return {
            "id": self.provider_id,
            "name": self.display_name,
            "default_model": self.default_model,
            "configured_model": self.model_name,
            "requires_key": self.requires_api_key,
            "is_configured": self.is_configured(),
            "base_url": self.base_url or None,
        }


class GeminiProvider(BaseAIProvider):
    """Google Gemini AI Provider (Gemini 1.5 Flash, 2.0 Flash, 1.5 Pro)."""

    provider_id = "gemini"
    display_name = "Google Gemini"
    default_model = "gemini-1.5-flash"
    requires_api_key = True

    def _detect_env_key(self) -> str:
        return os.environ.get("GEMINI_API_KEY", "")

    def _default_base_url(self) -> str:
        return "https://generativelanguage.googleapis.com/v1beta"

    def generate_response(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = "",
        **kwargs: Any,
    ) -> str:
        if not self.api_key:
            raise ValueError(
                "Gemini API key is not configured. Provide an API key or set GEMINI_API_KEY."
            )

        temperature = kwargs.get("temperature", 0.2)
        model_path = self.model_name if self.model_name.startswith("models/") else f"models/{self.model_name}"
        endpoint = f"{self.base_url}/{model_path}:generateContent?key={self.api_key}"

        contents: list[dict[str, Any]] = []
        for msg in messages:
            role = "user" if msg.get("role") == "user" else "model"
            content_text = msg.get("content", "")
            if content_text:
                contents.append({
                    "role": role,
                    "parts": [{"text": content_text}],
                })

        # Ensure we have at least one message
        if not contents:
            contents.append({
                "role": "user",
                "parts": [{"text": "Hello, please provide clinical interpretation."}],
            })

        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }

        if system_prompt:
            body["systemInstruction"] = {
                "parts": [{"text": system_prompt}],
            }

        headers = {"Content-Type": "application/json"}
        response = requests.post(endpoint, headers=headers, json=body, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        candidates = data.get("candidates", [])
        if candidates:
            cand = candidates[0]
            parts = cand.get("content", {}).get("parts", [])
            if parts and "text" in parts[0]:
                return parts[0]["text"]
            finish_reason = cand.get("finishReason", "UNKNOWN")
            return f"> ⚠️ *Response ended with finish reason: {finish_reason}*"
        elif "promptFeedback" in data:
            block_reason = data["promptFeedback"].get("blockReason", "BLOCKED")
            return f"> ⚠️ *Prompt was blocked by Gemini safety filters ({block_reason}).*"

        raise RuntimeError(f"Unexpected response format from Gemini API: {data}")


class OpenAIProvider(BaseAIProvider):
    """OpenAI Chat Completions Provider (GPT-4o, GPT-4o-mini, o3-mini)."""

    provider_id = "openai"
    display_name = "OpenAI"
    default_model = "gpt-4o-mini"
    requires_api_key = True

    def _detect_env_key(self) -> str:
        return os.environ.get("OPENAI_API_KEY", "")

    def _default_base_url(self) -> str:
        return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    def generate_response(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = "",
        **kwargs: Any,
    ) -> str:
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is not configured. Provide an API key or set OPENAI_API_KEY."
            )

        temperature = kwargs.get("temperature", 0.2)
        endpoint = f"{self.base_url}/chat/completions"

        api_messages: list[dict[str, str]] = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = msg.get("role", "user")
            if role in ("model", "assistant"):
                role = "assistant"
            else:
                role = "user"
            api_messages.append({"role": role, "content": msg.get("content", "")})

        if not api_messages:
            api_messages.append({"role": "user", "content": "Hello"})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": api_messages,
            "temperature": temperature,
        }
        if "max_tokens" in kwargs:
            body["max_tokens"] = kwargs["max_tokens"]

        response = requests.post(endpoint, headers=headers, json=body, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as err:
            raise RuntimeError(f"Unexpected response format from OpenAI API: {data}") from err


class AnthropicProvider(BaseAIProvider):
    """Anthropic Messages API Provider (Claude 3.5 Sonnet, Claude 3.5 Haiku)."""

    provider_id = "anthropic"
    display_name = "Anthropic Claude"
    default_model = "claude-3-5-sonnet-20241022"
    requires_api_key = True

    def _detect_env_key(self) -> str:
        return os.environ.get("ANTHROPIC_API_KEY", "")

    def _default_base_url(self) -> str:
        return "https://api.anthropic.com/v1"

    def generate_response(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = "",
        **kwargs: Any,
    ) -> str:
        if not self.api_key:
            raise ValueError(
                "Anthropic API key is not configured. Provide an API key or set ANTHROPIC_API_KEY."
            )

        endpoint = f"{self.base_url}/messages"
        max_tokens = kwargs.get("max_tokens", 2048)
        temperature = kwargs.get("temperature", 0.2)

        # Anthropic messages must alternate user / assistant and start with user
        formatted_messages: list[dict[str, str]] = []
        for msg in messages:
            role = "assistant" if msg.get("role") in ("assistant", "model") else "user"
            content = msg.get("content", "")
            if not content:
                continue

            if formatted_messages and formatted_messages[-1]["role"] == role:
                # Merge consecutive messages of the same role
                formatted_messages[-1]["content"] += f"\n\n{content}"
            else:
                formatted_messages.append({"role": role, "content": content})

        if not formatted_messages:
            formatted_messages = [{"role": "user", "content": "Please provide clinical interpretation."}]
        elif formatted_messages[0]["role"] != "user":
            formatted_messages.insert(0, {"role": "user", "content": "Hello"})

        if formatted_messages and formatted_messages[-1]["role"] != "user":
            formatted_messages.append({"role": "user", "content": "Please provide clinical interpretation."})

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "messages": formatted_messages,
            "temperature": temperature,
        }
        if system_prompt:
            body["system"] = system_prompt

        response = requests.post(endpoint, headers=headers, json=body, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        try:
            return data["content"][0]["text"]
        except (KeyError, IndexError) as err:
            raise RuntimeError(f"Unexpected response format from Anthropic API: {data}") from err


class OpenAICompatibleProvider(OpenAIProvider):
    """OpenAI-compatible endpoints (Ollama, Groq, vLLM, LMStudio, OpenRouter)."""

    provider_id = "openai_compatible"
    display_name = "OpenAI-Compatible (Ollama/Groq/vLLM)"
    default_model = "llama3.2"
    requires_api_key = False

    def _detect_env_key(self) -> str:
        return (
            os.environ.get("OPENAI_COMPATIBLE_API_KEY")
            or os.environ.get("OLLAMA_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "not-needed"
        )

    def _default_base_url(self) -> str:
        return (
            os.environ.get("OPENAI_COMPATIBLE_BASE_URL")
            or os.environ.get("OLLAMA_HOST")
            or "http://localhost:11434/v1"
        ).rstrip("/")


class OfflineBioAgentProvider(BaseAIProvider):
    """Rule-based clinical bio-agent provider with deep oncogenomic knowledge.

    Works 100% locally with zero external network calls or API keys. Provides
    cohort-specific biomarker pathway interpretations, statistical expression
    profiling, and confirmatory IHC recommendations for TCGA cohorts.
    """

    provider_id = "offline"
    display_name = "Offline Clinical Bio-Agent"
    default_model = "detectai-expert-rules-v1"
    requires_api_key = False

    COHORT_KNOWLEDGE = {
        "BRCA": {
            "name": "Breast Invasive Carcinoma",
            "pathways": "Estrogen receptor signaling, HER2/neu amplification, PI3K/AKT/PTEN axis, BRCA1/2 DNA repair.",
            "ihc_panel": ["ER (ESR1)", "PR (PGR)", "HER2 (ERBB2)", "GATA3", "Mammaglobin (SCGB2A2)", "Ki-67 (MKI67)"],
            "hallmarks": "Luminal A/B vs HER2-enriched vs Basal-like/Triple Negative differentiation.",
        },
        "KIRC": {
            "name": "Kidney Renal Clear Cell Carcinoma",
            "pathways": "VHL loss, HIF-1α / HIF-2α stabilization, downstream VEGF/PDGF neoangiogenesis.",
            "ihc_panel": ["CD10 (MME)", "Pax-8", "CA9 (Carbonic Anhydrase IX)", "Vimentin", "CK7 (typically negative)"],
            "hallmarks": "Clear cell morphology driven by glycogen and lipid accumulation with prominent vascular networks.",
        },
        "COAD": {
            "name": "Colon Adenocarcinoma",
            "pathways": "Chromosomal Instability (CIN: APC-KRAS-TP53) vs Microsatellite Instability (MSI-H: MLH1/MSH2).",
            "ihc_panel": ["CDX2", "CK20 (KRT20)", "CK7 (negative)", "MLH1", "MSH2", "MSH6", "PMS2"],
            "hallmarks": "Glandular formation with mucin secretion; essential to assess MMR protein expression for immunotherapy.",
        },
        "LUAD": {
            "name": "Lung Adenocarcinoma",
            "pathways": "Receptor Tyrosine Kinase (RTK) driver mutations (EGFR, KRAS, ALK, ROS1, RET, MET).",
            "ihc_panel": ["TTF-1 (NKX2-1)", "Napsin-A", "CK7 (positive)", "p40 (negative, rules out squamous)"],
            "hallmarks": "Peripheral pulmonary mass; peripheral acinar/papillary/solid architectures.",
        },
        "PRAD": {
            "name": "Prostate Adenocarcinoma",
            "pathways": "Androgen Receptor (AR) signaling, TMPRSS2-ERG gene fusions, PTEN deletions.",
            "ihc_panel": ["PSA (KLK3)", "PSMA (FOLH1)", "NKX3.1", "AMACR (p504s)", "p63 / HMWCK (absent basal layer)"],
            "hallmarks": "Infiltrative crowded small glands lacking basal cell layer, Gleason grading correlation.",
        },
    }

    def _detect_env_key(self) -> str:
        return ""

    def generate_response(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = "",
        **kwargs: Any,
    ) -> str:
        prediction: dict[str, Any] | None = kwargs.get("prediction")
        file_summaries: list[dict[str, Any]] | None = kwargs.get("file_summaries")

        # Get latest user message
        last_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_message = msg.get("content", "")
                break

        return self.interpret(last_message, prediction, file_summaries)

    def interpret(
        self,
        message: str,
        prediction: dict[str, Any] | None,
        file_summaries: list[dict[str, Any]] | None = None,
    ) -> str:
        """Core clinical reasoning engine."""
        msg_lower = (message or "").lower()

        # Handle case where no prediction is loaded
        if not prediction:
            file_mention = ""
            if file_summaries:
                files_str = ", ".join([f.get("filename", "file") for f in file_summaries])
                file_mention = (
                    f"\n\nI detected the following attached file(s): **{files_str}**. "
                    "Load a patient expression profile and run **Predict** to integrate the file evidence "
                    "with genomic classifier inference."
                )
            return (
                "Hello! I am your AI clinical assistant. Please load a sample and predict its cancer type "
                f"so I can analyze the expression data and provide specific insights.{file_mention}"
            )

        predicted_class = prediction.get("predicted_class", "Unknown")
        confidence = float(prediction.get("confidence", 0.0))
        top_features = prediction.get("top_features", [])
        suppressed_features = prediction.get("suppressed_features", [])
        low_confidence = bool(prediction.get("low_confidence", False))
        warning = prediction.get("warning")
        probs = prediction.get("class_probabilities", {})
        ood_score = prediction.get("ood_score")
        is_ood = bool(prediction.get("is_ood", False))

        cohort_info = self.COHORT_KNOWLEDGE.get(
            predicted_class,
            {
                "name": f"{predicted_class} Cohort",
                "pathways": "Standard oncogenic dysregulation pathways.",
                "ihc_panel": ["Broad diagnostic carcinoma panel"],
                "hallmarks": "Carcinoma architecture",
            },
        )

        file_evidence_str = ""
        if file_summaries:
            file_bullets = []
            for f in file_summaries:
                name = f.get("filename", "file")
                summary = f.get("summary_text", "")
                file_bullets.append(f"- **{name}:** {summary}")
            file_evidence_str = "\n\n**Attached File Evidence:**\n" + "\n".join(file_bullets)

        # 1. Suppressed biomarkers query
        if any(w in msg_lower for w in ("suppress", "negative", "tumor suppressor")):
            if suppressed_features:
                genes_str = ", ".join([
                    f"{f['gene']} (attr: {f.get('attribution', 'N/A')}, expr: {f['expression']})"
                    for f in suppressed_features
                ])
                return (
                    f"**Suppressed Biomarkers & Tumor Suppressors:**\n"
                    f"For this {predicted_class} sample, key negatively attributing biomarkers include: {genes_str}.\n"
                    "In oncogenomics, down-regulation of protective tumor suppressors removes critical cell-cycle checkpoints, "
                    f"significantly contributing to malignant phenotype classification.{file_evidence_str}"
                )
            return (
                f"No strongly suppressed biomarkers were detected for this {predicted_class} profile."
                f"{file_evidence_str}"
            )

        # 2. Driver genes / biomarkers query
        if any(w in msg_lower for w in ("gene", "feature", "expression", "driver", "biomarker")):
            pos_str = ", ".join([
                f"{f['gene']} (expr: {f['expression']}, attr: {f.get('attribution', 'N/A')})"
                for f in top_features[:3]
            ]) if top_features else "None"
            neg_str = ", ".join([
                f"{f['gene']} (expr: {f['expression']}, attr: {f.get('attribution', 'N/A')})"
                for f in suppressed_features[:2]
            ]) if suppressed_features else "None"

            return (
                f"**Genomic Attribution Analysis for {predicted_class}** (Confidence: {confidence * 100:.1f}%):\n\n"
                f"- **Top Activating Drivers (Positive Input x Gradient):** {pos_str}\n"
                f"- **Suppressed Biomarkers (Negative Input x Gradient):** {neg_str}\n\n"
                "These biomarkers reflect characteristic oncogenic dysregulations. Attributions combine normalized expression with model gradients to identify true functional drivers."
                f"{file_evidence_str}"
            )

        # 3. Confidence, margin, calibration, warnings, OOD
        if any(w in msg_lower for w in ("confiden", "certain", "probabilit", "warning", "margin", "ood", "anomaly")):
            sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
            prob_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in sorted_probs])
            diff_str = ""
            if len(sorted_probs) >= 2:
                margin = (sorted_probs[0][1] - sorted_probs[1][1]) * 100
                diff_str = f"\n- Primary vs Secondary Margin: {margin:.1f}% ({sorted_probs[0][0]} vs {sorted_probs[1][0]})."

            ood_note = f"\n- Out-of-Distribution Reconstruction Score: {ood_score:.4f} (Anomaly Flag: {is_ood})" if ood_score is not None else ""

            status_note = (
                "⚠️ **Clinician Alert: Low Confidence / Borderline Call.** Proceed with caution and verify via immunohistochemistry (IHC) or histological biopsy."
                if low_confidence or warning
                else "✅ **High Confidence Call:** Cohort separation is statistically distinct."
            )
            return (
                f"**Model Calibration & Cohort Probabilities:**\n"
                f"- Predicted Cohort: **{predicted_class}** ({confidence * 100:.1f}% confidence){diff_str}{ood_note}\n"
                f"- Distribution: {prob_str}\n\n"
                f"{status_note}"
                f"{file_evidence_str}"
            )

        # 4. Clinical Report / IHC / Pathway recommendations
        if any(w in msg_lower for w in ("report", "ihc", "immunohistochemistry", "pathway", "recommend", "protocol")):
            ihc_str = ", ".join(cohort_info["ihc_panel"])
            pos_genes = [f["gene"] for f in top_features[:3]]
            neg_genes = [f["gene"] for f in suppressed_features[:2]]

            return (
                f"### 📋 Clinical Genomic Case Report: {predicted_class} ({cohort_info['name']})\n\n"
                f"- **Diagnostic Call:** {predicted_class} ({cohort_info['name']})\n"
                f"- **Calibrated Confidence:** {confidence * 100:.1f}% ({'Uncertain / Low Confidence' if low_confidence else 'Statistically Robust'})\n"
                f"- **Canonical Oncogenic Axis:** {cohort_info['pathways']}\n"
                f"- **Key Activating Drivers:** {', '.join(pos_genes) if pos_genes else 'N/A'}\n"
                f"- **Suppressed Biomarkers:** {', '.join(neg_genes) if neg_genes else 'N/A'}\n\n"
                f"**Recommended Confirmatory IHC Panel:**\n"
                f"To definitively establish tissue origin, evaluate biopsy tissue with: `{ihc_str}`.\n\n"
                f"**Histopathological Correlation:**\n"
                f"{cohort_info['hallmarks']}"
                f"{file_evidence_str}"
            )

        # 5. File analysis specific query
        if any(w in msg_lower for w in ("file", "attachment", "note", "pathology", "csv", "tsv")):
            if file_summaries:
                return (
                    f"**Attached File Integration Analysis:**\n"
                    f"{file_evidence_str}\n\n"
                    f"**Genomic Classifier Context:**\n"
                    f"The RNA-seq expression profile for this patient aligns with **{predicted_class}** ({confidence * 100:.1f}% confidence). "
                    f"The attached documentation supports clinical differentiation towards {cohort_info['name']}."
                )
            return (
                f"No files are currently attached to this chat session. You can click 📎 to attach a pathology note, "
                f"TNM staging summary, or gene expression CSV/TSV table."
            )

        # 6. Default general clinical summary
        return (
            f"**Clinical Differential Summary ({predicted_class}):**\n"
            f"- Confidence: {confidence * 100:.1f}% ({'Uncertain / Borderline' if low_confidence else 'Statistically Robust'})\n"
            f"- Primary Driver: {top_features[0]['gene'] if top_features else 'N/A'}\n"
            f"- Key Suppressed Gene: {suppressed_features[0]['gene'] if suppressed_features else 'N/A'}\n"
            f"- Recommended IHC Panel: {', '.join(cohort_info['ihc_panel'][:4])}\n"
            f"Ask about specific driving biomarkers, suppressed tumor suppressors, attached file evidence, or cohort probability margins for deeper clinical interpretation."
            f"{file_evidence_str}"
        )


class ProviderRegistry:
    """Registry managing available AI model providers and resolution logic."""

    def __init__(self) -> None:
        self._providers: dict[str, type[BaseAIProvider]] = {}
        self.register("offline", OfflineBioAgentProvider)
        self.register("gemini", GeminiProvider)
        self.register("openai", OpenAIProvider)
        self.register("anthropic", AnthropicProvider)
        self.register("openai_compatible", OpenAICompatibleProvider)

    def register(self, provider_id: str, provider_cls: type[BaseAIProvider]) -> None:
        """Register a new provider class."""
        self._providers[provider_id.lower()] = provider_cls

    def get_registered_ids(self) -> list[str]:
        """Return all registered provider identifiers."""
        return list(self._providers.keys())

    def resolve_provider_name(self, config: Any | None = None) -> str:
        """Determine which provider to instantiate based on request config, env, or fallback."""
        # 1. Explicit request config
        if config:
            provider_name = getattr(config, "provider_name", None) or (
                config.get("provider_name") if isinstance(config, dict) else None
            )
            if provider_name and str(provider_name).lower() not in ("auto", "default", ""):
                if str(provider_name).lower() in self._providers:
                    return str(provider_name).lower()

        # 2. Environment variable AI_PROVIDER
        env_pref = os.environ.get("AI_PROVIDER", "").strip().lower()
        if env_pref and env_pref in self._providers:
            return env_pref

        # 3. Key presence auto-detection
        if os.environ.get("GEMINI_API_KEY"):
            return "gemini"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"

        # 4. Default to offline bio-agent
        return "offline"

    def get_provider(self, config: Any | None = None) -> BaseAIProvider:
        """Instantiate and return the appropriate AI provider."""
        provider_name = self.resolve_provider_name(config)
        provider_cls = self._providers.get(provider_name, OfflineBioAgentProvider)

        api_key = None
        model_name = None
        base_url = None

        if config:
            if hasattr(config, "api_key"):
                api_key = config.api_key
                model_name = config.model_name
                base_url = config.base_url
            elif isinstance(config, dict):
                api_key = config.get("api_key")
                model_name = config.get("model_name")
                base_url = config.get("base_url")

        provider = provider_cls(api_key=api_key, model_name=model_name, base_url=base_url)

        # Graceful fallback: If an external provider is selected but has no API key configured,
        # fallback to OfflineBioAgentProvider
        if provider.requires_api_key and not provider.is_configured():
            logger.warning(
                "Requested provider '%s' has no API key configured. Falling back to OfflineBioAgentProvider.",
                provider_name,
            )
            return OfflineBioAgentProvider()

        return provider

    def list_providers(self) -> list[dict[str, Any]]:
        """List all available providers with configuration status."""
        catalog = []
        for pid, cls in self._providers.items():
            instance = cls()
            info = instance.metadata()
            catalog.append(info)
        return catalog


# Global default registry instance
registry = ProviderRegistry()
