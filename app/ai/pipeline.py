"""Clinical Interpretation Agent Pipeline for DetectAI.

Orchestrates the 4-stage workflow:
1. File Ingestion & Profiling (via file_parser)
2. Clinical Case Dossier Aggregation
3. Standardized Prompt Construction & AI Provider Dispatch
4. Structured Output Generation & Interaction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.ai.file_parser import FileParseResult, parse_attached_file
from app.ai.providers import (
    BaseAIProvider,
    OfflineBioAgentProvider,
    ProviderRegistry,
    registry as default_registry,
)
from app.schemas import AttachedFile, ChatRequest, ChatResponse, ProviderConfig

logger = logging.getLogger(__name__)


@dataclass
class GenomicCaseDossier:
    """Consolidated clinical and genomic case information for interpretation."""

    prediction: dict[str, Any] | None
    attached_files: list[FileParseResult] = field(default_factory=list)
    chat_message: str = ""
    history: list[dict[str, str]] = field(default_factory=list)

    @property
    def predicted_class(self) -> str:
        if not self.prediction:
            return "N/A"
        return self.prediction.get("predicted_class", "Unknown")

    @property
    def confidence(self) -> float:
        if not self.prediction:
            return 0.0
        return float(self.prediction.get("confidence", 0.0))

    @property
    def low_confidence(self) -> bool:
        if not self.prediction:
            return False
        return bool(self.prediction.get("low_confidence", False))

    @property
    def top_features(self) -> list[dict[str, Any]]:
        if not self.prediction:
            return []
        return self.prediction.get("top_features", [])

    @property
    def suppressed_features(self) -> list[dict[str, Any]]:
        if not self.prediction:
            return []
        return self.prediction.get("suppressed_features", [])

    @property
    def class_probabilities(self) -> dict[str, float]:
        if not self.prediction:
            return {}
        return self.prediction.get("class_probabilities", {})

    @property
    def ood_score(self) -> float | None:
        if not self.prediction:
            return None
        return self.prediction.get("ood_score")

    @property
    def is_ood(self) -> bool:
        if not self.prediction:
            return False
        return bool(self.prediction.get("is_ood", False))

    @property
    def pca_coords(self) -> dict[str, float]:
        if not self.prediction:
            return {}
        return self.prediction.get("pca_coords", {})


class ClinicalInterpretationAgent:
    """Agent coordinating file parsing, provider selection, and clinical synthesis."""

    def __init__(
        self,
        provider_registry: ProviderRegistry | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self.registry = provider_registry or default_registry
        self.feature_names = feature_names

    def run(self, request: ChatRequest) -> ChatResponse:
        """Execute the 4-stage interpretation workflow."""
        # Stage 1: File Ingestion & Profiling
        parsed_files = self._stage1_ingest_files(request.attached_files)

        # Stage 2: Clinical Dossier Aggregation
        dossier = self._stage2_build_dossier(
            prediction=request.prediction,
            parsed_files=parsed_files,
            message=request.message,
            history=request.history,
        )

        # Stage 3: AI Provider Selection & Generation
        provider, reply_text = self._stage3_dispatch_and_generate(
            dossier=dossier,
            provider_config=request.provider_config,
        )

        # Stage 4: Structured Output & Clinical Insights Extraction
        file_summaries, structured_insights = self._stage4_synthesize_insights(
            dossier=dossier,
            reply_text=reply_text,
            provider=provider,
        )

        return ChatResponse(
            reply=reply_text,
            provider_used=provider.display_name,
            file_summaries=file_summaries,
            structured_insights=structured_insights,
        )

    # -------------------------------------------------------------------------
    # Stage 1: File Ingestion & QC
    # -------------------------------------------------------------------------
    def _stage1_ingest_files(
        self,
        attached_files: list[AttachedFile] | None,
    ) -> list[FileParseResult]:
        parsed_results: list[FileParseResult] = []
        if not attached_files:
            return parsed_results

        for af in attached_files:
            try:
                res = parse_attached_file(
                    filename=af.filename,
                    content=af.content,
                    feature_names=self.feature_names,
                )
                parsed_results.append(res)
            except Exception as exc:
                logger.warning("Error parsing attached file '%s': %s", af.filename, exc)
                parsed_results.append(
                    FileParseResult(
                        filename=af.filename,
                        file_type="unknown",
                        file_size_bytes=len(af.content.encode("utf-8")),
                        summary_text=f"Error parsing file: {exc}",
                        warnings=[str(exc)],
                    )
                )

        return parsed_results

    # -------------------------------------------------------------------------
    # Stage 2: Clinical Dossier Aggregator
    # -------------------------------------------------------------------------
    def _stage2_build_dossier(
        self,
        prediction: dict[str, Any] | None,
        parsed_files: list[FileParseResult],
        message: str,
        history: list[dict[str, Any]],
    ) -> GenomicCaseDossier:
        clean_history = []
        for h in history:
            role = str(h.get("role", "user"))
            content = str(h.get("content", ""))
            clean_history.append({"role": role, "content": content})

        return GenomicCaseDossier(
            prediction=prediction,
            attached_files=parsed_files,
            chat_message=message,
            history=clean_history,
        )

    # -------------------------------------------------------------------------
    # Stage 3: Prompt Construction & Dispatch
    # -------------------------------------------------------------------------
    def _stage3_dispatch_and_generate(
        self,
        dossier: GenomicCaseDossier,
        provider_config: ProviderConfig | None,
    ) -> tuple[BaseAIProvider, str]:
        provider = self.registry.get_provider(provider_config)
        system_prompt = self._construct_system_prompt(dossier)

        user_content = dossier.chat_message.strip()
        if not user_content:
            if dossier.attached_files:
                user_content = "Please analyze the attached file evidence and integrate it with clinical interpretation."
            else:
                user_content = "Please provide clinical interpretation."

        # Build message history for provider, avoiding duplicate prompt if already last in history
        messages = list(dossier.history)
        if not (messages and messages[-1].get("role") == "user" and messages[-1].get("content") == user_content):
            messages.append({"role": "user", "content": user_content})

        file_summaries_dicts = [f.to_dict() for f in dossier.attached_files]

        try:
            reply = provider.generate_response(
                messages=messages,
                system_prompt=system_prompt,
                prediction=dossier.prediction,
                file_summaries=file_summaries_dicts,
            )
            return provider, reply
        except Exception as exc:
            logger.warning(
                "Provider '%s' failed (%s). Gracefully falling back to OfflineBioAgentProvider.",
                provider.display_name,
                exc,
            )
            fallback_provider = OfflineBioAgentProvider()
            fallback_reply = fallback_provider.generate_response(
                messages=messages,
                system_prompt=system_prompt,
                prediction=dossier.prediction,
                file_summaries=file_summaries_dicts,
            )
            notice = f"> ⚠️ *Notice: {provider.display_name} encountered an error ({exc}). Falling back to Offline Bio-Agent.* \n\n"
            return fallback_provider, notice + fallback_reply

    def _construct_system_prompt(self, dossier: GenomicCaseDossier) -> str:
        """Build standardized clinical prompt with model predictions and file context."""
        parts = [
            "You are a clinical bioinformatics AI assistant for the Gene Expression Cancer Classifier (DetectAI).",
            "Your goal is to explain and interpret predictions made by the deep learning model to the user.",
            "",
        ]

        if dossier.prediction:
            parts.extend([
                "### Clinical Genomic Context:",
                f"- Model Prediction: {dossier.predicted_class}",
                f"- Calibrated Confidence: {dossier.confidence * 100:.2f}%",
                f"- Low Confidence Flag: {dossier.low_confidence}",
                f"- Class Probabilities: {dossier.class_probabilities}",
                f"- Top Contributing Oncogenes (Drivers): {dossier.top_features}",
                f"- Suppressed Biomarkers (Tumor Suppressors): {dossier.suppressed_features}",
            ])
            if dossier.ood_score is not None:
                parts.append(
                    f"- Autoencoder OOD Score: {dossier.ood_score:.4f} (Flagged Anomaly: {dossier.is_ood})"
                )
            if dossier.pca_coords:
                parts.append(f"- PCA 2D Coordinates: {dossier.pca_coords}")
        else:
            parts.extend([
                "### Clinical Genomic Context:",
                "- Model Prediction Status: No patient expression profile is currently loaded or classified.",
                "- Instruction: You can analyze general oncogenomics or attached files, and remind the user to load a sample and click Predict for model inference.",
            ])

        if dossier.attached_files:
            parts.append("")
            parts.append("### Attached Patient Documentation & Evidence:")
            for idx, f in enumerate(dossier.attached_files, 1):
                parts.append(f"{idx}. [{f.filename}] ({f.file_type}): {f.summary_text}")
                if f.detected_entities:
                    parts.append(f"   Entities: {f.detected_entities}")
                if f.tabular_stats:
                    parts.append(f"   Expression Stats: {f.tabular_stats}")

        parts.extend([
            "",
            "Respond directly, concisely, and professionally in markdown format. Do not use generic explanations; ",
            "always link the user's questions back to this patient's specific expression levels, attached files, and cohorts.",
        ])

        return "\n".join(parts)

    # -------------------------------------------------------------------------
    # Stage 4: Structured Output & Insights Extraction
    # -------------------------------------------------------------------------
    def _stage4_synthesize_insights(
        self,
        dossier: GenomicCaseDossier,
        reply_text: str,
        provider: BaseAIProvider,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        file_summaries = [f.to_dict() for f in dossier.attached_files]

        cohort = dossier.predicted_class
        cohort_info = OfflineBioAgentProvider.COHORT_KNOWLEDGE.get(cohort, {})
        ihc_panel = cohort_info.get("ihc_panel", ["Broad Pan-Cytokeratin panel"])

        if dossier.prediction:
            confidence_status = (
                "Borderline / Low Confidence" if dossier.low_confidence else "High Confidence"
            )
            top_drivers = [f.get("gene") for f in dossier.top_features[:3]]
            suppressed = [f.get("gene") for f in dossier.suppressed_features[:2]]
        else:
            confidence_status = "No Prediction Loaded"
            top_drivers = []
            suppressed = []

        structured_insights = {
            "predicted_class": cohort,
            "confidence": dossier.confidence,
            "confidence_status": confidence_status,
            "top_drivers": top_drivers,
            "suppressed_biomarkers": suppressed,
            "ihc_recommendations": ihc_panel if dossier.prediction else [],
            "attached_file_count": len(dossier.attached_files),
            "provider_used": provider.display_name,
        }

        return file_summaries, structured_insights
