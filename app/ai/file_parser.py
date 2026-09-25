"""File Ingestion & Data Profiler for DetectAI.

Parses tabular expression files (.csv, .tsv), clinical pathology notes (.txt, .md),
and genomic variant/annotation records (.json) to extract structured clinical evidence
and statistical profiles for the AI interpretation agent.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB


@dataclass
class FileParseResult:
    """Standardized representation of a parsed clinical or genomic file."""

    filename: str
    file_type: str  # "tabular", "clinical_note", "genomic_json", "unknown"
    file_size_bytes: int
    summary_text: str
    detected_entities: dict[str, Any] = field(default_factory=dict)
    tabular_stats: dict[str, Any] | None = None
    matched_genes: list[str] = field(default_factory=list)
    extracted_vector: list[float] | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _decode_content(content: str | bytes) -> str:
    """Safely decode string or base64 bytes/string to unicode text."""
    if isinstance(content, bytes):
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                # Might be base64-encoded binary/text
                decoded = base64.b64decode(content)
                return decoded.decode("utf-8", errors="replace")
            except Exception:
                return content.decode("latin-1", errors="replace")

    # If it's a string, check if it's base64 encoded
    content_str = content.strip()
    if content_str.startswith("data:") and ";base64," in content_str:
        # Data URL format
        _, b64_part = content_str.split(";base64,", 1)
        try:
            return base64.b64decode(b64_part).decode("utf-8", errors="replace")
        except Exception:
            pass

    return content


def parse_attached_file(
    filename: str,
    content: str | bytes,
    feature_names: list[str] | None = None,
) -> FileParseResult:
    """Main entrypoint to parse and profile any supported attached file.

    Args:
        filename: Name of the uploaded file.
        content: Raw string or bytes content.
        feature_names: Optional reference feature names (e.g. 2,000 DetectAI genes).

    Returns:
        FileParseResult containing summary text and structured metadata.
    """
    # Quick size check on raw input if bytes or str
    raw_size = len(content) if isinstance(content, (str, bytes)) else 0
    if raw_size > MAX_FILE_SIZE_BYTES * 2:  # Allow headroom for base64 wrapping
        return FileParseResult(
            filename=filename,
            file_type="unknown",
            file_size_bytes=raw_size,
            summary_text=f"File exceeds maximum allowed size of 5 MB ({raw_size / (1024 * 1024):.2f} MB).",
            warnings=["File size exceeds 5 MB limit. Parsing was skipped."],
        )

    raw_text = _decode_content(content)
    size_bytes = len(raw_text.encode("utf-8"))

    if size_bytes > MAX_FILE_SIZE_BYTES:
        return FileParseResult(
            filename=filename,
            file_type="unknown",
            file_size_bytes=size_bytes,
            summary_text=f"File exceeds maximum allowed size of 5 MB ({size_bytes / (1024 * 1024):.2f} MB).",
            warnings=["File size exceeds 5 MB limit. Parsing was skipped."],
        )

    if not raw_text.strip():
        return FileParseResult(
            filename=filename,
            file_type="unknown",
            file_size_bytes=size_bytes,
            summary_text="Empty file provided.",
            warnings=["The uploaded file is empty."],
        )

    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext in ("csv", "tsv"):
        return _parse_tabular_expression(filename, raw_text, ext, feature_names)
    elif ext in ("txt", "md"):
        return _parse_clinical_text(filename, raw_text)
    elif ext == "json":
        return _parse_genomic_json(filename, raw_text, feature_names)
    else:
        # Fallback based on content sniffing
        if raw_text.strip().startswith(("{", "[")):
            return _parse_genomic_json(filename, raw_text, feature_names)
        elif "\n" in raw_text and ("," in raw_text or "\t" in raw_text):
            return _parse_tabular_expression(filename, raw_text, "csv" if "," in raw_text else "tsv", feature_names)
        else:
            return _parse_clinical_text(filename, raw_text)


def _parse_tabular_expression(
    filename: str,
    text: str,
    ext: str,
    feature_names: list[str] | None = None,
) -> FileParseResult:
    """Parse CSV/TSV expression tables, detect orientation, and compute statistics."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]

    if not lines:
        return FileParseResult(
            filename=filename,
            file_type="tabular",
            file_size_bytes=len(text.encode("utf-8")),
            summary_text="Empty tabular file.",
            warnings=["No tabular data rows found."],
        )

    # Delimiter detection
    first_line = lines[0]
    if ext == "tsv" or "\t" in first_line:
        delimiter = "\t"
    elif ";" in first_line and "," not in first_line:
        delimiter = ";"
    else:
        delimiter = ","

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows: list[list[str]] = [[c.strip() for c in row if c is not None] for row in reader if row]

    if not rows:
        return FileParseResult(
            filename=filename,
            file_type="tabular",
            file_size_bytes=len(text.encode("utf-8")),
            summary_text="Unable to read tabular rows.",
            warnings=["Tabular reader produced 0 rows."],
        )

    # Sniff whether row 0 is a header row or numeric data
    row0_numeric = sum(1 for c in rows[0] if _is_float(c))
    row0_ratio = (row0_numeric / len(rows[0])) if rows[0] else 0.0

    if row0_ratio < 0.5:
        header = [c for c in rows[0]]
        data_rows = rows[1:]
        has_header = True
    else:
        header = []
        data_rows = rows
        has_header = False

    feature_set = set(feature_names or [])
    header_overlap = sum(1 for c in header if c in feature_set) if feature_set else 0
    first_col_overlap = sum(1 for r in data_rows if r and r[0] in feature_set) if feature_set else 0

    gene_values: dict[str, float] = {}
    extracted_vector: list[float] | None = None
    warnings: list[str] = []

    if header_overlap > 0 and header_overlap >= first_col_overlap:
        # Mode A: Header contains gene names
        if data_rows:
            v_row = data_rows[0]
            start_idx = 1 if len(v_row) == len(header) and not _is_float(v_row[0]) else 0
            for g, val_str in zip(header[start_idx:], v_row[start_idx:]):
                if _is_float(val_str):
                    gene_values[g] = float(val_str)
    elif first_col_overlap > 0:
        # Mode B: First col contains gene names
        for r in data_rows:
            if len(r) >= 2 and _is_float(r[1]):
                gene_values[r[0]] = float(r[1])
    else:
        # Heuristic detection when no known feature list matches
        first_h = header[0].lower() if header else ""
        gene_header_keywords = ("gene", "genes", "gene_id", "gene_name", "symbol", "feature", "probe")
        sample_header_keywords = ("sample", "sample_id", "patient", "patient_id", "case", "barcode")

        is_gene_col = any(k in first_h for k in gene_header_keywords)
        is_sample_col = any(k in first_h for k in sample_header_keywords)

        col0_non_num = sum(1 for r in data_rows if r and not _is_float(r[0]))
        col1_num = sum(1 for r in data_rows if len(r) >= 2 and _is_float(r[1]))
        total_data = len(data_rows)

        if is_sample_col and data_rows:
            # Header has gene names, column 0 has sample ID
            v_row = data_rows[0]
            for g, v in zip(header[1:], v_row[1:]):
                if _is_float(v):
                    gene_values[g] = float(v)
        elif is_gene_col:
            # First column is explicitly genes
            for r in data_rows:
                if len(r) >= 2 and _is_float(r[1]):
                    gene_values[r[0]] = float(r[1])
        elif total_data > 1 and (col0_non_num / total_data >= 0.6) and (col1_num / total_data >= 0.6):
            # Multiple rows with non-numeric in col 0 and numeric in col 1 -> genes in rows
            for r in data_rows:
                if len(r) >= 2 and _is_float(r[1]):
                    gene_values[r[0]] = float(r[1])
        elif total_data == 1 and len(data_rows[0]) == 2 and not _is_float(data_rows[0][0]) and _is_float(data_rows[0][1]):
            # 1 row, 2 cols: [gene_name, value]
            gene_values[data_rows[0][0]] = float(data_rows[0][1])
        elif data_rows and sum(1 for c in data_rows[0] if _is_float(c)) >= len(data_rows[0]) * 0.5:
            # Genes in header columns
            v_row = data_rows[0]
            start_idx = 1 if len(v_row) == len(header) and not _is_float(v_row[0]) else 0
            for idx, c in enumerate(v_row[start_idx:]):
                if _is_float(c):
                    col_idx = idx + start_idx
                    g_name = header[col_idx] if col_idx < len(header) else f"feature_{col_idx}"
                    gene_values[g_name] = float(c)
        else:
            for r in rows:
                if len(r) >= 2 and _is_float(r[1]):
                    gene_values[r[0]] = float(r[1])

    if not gene_values:
        # Try extracting all numeric floats from the file
        all_floats = []
        for r in rows:
            for c in r:
                if _is_float(c):
                    all_floats.append(float(c))
        for idx, val in enumerate(all_floats):
            gene_values[f"var_{idx}"] = val

    values = list(gene_values.values())
    if not values:
        return FileParseResult(
            filename=filename,
            file_type="tabular",
            file_size_bytes=len(text.encode("utf-8")),
            summary_text="Tabular file contains no numeric expression values.",
            warnings=["No numeric values could be extracted."],
        )

    # Compute statistical profile
    count = len(values)
    mean_val = sum(values) / count
    variance = sum((x - mean_val) ** 2 for x in values) / count if count > 1 else 0.0
    std_val = variance ** 0.5
    min_val = min(values)
    max_val = max(values)
    zero_count = sum(1 for x in values if abs(x) < 1e-6)
    zero_pct = (zero_count / count) * 100.0

    # Top & lowest expressed genes
    sorted_genes = sorted(gene_values.items(), key=lambda item: item[1], reverse=True)
    top_expressed = [{"gene": g, "value": round(v, 4)} for g, v in sorted_genes[:5]]
    lowest_expressed = [{"gene": g, "value": round(v, 4)} for g, v in sorted_genes[-5:]]

    # Feature matching against DetectAI
    matched_genes = [g for g in gene_values if g in feature_set] if feature_set else []

    # If full vector can be mapped
    if feature_names and len(matched_genes) == len(feature_names):
        extracted_vector = [gene_values[g] for g in feature_names]

    stats = {
        "num_genes": count,
        "mean_expression": round(mean_val, 4),
        "std_expression": round(std_val, 4),
        "min_expression": round(min_val, 4),
        "max_expression": round(max_val, 4),
        "zero_count": zero_count,
        "zero_percentage": round(zero_pct, 2),
        "matched_features_count": len(matched_genes),
        "top_expressed": top_expressed,
        "lowest_expressed": lowest_expressed,
    }

    summary_text = (
        f"Tabular Expression Profile ({count:,} features analyzed). "
        f"Mean: {mean_val:.2f}, Std: {std_val:.2f}, Range: [{min_val:.2f}, {max_val:.2f}], "
        f"Zero Fraction: {zero_pct:.1f}%. "
        f"Matched {len(matched_genes)}/{len(feature_names or [])} model features. "
        f"Highest: {', '.join([f'{g['gene']} ({g['value']})' for g in top_expressed[:3]])}."
    )

    return FileParseResult(
        filename=filename,
        file_type="tabular",
        file_size_bytes=len(text.encode("utf-8")),
        summary_text=summary_text,
        tabular_stats=stats,
        matched_genes=matched_genes,
        extracted_vector=extracted_vector,
        warnings=warnings,
    )


def _extract_ihc_marker(marker_pattern: str, text: str) -> str | None:
    """Extract IHC marker status, percentage, or intensity."""
    pattern = (
        rf"\b(?:{marker_pattern})\s*(?:[:=]|\s+is\s+)?\s*(?:-\s*)?"
        rf"(\(\+\)|\(\-\)|positive|negative|pos|neg|\+|\-|[0-3]\+|equivocal|intact|retained|loss|[0-9]{{1,3}}%)(?!\w)"
    )
    m = re.search(pattern, text, re.IGNORECASE)
    if m and m.group(1):
        raw = m.group(1).strip()
        raw_lower = raw.lower()
        if raw_lower in ("positive", "pos", "+", "(+)"):
            return "positive"
        elif raw_lower in ("negative", "neg", "-", "(-)"):
            return "negative"
        return raw
    return None


def _parse_clinical_text(filename: str, text: str) -> FileParseResult:
    """Extract clinical entities: TNM stage, histology, demographics, and IHC markers."""
    entities: dict[str, Any] = {}
    warnings: list[str] = []

    # 1. TNM Staging & Stage (supports pathological [p], clinical [c], Tis, and Tx/Nx/Mx)
    tnm_match = re.search(
        r"\b([pc]?(?:T[0-4x]|Tis)[a-c]?\s*[pc]?N[0-3x][a-c]?\s*[pc]?M[0-1x][a-c]?)\b",
        text,
        re.IGNORECASE,
    )
    if tnm_match:
        entities["tnm_stage"] = tnm_match.group(1).upper().replace(" ", "")

    stage_group = re.search(
        r"\b(?:clinical|pathologic(?:al)?)?\s*stage\s*[:=]?\s*(0|I{1,3}[A-C]?|IV[A-C]?|[1-4][A-C]?)\b",
        text,
        re.IGNORECASE,
    )
    if stage_group:
        entities["clinical_stage"] = f"Stage {stage_group.group(1).upper()}"

    # 2. Demographics: Age and Sex (supports "Age: 58" and "62yo")
    age_match = re.search(
        r"\b(?:age\s*[:=]?\s*(\d{1,3})|(\d{1,3})\s*(?:yo|year[s]?\s*old|-year-old|years of age))\b",
        text,
        re.IGNORECASE,
    )
    if age_match:
        entities["age"] = int(age_match.group(1) or age_match.group(2))

    sex_match = re.search(r"\b(female|male|woman|man)\b", text, re.IGNORECASE)
    if sex_match:
        val = sex_match.group(1).lower()
        entities["sex"] = "Female" if val in ("female", "woman") else "Male"

    # 3. Histological Type
    histology_patterns = [
        r"\binvasive ductal carcinoma\b",
        r"\binfiltrating ductal carcinoma\b",
        r"\binvasive lobular carcinoma\b",
        r"\bclear cell renal cell carcinoma\b",
        r"\bclear cell carcinoma\b",
        r"\bcolon adenocarcinoma\b",
        r"\blung adenocarcinoma\b",
        r"\bprostate adenocarcinoma\b",
        r"\brenal cell carcinoma\b",
        r"\bacinar adenocarcinoma\b",
        r"\bpapillary carcinoma\b",
        r"\bsquamous cell carcinoma\b",
    ]
    detected_histology = []
    for pat in histology_patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            detected_histology.append(match.group(0).title())
    if detected_histology:
        entities["histology"] = list(set(detected_histology))

    # 4. IHC Biomarkers
    ihc_findings: dict[str, str] = {}

    # Breast markers
    er_val = _extract_ihc_marker(r"ER|ESR1", text)
    if er_val:
        ihc_findings["ER"] = er_val

    pr_val = _extract_ihc_marker(r"PR|PGR", text)
    if pr_val:
        ihc_findings["PR"] = pr_val

    her2_val = _extract_ihc_marker(r"HER2(?:/neu)?|ERBB2", text)
    if her2_val:
        ihc_findings["HER2"] = her2_val

    ki67_val = _extract_ihc_marker(r"Ki-?67|MKI67", text)
    if ki67_val:
        ihc_findings["Ki-67"] = ki67_val

    # Kidney markers
    for k_key, k_pat in [("CD10", r"CD10|MME"), ("PAX-8", r"Pax-8|PAX8"), ("CA9", r"CA9|CA-IX"), ("VIMENTIN", r"Vimentin")]:
        val = _extract_ihc_marker(k_pat, text)
        if val:
            ihc_findings[k_key] = val

    # Colon markers
    for c_key, c_pat in [("CDX2", r"CDX2"), ("CK20", r"CK20|KRT20"), ("CK7", r"CK7|KRT7"), ("MLH1", r"MLH1"), ("MSH2", r"MSH2"), ("MSH6", r"MSH6"), ("PMS2", r"PMS2")]:
        val = _extract_ihc_marker(c_pat, text)
        if val:
            ihc_findings[c_key] = val

    # Lung markers
    for l_key, l_pat in [("TTF-1", r"TTF-1|TTF1|NKX2-1"), ("NAPSIN-A", r"Napsin-A|NAPSA"), ("CK7", r"CK7|KRT7"), ("P40", r"p40|DeltaNp63")]:
        val = _extract_ihc_marker(l_pat, text)
        if val and l_key not in ihc_findings:
            ihc_findings[l_key] = val

    # Prostate markers
    for p_key, p_pat in [("PSA", r"PSA|KLK3"), ("PSMA", r"PSMA|FOLH1"), ("NKX3.1", r"NKX3\.1|NKX3-1"), ("AMACR", r"AMACR|p504s")]:
        val = _extract_ihc_marker(p_pat, text)
        if val and p_key not in ihc_findings:
            ihc_findings[p_key] = val

    gleason_match = re.search(r"\bgleason\s*(?:score)?\s*[:=]?\s*(\d\s*\+\s*\d|\d{1,2})\b", text, re.IGNORECASE)
    if gleason_match:
        entities["gleason_score"] = gleason_match.group(1).replace(" ", "")

    if ihc_findings:
        entities["ihc_markers"] = ihc_findings

    # 5. Prior Therapies
    therapies = []
    for therapy in ("chemotherapy", "radiation", "mastectomy", "nephrectomy", "colectomy", "tamoxifen", "immunotherapy"):
        if re.search(rf"\b{therapy}\b", text, re.IGNORECASE):
            therapies.append(therapy.capitalize())
    if therapies:
        entities["prior_treatments"] = therapies

    # Generate summary text
    parts = []
    if "histology" in entities:
        parts.append(f"Histology: {', '.join(entities['histology'])}")
    if "tnm_stage" in entities or "clinical_stage" in entities:
        stage_desc = entities.get("clinical_stage", "")
        if "tnm_stage" in entities:
            stage_desc += f" ({entities['tnm_stage']})" if stage_desc else entities["tnm_stage"]
        parts.append(f"Staging: {stage_desc}")
    if "age" in entities or "sex" in entities:
        demog = f"Patient: {entities.get('age', 'N/A')} y/o {entities.get('sex', '')}".strip()
        parts.append(demog)
    if "ihc_markers" in entities:
        ihc_str = ", ".join([f"{k}: {v}" for k, v in entities["ihc_markers"].items()])
        parts.append(f"IHC: {ihc_str}")
    if "gleason_score" in entities:
        parts.append(f"Gleason: {entities['gleason_score']}")

    summary_text = (
        f"Clinical Pathology Note. " + " | ".join(parts)
        if parts
        else f"Clinical text note ({len(text)} characters). Extracted general narrative context."
    )

    return FileParseResult(
        filename=filename,
        file_type="clinical_note",
        file_size_bytes=len(text.encode("utf-8")),
        summary_text=summary_text,
        detected_entities=entities,
        warnings=warnings,
    )


def _parse_genomic_json(
    filename: str,
    text: str,
    feature_names: list[str] | None = None,
) -> FileParseResult:
    """Parse JSON containing gene expression dictionaries, mutations, or clinical metadata."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as err:
        return FileParseResult(
            filename=filename,
            file_type="genomic_json",
            file_size_bytes=len(text.encode("utf-8")),
            summary_text=f"Failed to parse JSON file: {err}",
            warnings=[f"Malformed JSON syntax: {err}"],
        )

    entities: dict[str, Any] = {}
    matched_genes: list[str] = []
    extracted_vector: list[float] | None = None
    tabular_stats: dict[str, Any] | None = None
    summary_parts: list[str] = []

    feature_set = set(feature_names or [])

    def _calc_stats(expr_map: dict[str, float]) -> tuple[dict[str, Any], list[str], list[float] | None]:
        vals = list(expr_map.values())
        cnt = len(vals)
        mean_v = sum(vals) / cnt if cnt else 0.0
        var_v = sum((x - mean_v) ** 2 for x in vals) / cnt if cnt > 1 else 0.0
        std_v = var_v ** 0.5
        min_v = min(vals) if vals else 0.0
        max_v = max(vals) if vals else 0.0
        z_cnt = sum(1 for x in vals if abs(x) < 1e-6)
        z_pct = (z_cnt / cnt) * 100.0 if cnt else 0.0

        matched = [g for g in expr_map if g in feature_set]
        vec = None
        if feature_names and len(matched) == len(feature_names):
            vec = [float(expr_map[g]) for g in feature_names]

        sorted_g = sorted(expr_map.items(), key=lambda item: item[1], reverse=True)
        top_exp = [{"gene": g, "value": round(v, 4)} for g, v in sorted_g[:5]]
        low_exp = [{"gene": g, "value": round(v, 4)} for g, v in sorted_g[-5:]]

        st = {
            "num_genes": cnt,
            "mean_expression": round(mean_v, 4),
            "std_expression": round(std_v, 4),
            "min_expression": round(min_v, 4),
            "max_expression": round(max_v, 4),
            "zero_count": z_cnt,
            "zero_percentage": round(z_pct, 2),
            "matched_features_count": len(matched),
            "top_expressed": top_exp,
            "lowest_expressed": low_exp,
        }
        return st, matched, vec

    if isinstance(data, dict):
        # Check for nested gene expression dictionary or vector
        nested_expr = data.get("gene_values") or data.get("expression") or data.get("features")
        if isinstance(nested_expr, dict) and any(isinstance(v, (int, float)) for v in nested_expr.values()):
            expr_dict = {str(k): float(v) for k, v in nested_expr.items() if _is_float(v)}
            tabular_stats, matched_genes, extracted_vector = _calc_stats(expr_dict)
            summary_parts.append(
                f"JSON Expression dictionary ({len(expr_dict):,} genes, Mean: {tabular_stats['mean_expression']:.2f}). "
                f"Matched {len(matched_genes)}/{len(feature_names or [])} model features."
            )
        elif isinstance(nested_expr, list) and nested_expr and all(_is_float(v) for v in nested_expr[:50]):
            float_vals = [float(v) for v in nested_expr if _is_float(v)]
            if feature_names and len(float_vals) == len(feature_names):
                expr_dict = {g: float_vals[i] for i, g in enumerate(feature_names)}
                extracted_vector = float_vals
            else:
                expr_dict = {f"gene_{i}": v for i, v in enumerate(float_vals)}
                if len(float_vals) in (len(feature_names or []), 20531):
                    extracted_vector = float_vals
            tabular_stats, matched_genes, _ = _calc_stats(expr_dict)
            summary_parts.append(
                f"JSON Expression vector ({len(float_vals):,} features, Mean: {tabular_stats['mean_expression']:.2f})."
            )
        else:
            # Check if top-level is directly a gene expression dictionary
            numeric_count = sum(1 for v in data.values() if isinstance(v, (int, float)))
            if numeric_count > 0 and numeric_count >= len(data) * 0.7:
                expr_dict = {str(k): float(v) for k, v in data.items() if _is_float(v)}
                tabular_stats, matched_genes, extracted_vector = _calc_stats(expr_dict)
                summary_parts.append(
                    f"JSON Gene Expression dictionary with {len(expr_dict):,} genes (Mean: {tabular_stats['mean_expression']:.2f}). "
                    f"Matched {len(matched_genes)} model features."
                )

        # Extract clinical / metadata fields
        if "mutations" in data or "variants" in data:
            muts = data.get("mutations") or data.get("variants")
            entities["variants"] = muts
            summary_parts.append(f"Variants: {muts}")
        if "stage" in data:
            entities["stage"] = data["stage"]
            summary_parts.append(f"Stage: {data['stage']}")
        if "ihc" in data or "ihc_markers" in data:
            ihc_data = data.get("ihc") or data.get("ihc_markers")
            entities["ihc_markers"] = ihc_data
            summary_parts.append(f"IHC: {ihc_data}")
        if "patient_id" in data or "sample_id" in data:
            entities["sample_id"] = data.get("patient_id") or data.get("sample_id")
        if "histology" in data:
            entities["histology"] = data["histology"]
        if "age" in data:
            entities["age"] = data["age"]
        if "sex" in data or "gender" in data:
            entities["sex"] = data.get("sex") or data.get("gender")

        if not summary_parts:
            summary_parts.append(f"JSON Record with keys: {', '.join(list(data.keys())[:6])}")

    elif isinstance(data, list):
        if data and all(_is_float(v) for v in data[:50]):
            float_vals = [float(v) for v in data if _is_float(v)]
            expr_dict = {f"gene_{i}": v for i, v in enumerate(float_vals)}
            tabular_stats, matched_genes, _ = _calc_stats(expr_dict)
            if len(float_vals) in (len(feature_names or []), 20531):
                extracted_vector = float_vals
            summary_parts.append(
                f"JSON Expression array ({len(float_vals):,} features, Mean: {tabular_stats['mean_expression']:.2f})."
            )
        else:
            summary_parts.append(f"JSON Array containing {len(data)} items.")
            if data and isinstance(data[0], dict):
                summary_parts.append(f"Item fields: {', '.join(list(data[0].keys())[:5])}")

    summary_text = " | ".join(summary_parts) if summary_parts else "JSON genomic record."

    return FileParseResult(
        filename=filename,
        file_type="genomic_json",
        file_size_bytes=len(text.encode("utf-8")),
        summary_text=summary_text,
        detected_entities=entities,
        tabular_stats=tabular_stats,
        matched_genes=matched_genes,
        extracted_vector=extracted_vector,
    )


def _is_float(val: Any) -> bool:
    """Check if value can be converted to float."""
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False
