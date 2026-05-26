import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch


def hash_feature_names(feature_names: list[str]) -> str:
    joined = "\n".join(feature_names).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def environment_snapshot() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
    }


def write_json_manifest(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    manifest = dict(payload)
    manifest.setdefault("created_at_utc", datetime.now(timezone.utc).isoformat())
    manifest.setdefault("run_id", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
