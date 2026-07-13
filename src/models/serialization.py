"""
Automatic Serialization — save and load models with format detection,
compression, metadata embedding, and version tracking.

Supported formats
-----------------
- ``.joblib`` — Default. Fast for scikit-learn/XGBoost/LightGBM objects.
- ``.pkl`` / ``.pickle`` — Standard Python pickling.
- ``.json`` — Metadata-only export.
- ``.onnx`` — ONNX format (if onnxruntime is installed).

The serializer embeds a metadata header into the saved file so models
can be self-describing (type, version, feature count, etc.).
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────

_MODEL_DIR = Path("models")

# Format → (extension, priority)
_FORMATS: list[tuple[str, str, int]] = [
    ("joblib", ".joblib", 90),
    ("pickle", ".pkl", 50),
    ("pickle", ".pickle", 40),
    ("onnx", ".onnx", 30),
    ("json", ".json", 10),
]

# Format detectors (lowercased)
_FORMAT_MAP: dict[str, str] = {}
for fmt, ext, _prio in _FORMATS:
    _FORMAT_MAP[ext] = fmt


def detect_format(path: str | Path) -> str:
    """Detect serialization format from file extension.

    Parameters
    ----------
    path : str | Path
        File path.

    Returns
    -------
    str
        Format name (``joblib``, ``pickle``, ``json``, ``onnx``).

    Raises
    ------
    ValueError
        If the extension is not recognised.
    """
    ext = Path(path).suffix.lower()
    fmt = _FORMAT_MAP.get(ext)
    if fmt is None:
        raise ValueError(
            f"Unrecognised model file extension '{ext}'. "
            f"Supported: {', '.join(sorted(set(_FORMAT_MAP.values())))}"
        )
    return fmt


def resolve_save_path(
    model: BaseModel,
    path: str | None = None,
    fmt: str = "joblib",
) -> Path:
    """Resolve the output path for saving a model.

    Parameters
    ----------
    model : BaseModel
    path : str, optional
        Explicit path. Auto-generated if omitted.
    fmt : str
        Format name (default ``joblib``).

    Returns
    -------
    Path
    """
    if path is not None:
        return Path(path)

    ext = _get_extension(fmt)
    filename = f"{model.model_name}_v{model.model_version}{ext}"
    return _MODEL_DIR / filename


def _get_extension(fmt: str) -> str:
    """Get file extension for a format name."""
    for f, ext, _prio in _FORMATS:
        if f == fmt:
            return ext
    return ".joblib"


# ═══════════════════════════════════════════════════════════
#  Serializer
# ═══════════════════════════════════════════════════════════


class ModelSerializer:
    """Handles serialization and deserialization of models.

    Usage
    -----
    ::

        # Auto-detect format from path
        ModelSerializer.save(model, "models/my_model.joblib")
        loaded = ModelSerializer.load("models/my_model.joblib")

        # Automatic path + format
        path = ModelSerializer.save(model)
        loaded = ModelSerializer.load(path)
    """

    @staticmethod
    def save(
        model: BaseModel,
        path: str | None = None,
        fmt: str | None = None,
        compress: bool = True,
        include_metadata: bool = True,
        **kwargs: Any,
    ) -> str:
        """Save a model to disk.

        Parameters
        ----------
        model : BaseModel
            The model to save.
        path : str, optional
            Output path. Auto-generated from model name and version if omitted.
        fmt : str, optional
            Format override. Auto-detected from path extension if omitted.
        compress : bool
            Enable compression (default True). Only applies to joblib/pickle.
        include_metadata : bool
            Embed metadata into the saved file (default True).
        **kwargs
            Additional format-specific options.

        Returns
        -------
        str
            Path to the saved file.
        """
        # Determine format
        if path is not None:
            detected = detect_format(path)
            fmt = fmt or detected
        else:
            fmt = fmt or "joblib"

        # Resolve path
        out_path = resolve_save_path(model, path, fmt)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Build payload
        payload: Any = model
        if include_metadata:
            payload = _embed_metadata(model)

        # Dispatch
        serializers = {
            "joblib": lambda: _save_joblib(payload, out_path, compress),
            "pickle": lambda: _save_pickle(payload, out_path, compress),
            "json": lambda: _save_json(payload, out_path),
            "onnx": lambda: _save_onnx(payload, out_path),
        }

        serializer = serializers.get(fmt)
        if serializer is None:
            raise ValueError(f"Unsupported format: {fmt}")

        try:
            serializer()
        except ImportError as exc:
            raise ImportError(
                f"Format '{fmt}' requires extra dependencies: {exc}"
            )

        file_size = out_path.stat().st_size
        logger.info(
            "Saved %s v%s to %s (%.1f MB, %s)",
            model.model_name, model.model_version,
            out_path, file_size / (1024 * 1024), fmt,
        )
        return str(out_path)

    @staticmethod
    def load(path: str, **kwargs: Any) -> BaseModel:
        """Load a model from disk.

        Parameters
        ----------
        path : str
            Path to the saved model file.
        **kwargs
            Additional options (passed to format-specific loader).

        Returns
        -------
        BaseModel

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the format is unsupported.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")

        fmt = detect_format(path)

        loaders = {
            "joblib": lambda: _load_joblib(file_path),
            "pickle": lambda: _load_pickle(file_path),
            "json": lambda: _load_json(file_path),
            "onnx": lambda: _load_onnx(file_path),
        }

        loader = loaders.get(fmt)
        if loader is None:
            raise ValueError(f"Unsupported format: {fmt}")

        try:
            payload = loader()
        except ImportError as exc:
            raise ImportError(
                f"Format '{fmt}' requires extra dependencies: {exc}"
            )

        # Extract model from metadata wrapper
        if isinstance(payload, dict) and "model" in payload:
            model = payload["model"]
        else:
            model = payload

        if not isinstance(model, BaseModel):
            raise TypeError(
                f"Loaded object is not a BaseModel instance: "
                f"{type(model).__name__}"
            )

        logger.info(
            "Loaded %s v%s from %s",
            model.model_name, model.model_version, file_path,
        )
        return model


# ═══════════════════════════════════════════════════════════
#  Metadata embedding
# ═══════════════════════════════════════════════════════════


def _embed_metadata(model: BaseModel) -> dict[str, Any]:
    """Wrap a model in a metadata dict for self-describing files."""
    return {
        "model": model,
        "model_name": model.model_name,
        "model_type": model.model_type,
        "model_version": model.model_version,
        "n_features": model._n_features,
        "n_classes": model._n_classes,
        "fitted": model.fitted,
        "calibrated": model.calibrated,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }


# ═══════════════════════════════════════════════════════════
#  Format-specific serializers
# ═══════════════════════════════════════════════════════════


def _save_joblib(payload: Any, path: Path, compress: bool) -> None:
    """Save using joblib (fast for numpy/scikit-learn objects)."""
    import joblib
    joblib.dump(payload, path, compress=compress)


def _load_joblib(path: Path) -> Any:
    """Load using joblib."""
    import joblib
    return joblib.load(path)


def _save_pickle(payload: Any, path: Path, compress: bool) -> None:
    """Save using Python pickle."""
    protocol = pickle.HIGHEST_PROTOCOL
    if compress:
        import gzip
        with gzip.open(path, "wb") as f:
            pickle.dump(payload, f, protocol=protocol)
    else:
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=protocol)


def _load_pickle(path: Path) -> Any:
    """Load using Python pickle (auto-detects gzip compression)."""
    import gzip
    # gzip.open() auto-detects gzip magic bytes; falls through to
    # plain pickle for uncompressed files.
    with gzip.open(path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            pass
    # If gzip.open() didn't work, try plain open
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_json(payload: Any, path: Path) -> None:
    """Save metadata as JSON (model must be JSON-serializable)."""
    if hasattr(payload, "to_dict"):
        data = payload.to_dict()
    elif isinstance(payload, dict):
        data = {k: str(v) if not isinstance(v, (str, int, float, bool, list, dict)) else v
                for k, v in payload.items()}
    else:
        data = {"model_type": type(payload).__name__, "warning": "Not JSON-serializable"}
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _load_json(path: Path) -> Any:
    """Load metadata JSON and reconstruct a minimal wrapper."""
    with open(path) as f:
        data = json.load(f)
    # JSON can't store the actual model object, so return metadata dict
    return data


def _save_onnx(payload: Any, path: Path) -> None:
    """Convert and save as ONNX."""
    try:
        import skl2onnx
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        raise ImportError("ONNX export requires 'skl2onnx'. Install: pip install skl2onnx")

    model = payload.get("model", payload) if isinstance(payload, dict) else payload

    # Get the underlying sklearn model if wrapped
    if hasattr(model, "_model"):
        model = model._model

    n_features = payload.get("n_features", 1) if isinstance(payload, dict) else 1
    initial_types = [("float_input", FloatTensorType([None, n_features]))]

    try:
        onnx_model = convert_sklearn(model, initial_types=initial_types)
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())
    except Exception as exc:
        raise RuntimeError(f"ONNX conversion failed: {exc}")


def _load_onnx(path: Path) -> Any:
    """Load an ONNX model."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "ONNX loading requires 'onnxruntime'. Install: pip install onnxruntime"
        )

    session = ort.InferenceSession(str(path))
    return ONNXWrapper(session)


class ONNXWrapper:
    """Wrapper for ONNX runtime sessions to match BaseModel predict interface."""

    def __init__(self, session: Any) -> None:
        self.session = session
        self.input_name = session.get_inputs()[0].name

    def predict(self, X: Any) -> Any:
        import numpy as np
        X_arr = X.values if hasattr(X, "values") else X
        X_arr = X_arr.astype(np.float32)
        output = self.session.run(None, {self.input_name: X_arr})
        return np.argmax(output[0], axis=1)

    def predict_proba(self, X: Any) -> Any:
        import numpy as np
        X_arr = X.values if hasattr(X, "values") else X
        X_arr = X_arr.astype(np.float32)
        output = self.session.run(None, {self.input_name: X_arr})
        return output[0]
