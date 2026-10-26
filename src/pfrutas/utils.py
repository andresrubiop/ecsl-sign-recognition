from __future__ import annotations

import json
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np


def slugify(value: str) -> str:
    """Normaliza etiquetas para comparaciones y argumentos de línea de comandos."""
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.upper().strip()
    value = re.sub(r"[^A-Z0-9]+", "_", value)
    return value.strip("_")


def ascii_text(value: str) -> str:
    """Convierte texto Unicode a una forma que OpenCV pueda dibujar."""
    value = unicodedata.normalize("NFKD", value)
    return "".join(char for char in value if not unicodedata.combining(char))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"No se puede serializar {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Se solicitó CUDA, pero PyTorch no detecta una GPU compatible.")
        if requested == "mps" and not (
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ):
            raise RuntimeError("Se solicitó MPS, pero PyTorch no detecta un dispositivo compatible.")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_label(query: str, labels: list[str]) -> str:
    wanted = slugify(query)
    matches = [label for label in labels if slugify(label) == wanted]
    if not matches:
        available = ", ".join(labels)
        raise ValueError(f"Fruta desconocida: {query!r}. Opciones: {available}")
    return matches[0]
