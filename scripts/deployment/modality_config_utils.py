"""Helpers for optional custom embodiment modality registration."""

from __future__ import annotations

from hashlib import sha1
import importlib.util
from pathlib import Path
import sys


def import_modality_config(path: str | Path | None) -> None:
    """Import a custom modality config module if a path is provided."""
    if path is None:
        return
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Modality config does not exist: {config_path}")

    resolved_path = config_path.resolve()
    module_hash = sha1(str(resolved_path).encode()).hexdigest()
    module_name = f"_gr00t_modality_config_{module_hash}"
    if module_name in sys.modules:
        return

    spec = importlib.util.spec_from_file_location(module_name, resolved_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load modality config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
