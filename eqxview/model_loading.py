from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import cloudpickle
import equinox as eqx


class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded from the supplied specification."""


def _resolve_factory(factory_spec: str):
    if ":" not in factory_spec:
        raise ModelLoadError(
            "Factory must be in the form 'module.submodule:function_name'."
        )

    module_name, function_name = factory_spec.split(":", maxsplit=1)
    try:
        module = import_module(module_name)
    except Exception as exc:  # pragma: no cover - import failure path
        raise ModelLoadError(f"Failed to import module '{module_name}': {exc}") from exc

    try:
        factory = getattr(module, function_name)
    except AttributeError as exc:  # pragma: no cover - attribute failure path
        raise ModelLoadError(
            f"Function '{function_name}' not found in module '{module_name}'."
        ) from exc

    if not callable(factory):
        raise ModelLoadError(
            f"Object '{function_name}' in module '{module_name}' is not callable."
        )
    return factory


def load_model(
    factory_spec: str,
    factory_kwargs: dict[str, Any] | None = None,
    checkpoint_path: str | None = None,
):
    """Instantiate an Equinox model and optionally load checkpointed leaves."""
    factory = _resolve_factory(factory_spec)
    kwargs = factory_kwargs or {}

    try:
        model = factory(**kwargs)
    except Exception as exc:
        raise ModelLoadError(f"Factory invocation failed: {exc}") from exc

    if checkpoint_path:
        ckpt = Path(checkpoint_path).expanduser().resolve()
        if not ckpt.exists():
            raise ModelLoadError(f"Checkpoint path does not exist: {ckpt}")
        try:
            model = eqx.tree_deserialise_leaves(str(ckpt), model)
        except Exception as exc:
            raise ModelLoadError(f"Checkpoint loading failed: {exc}") from exc

    return model


def load_model_from_pickle_bytes(content: bytes):
    """Load a pickled model object from raw bytes."""
    if not content:
        raise ModelLoadError("Uploaded file is empty.")

    try:
        model = cloudpickle.loads(content)
    except Exception as exc:
        raise ModelLoadError(f"Failed to deserialize cloudpickle file: {exc}") from exc

    return model
