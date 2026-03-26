from __future__ import annotations

import json
from importlib import import_module
from io import BytesIO
from pathlib import Path
import tempfile
from typing import Any, Callable
import zipfile

import cloudpickle
import equinox as eqx


class LoadedModel:
    """Container describing a loaded model artifact.

    `parameter_tree` is used for architecture introspection/visualization.
    `run_fn` is optional and should accept one input tensor-like argument.
    """

    def __init__(
        self,
        *,
        framework: str,
        model: Any,
        parameter_tree: Any,
        run_fn: Callable[[Any], Any] | None,
    ) -> None:
        self.framework = framework
        self.model = model
        self.parameter_tree = parameter_tree
        self.run_fn = run_fn


def _as_loaded_eqx(model: Any) -> LoadedModel:
    return LoadedModel(
        framework="equinox",
        model=model,
        parameter_tree=model,
        run_fn=lambda x: model(x),
    )


def _import_flax_linen_module_class():
    try:
        from flax import linen as nn  # type: ignore

        return nn.Module
    except Exception:
        return None


def _is_flax_module_instance(obj: Any) -> bool:
    module_cls = _import_flax_linen_module_class()
    if module_cls is None:
        return False
    return isinstance(obj, module_cls)


def _normalize_flax_variables(variables: Any) -> Any:
    """Return a mutable mapping-like variables tree when possible."""
    try:
        # Flax FrozenDict exposes `.unfreeze()`.
        if hasattr(variables, "unfreeze") and callable(variables.unfreeze):
            return variables.unfreeze()
    except Exception:
        pass
    return variables


def _flax_parameter_tree(variables: Any) -> Any:
    norm = _normalize_flax_variables(variables)
    if isinstance(norm, dict) and "params" in norm:
        return norm["params"]
    return norm


def _build_flax_run_fn(module: Any, variables: Any) -> Callable[[Any], Any]:
    """Build a tolerant apply wrapper for Flax modules.

    Tries a few common calling conventions without hard-coding architecture.
    """

    norm_vars = _normalize_flax_variables(variables)

    def _runner(x: Any):
        attempts: list[tuple[Any, dict[str, Any]]] = []
        attempts.append((norm_vars, {}))
        if isinstance(norm_vars, dict) and "params" not in norm_vars:
            attempts.append(({"params": norm_vars}, {}))
        attempts.append((norm_vars, {"train": False}))
        if isinstance(norm_vars, dict) and "params" not in norm_vars:
            attempts.append(({"params": norm_vars}, {"train": False}))

        last_exc: Exception | None = None
        for vars_arg, kwargs in attempts:
            try:
                return module.apply(vars_arg, x, **kwargs)
            except Exception as exc:  # pragma: no cover - best-effort fallbacks
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc
        return module.apply(norm_vars, x)

    return _runner


def _as_loaded_flax(module: Any, variables: Any) -> LoadedModel:
    return LoadedModel(
        framework="flax",
        model={"module": module, "variables": variables},
        parameter_tree=_flax_parameter_tree(variables),
        run_fn=_build_flax_run_fn(module, variables),
    )


def _extract_flax_from_obj(obj: Any) -> LoadedModel | None:
    """Try to coerce pickle payloads into a Flax loaded model."""

    # Common tuple shape: (module, variables)
    if isinstance(obj, tuple) and len(obj) == 2:
        module, variables = obj
        if _is_flax_module_instance(module):
            return _as_loaded_flax(module, variables)

    # Common dict shape with explicit keys.
    if isinstance(obj, dict):
        module = obj.get("module") or obj.get("model")
        variables = obj.get("variables")
        if variables is None:
            # Allow params-only payloads.
            variables = obj.get("params")
        if module is not None and variables is not None and _is_flax_module_instance(module):
            return _as_loaded_flax(module, variables)

    return None


class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded from the supplied specification."""


EQX_BUNDLE_MANIFEST = "manifest.json"
EQX_BUNDLE_WEIGHTS = "model.eqx"
EQX_BUNDLE_VERSION = 1


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

    flax_loaded = _extract_flax_from_obj(model)
    if flax_loaded is not None:
        return flax_loaded

    return _as_loaded_eqx(model)


def save_model_bundle(
    model: Any,
    bundle_path: str | Path,
    *,
    factory_spec: str,
    factory_kwargs: dict[str, Any] | None = None,
) -> Path:
    """Save a model as a single-file Equinox bundle.

    The bundle contains:
    - `manifest.json`: factory metadata needed to rebuild the module skeleton
    - `model.eqx`: leaf weights serialized via `eqx.tree_serialise_leaves`
    """
    bundle_path = Path(bundle_path).expanduser().resolve()
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format": "eqxview.eqxbundle",
        "version": EQX_BUNDLE_VERSION,
        "factory": factory_spec,
        "factory_kwargs": factory_kwargs or {},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        weights_path = tmpdir_path / EQX_BUNDLE_WEIGHTS
        eqx.tree_serialise_leaves(str(weights_path), model)

        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(EQX_BUNDLE_MANIFEST, json.dumps(manifest, indent=2, sort_keys=True))
            zf.write(weights_path, arcname=EQX_BUNDLE_WEIGHTS)

    return bundle_path


def load_model_from_eqx_bundle_bytes(content: bytes):
    """Load a model from a single-file Equinox bundle."""
    if not content:
        raise ModelLoadError("Uploaded file is empty.")

    try:
        with zipfile.ZipFile(BytesIO(content)) as zf:
            manifest = json.loads(zf.read(EQX_BUNDLE_MANIFEST).decode("utf-8"))
            weights = zf.read(EQX_BUNDLE_WEIGHTS)
    except KeyError as exc:
        raise ModelLoadError(f"Invalid eqxbundle: missing entry {exc}") from exc
    except zipfile.BadZipFile as exc:
        raise ModelLoadError(f"Invalid eqxbundle zip file: {exc}") from exc
    except Exception as exc:
        raise ModelLoadError(f"Failed to read eqxbundle: {exc}") from exc

    if manifest.get("format") != "eqxview.eqxbundle":
        raise ModelLoadError("Unsupported eqxbundle format.")
    if manifest.get("version") != EQX_BUNDLE_VERSION:
        raise ModelLoadError(
            f"Unsupported eqxbundle version: {manifest.get('version')}"
        )

    factory_spec = manifest.get("factory")
    if not isinstance(factory_spec, str) or not factory_spec:
        raise ModelLoadError("eqxbundle manifest is missing a valid 'factory' field.")

    factory_kwargs = manifest.get("factory_kwargs")
    if factory_kwargs is not None and not isinstance(factory_kwargs, dict):
        raise ModelLoadError("eqxbundle manifest field 'factory_kwargs' must be a dict.")

    try:
        model = load_model(factory_spec=factory_spec, factory_kwargs=factory_kwargs or {})
        with tempfile.NamedTemporaryFile(suffix=".eqx") as tmp:
            tmp.write(weights)
            tmp.flush()
            model = eqx.tree_deserialise_leaves(tmp.name, model)
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError(f"Failed to restore model from eqxbundle: {exc}") from exc

    return _as_loaded_eqx(model)


def load_uploaded_model(filename: str, content: bytes):
    """Load a model from any supported uploaded format."""
    lowered = filename.lower()
    if lowered.endswith((".pkl", ".pickle", ".cloudpickle")):
        return load_model_from_pickle_bytes(content)
    if lowered.endswith(".eqxbundle"):
        return load_model_from_eqx_bundle_bytes(content)

    raise ModelLoadError(
        "Unsupported file type. Use .eqxbundle, .pkl, .pickle, or .cloudpickle."
    )
