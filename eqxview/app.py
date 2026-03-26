from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from eqxview._capture import Capture
from eqxview.introspection import build_operation_graph, iter_module_paths
from eqxview.model_loading import LoadedModel, ModelLoadError, load_uploaded_model
from eqxview.tensor_projection import flat_values_for_client, to_heatmap_2d


app = FastAPI(title="eqxview", version="0.1.0")
CURRENT_LOADED: LoadedModel | None = None

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")

def _pick_primary_array_leaf(output_value: Any) -> Any:
    leaves = jax.tree_util.tree_leaves(output_value)
    array_like = [leaf for leaf in leaves if hasattr(leaf, "shape") and hasattr(leaf, "dtype")]
    if not array_like:
        raise ValueError("Model output did not contain any array-like leaves.")
    return array_like[0]


def _activation_payload(value: Any) -> dict[str, Any] | None:
    try:
        leaf = _pick_primary_array_leaf(value)
    except Exception:
        return None

    try:
        arr = np.asarray(leaf)
    except Exception:
        return None

    return {
        "shape": [int(v) for v in arr.shape],
        "dtype": str(arr.dtype),
        "bytes": int(arr.size * arr.dtype.itemsize),
        "heatmap": to_heatmap_2d(arr, max_size=64),
        "flat_values": flat_values_for_client(arr),
        "stats": {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        },
    }


def _run_model_with_capture(
    model: Any, model_input: jnp.ndarray
) -> tuple[Any, dict[str, dict[str, Any]]]:
    """Run model forward pass with Capture-based activation recording.

    Patches ``eqx.nn.Embedding.__call__`` to handle batched integer indices
    automatically (via vmap), and patches ``Capture.__call__`` to record outputs
    for every ``eqxview.Capture``-wrapped submodule, keyed by module path.
    """

    # Map id(capture_instance) -> path string for every Capture in the model.
    captures_by_id: dict[int, str] = {
        id(mod): path
        for path, mod in iter_module_paths(model)
        if isinstance(mod, Capture)
    }
    activations: dict[str, dict[str, Any]] = {}

    original_capture_call = Capture.__call__
    original_emb_call = eqx.nn.Embedding.__call__

    def _capture_call(self, x, *args, **kwargs):
        out = original_capture_call(self, x, *args, **kwargs)
        path = captures_by_id.get(id(self))
        if path:
            payload = _activation_payload(out)
            if payload is not None:
                activations[path] = payload
        return out

    def _embedding_call(self, x, *args, **kwargs):
        idx = jnp.asarray(x)
        if not jnp.issubdtype(idx.dtype, jnp.integer):
            idx = jnp.rint(idx).astype(jnp.int32)
        else:
            idx = idx.astype(jnp.int32)
        idx = jnp.mod(idx, self.num_embeddings)
        # Flatten multi-dimensional index arrays to 1D so the downstream
        # module sees a plain [seq_len, embed_dim] tensor regardless of the
        # shape of the raw input (e.g. a 2D image used as token ids).
        if idx.ndim > 1:
            idx = idx.reshape(-1)
        if idx.ndim == 0:
            return original_emb_call(self, idx, *args, **kwargs)
        flat = idx.reshape(-1)
        mapped = jax.vmap(lambda t: original_emb_call(self, t))(flat)
        return mapped.reshape((*idx.shape, *mapped.shape[1:]))

    Capture.__call__ = _capture_call
    eqx.nn.Embedding.__call__ = _embedding_call
    try:
        return model(model_input), activations
    finally:
        Capture.__call__ = original_capture_call
        eqx.nn.Embedding.__call__ = original_emb_call


@app.post("/api/introspect-upload")
async def introspect_upload(file: UploadFile = File(...)) -> dict:
    global CURRENT_LOADED

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    allowed_suffixes = (".eqxbundle", ".pkl", ".pickle", ".cloudpickle")
    lowered = file.filename.lower()
    if not lowered.endswith(allowed_suffixes):
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. Use .eqxbundle or pickle formats "
                "(.pkl/.pickle/.cloudpickle)."
            ),
        )

    try:
        raw = await file.read()
        loaded = load_uploaded_model(file.filename, raw)
    except ModelLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    CURRENT_LOADED = loaded
    return build_operation_graph(loaded.model, parameter_tree=loaded.parameter_tree)


@app.post("/api/run-current-model")
async def run_current_model(input_file: UploadFile = File(...)) -> dict[str, Any]:
    if CURRENT_LOADED is None:
        raise HTTPException(status_code=400, detail="No model loaded yet.")

    if not input_file.filename or not input_file.filename.lower().endswith(".npy"):
        raise HTTPException(status_code=400, detail="Input must be a .npy file.")

    try:
        raw = await input_file.read()
        np_input = np.load(BytesIO(raw), allow_pickle=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read .npy input: {exc}") from exc

    model_input = jnp.asarray(np_input)
    try:
        activations: dict[str, dict[str, Any]] = {}
        if CURRENT_LOADED.framework == "equinox":
            output_value, activations = _run_model_with_capture(
                CURRENT_LOADED.model, model_input
            )
        elif CURRENT_LOADED.run_fn is not None:
            output_value = CURRENT_LOADED.run_fn(model_input)
        else:
            raise RuntimeError(
                f"Loaded {CURRENT_LOADED.framework} model is not runnable."
            )
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Model forward pass failed: {exc}"
        ) from exc

    try:
        output_leaf = _pick_primary_array_leaf(output_value)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Output parsing failed: {exc}") from exc

    output_np = np.asarray(output_leaf)
    heatmap = to_heatmap_2d(output_np, max_size=64)
    stats = {
        "min": float(np.min(output_np)),
        "max": float(np.max(output_np)),
        "mean": float(np.mean(output_np)),
        "std": float(np.std(output_np)),
    }

    return {
        "shape": [int(v) for v in output_np.shape],
        "dtype": str(output_np.dtype),
        "heatmap": heatmap,
        "flat_values": flat_values_for_client(output_np),
        "stats": stats,
        "activations": activations,
    }
