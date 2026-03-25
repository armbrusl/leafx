from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from eqxview.introspection import build_operation_graph
from eqxview.model_loading import ModelLoadError, load_model_from_pickle_bytes


app = FastAPI(title="eqxview", version="0.1.0")

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.post("/api/introspect-upload")
async def introspect_upload(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    # Deserializing arbitrary pickles is code execution. Keep this local/trusted-only.
    allowed_suffixes = (".pkl", ".pickle", ".cloudpickle")
    lowered = file.filename.lower()
    if not lowered.endswith(allowed_suffixes):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use .pkl, .pickle, or .cloudpickle.",
        )

    try:
        raw = await file.read()
        model = load_model_from_pickle_bytes(raw)
    except ModelLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return build_operation_graph(model)
