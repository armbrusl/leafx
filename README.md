# leafx:  a local browser visualizer for Equinox and Flax model hierarchies.

![leafx is a local browser visualizer for Equinox and Flax model hierarchies.
](leafx_ex.png)

Current scope:
- load a serialized model by dragging a file into the browser window
- inspect an operation-style graph left-to-right
- render tensor-carrying edges (including parameter edges such as weight/bias)
- show summary stats (parameter count, bytes, leaf count)
- run the loaded model on a dropped `.npy` INPUT tensor and inspect OUTPUT shape/values

## Quick start

1. Activate your environment:

```bash
cd /home/users/armbrust/projects/leafx
source .venv/bin/activate
```

2. Sync dependencies with uv:

```bash
uv sync
```

3. Start the local server:

```bash
uv run uvicorn leafx.app:app --reload --host 127.0.0.1 --port 8000
```

4. Open in browser:

```text
http://127.0.0.1:8000
```

## Use your own model

Drag one file into the browser window.

Supported file extensions:
- .eqxbundle
- .pkl
- .pickle
- .cloudpickle

Pickle-based uploads can contain either:
- an Equinox model object, or
- a Flax payload such as `(module, variables)` or a dict with `module` + `variables`/`params`.

Recommended format right now: `.eqxbundle` containing factory metadata plus
`eqx.tree_serialise_leaves` weights.

To generate built-in example files (including Flax payloads):

```bash
python example_models.py
```

This writes Equinox bundles plus Flax `.cloudpickle` payloads (stored as
`(module, variables)` tuples) that can be dragged into the UI.

Example exporter:

```python
import jax.random as jr

from example_models import make_mlp
from leafx.model_loading import save_model_bundle

model = make_mlp(in_size=32, out_size=4, width=64, depth=3, seed=0)
save_model_bundle(
	model,
	"model.eqxbundle",
	factory_spec="example_models:make_mlp",
	factory_kwargs={"in_size": 32, "out_size": 4, "width": 64, "depth": 3, "seed": 0},
)
```

Security note: pickle-based formats still execute Python code on load. Prefer `.eqxbundle`
when possible, and keep all model loading local to trusted code.

## Run inference from browser input

After loading a model, drop a `.npy` tensor directly onto the `INPUT` node.
leafx will run a forward pass on the loaded model and populate the `OUTPUT` node
with inferred output shape and a heatmap preview of output values.

## Notes for large models

- operation nodes are laid out left-to-right by depth
- tensor labels are drawn on edges; parameter edges are highlighted
- use browser zoom/pan for navigation
- summary shows global parameter and memory size totals

## Next extensions

- parameter-mask extraction and save/load
- per-subtree statistics (norms, sparsity, histograms)
- architecture editing and round-trip export
