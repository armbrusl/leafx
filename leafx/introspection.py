from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np

from leafx._capture import Capture as _Capture
from leafx.tensor_projection import flat_values_for_client, to_heatmap_2d


@dataclass
class TreeNode:
    name: str
    kind: str = "node"
    type_name: str | None = None
    shape: list[int] | None = None
    dtype: str | None = None
    param_count: int = 0
    bytes: int = 0
    children: dict[str, "TreeNode"] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "type": self.type_name,
            "shape": self.shape,
            "dtype": self.dtype,
            "param_count": self.param_count,
            "bytes": self.bytes,
            "children": [child.as_json() for child in self.children.values()],
        }


def _path_item_to_string(item: Any) -> str:
    for attr in ("name", "key", "idx"):
        if hasattr(item, attr):
            return str(getattr(item, attr))
    return str(item)


def _leaf_metrics(leaf: Any) -> tuple[int, int, list[int] | None, str | None]:
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
        shape = [int(v) for v in leaf.shape]
        count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        itemsize = np.dtype(leaf.dtype).itemsize
        size_bytes = count * int(itemsize)
        return count, size_bytes, shape, str(leaf.dtype)
    return 0, 0, None, None


def _downsample_for_heatmap(leaf: Any) -> list[list[float]] | None:
    """Return a 2D nested list suitable for heatmap rendering.

    • 1D → single-row 2D
    • 2D → as-is (downsampled if large)
    • 3D+ → folded into a 2D matrix so all values remain visible
    • Scalars / non-arrays → None
    """
    if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
        return None
    if len(leaf.shape) == 0:
        return None
    return to_heatmap_2d(leaf, max_size=5000)


def _flatten_with_path(
    model: Any,
) -> list[tuple[tuple, Any]]:
    """Recursively flatten an eqx.Module, even if top-level pytree registration
    is broken (e.g. cloudpickle-restored user-defined modules).

    Falls back to dataclass field traversal when jax.tree_util returns the
    module itself as a single leaf.
    """
    from dataclasses import fields as dc_fields

    keyed, _ = jax.tree_util.tree_flatten_with_path(model)
    if keyed:
        return keyed

    # JAX saw the model as opaque — walk dataclass fields manually.
    result: list[tuple[tuple, Any]] = []

    def _walk(obj: Any, path: tuple) -> None:
        # If JAX can flatten this object into >0 leaves, use that.
        sub_keyed, _ = jax.tree_util.tree_flatten_with_path(obj)
        if sub_keyed:
            for sub_path, leaf in sub_keyed:
                result.append((path + tuple(sub_path), leaf))
            return

        # If it's a dataclass / eqx.Module, recurse through fields.
        if hasattr(obj, "__dataclass_fields__"):
            for f in dc_fields(obj):
                val = getattr(obj, f.name, None)
                key = jax.tree_util.GetAttrKey(f.name)
                if isinstance(val, list):
                    for i, item in enumerate(val):
                        _walk(item, path + (key, jax.tree_util.SequenceKey(i)))
                elif isinstance(val, dict):
                    for k, v in val.items():
                        _walk(v, path + (key, jax.tree_util.DictKey(k)))
                else:
                    _walk(val, path + (key,))

    _walk(model, ())
    return result


def iter_module_paths(model: Any) -> list[tuple[str, Any]]:
    """Return nested module paths for all submodules below `model`.

    Paths are expressed relative to the model root and use the same path segment
    conventions as `build_model_tree`, e.g. `blocks/0/linear1`.
    """
    from dataclasses import fields as dc_fields

    results: list[tuple[str, Any]] = []

    def _walk_value(value: Any, path_parts: list[str]) -> None:
        if hasattr(value, "__dataclass_fields__"):
            if path_parts:
                results.append(("/".join(path_parts), value))
            for field in dc_fields(value):
                child = getattr(value, field.name, None)
                _walk_value(child, path_parts + [field.name])
            return

        if isinstance(value, list):
            for idx, item in enumerate(value):
                _walk_value(item, path_parts + [str(idx)])
            return

        if isinstance(value, dict):
            for key, item in value.items():
                _walk_value(item, path_parts + [str(key)])

    _walk_value(model, [])
    return results


def build_model_tree(model: Any) -> dict[str, Any]:
    """Return hierarchical metadata for all leaves in a model pytree."""
    root = TreeNode(
        name=model.__class__.__name__, kind="root", type_name=type(model).__name__
    )

    keyed_leaves = _flatten_with_path(model)
    for path, leaf in keyed_leaves:
        cursor = root
        for item in path:
            key = _path_item_to_string(item)
            if key not in cursor.children:
                cursor.children[key] = TreeNode(name=key, kind="node")
            cursor = cursor.children[key]

        param_count, size_bytes, shape, dtype = _leaf_metrics(leaf)
        cursor.kind = "leaf"
        cursor.type_name = type(leaf).__name__
        cursor.shape = shape
        cursor.dtype = dtype
        cursor.param_count = param_count
        cursor.bytes = size_bytes

    def aggregate(node: TreeNode) -> tuple[int, int]:
        if not node.children:
            return node.param_count, node.bytes
        total_count = node.param_count
        total_bytes = node.bytes
        for child in node.children.values():
            c_count, c_bytes = aggregate(child)
            total_count += c_count
            total_bytes += c_bytes
        node.param_count = total_count
        node.bytes = total_bytes
        return total_count, total_bytes

    aggregate(root)

    return {
        "tree": root.as_json(),
        "summary": {
            "total_parameters": root.param_count,
            "total_bytes": root.bytes,
            "total_megabytes": round(root.bytes / (1024 * 1024), 3),
            "leaf_count": len(keyed_leaves),
        },
    }


def _node_id_from_parts(parts: list[str]) -> str:
    return "op:" + "/".join(parts)


def _edge_label(name: str, shape: list[int] | None, dtype: str | None) -> str:
    shape_txt = f"[{ 'x'.join(str(v) for v in shape) }]" if shape else ""
    dtype_txt = f" {dtype}" if dtype else ""
    return f"{name} {shape_txt}{dtype_txt}".strip()


def _natural_key(text: str) -> tuple:
    parts: list[Any] = []
    buf = ""
    is_digit = None

    for ch in text:
        cur_digit = ch.isdigit()
        if is_digit is None:
            is_digit = cur_digit
        if cur_digit != is_digit:
            parts.append(int(buf) if is_digit else buf)
            buf = ch
            is_digit = cur_digit
        else:
            buf += ch

    if buf:
        parts.append(int(buf) if is_digit else buf)
    return tuple(parts)


def _collect_leaf_groups(
    node: dict[str, Any], path: list[str], out: list[dict[str, Any]]
) -> None:
    children = node.get("children", [])
    if not children:
        return

    leaf_children = [
        child
        for child in children
        if child.get("kind") == "leaf" and child.get("shape") is not None
    ]
    if leaf_children:
        out.append(
            {
                "path": path,
                "name": path[-1] if path else node.get("name", "op"),
                "leaves": leaf_children,
            }
        )

    for child in children:
        if child.get("kind") != "leaf":
            _collect_leaf_groups(child, path + [child.get("name", "node")], out)


def build_operation_graph(
    model: Any, parameter_tree: Any | None = None
) -> dict[str, Any]:
    """Build an architecture-agnostic operation graph from parameter groups.

    The graph is intentionally generic: each parameter group contributes one
    operation node, with all leaf tensors connected as parameter edges.
    """
    param_source = model if parameter_tree is None else parameter_tree
    tree_payload = build_model_tree(param_source)
    root = tree_payload["tree"]

    # Build a path-string → actual leaf array map for heatmap data.
    # This must come from the same parameter tree used to build the graph,
    # otherwise frameworks like Flax end up with graph nodes but no heatmap values.
    keyed_leaves = _flatten_with_path(param_source)
    _leaf_arrays: dict[str, Any] = {}
    for path, leaf in keyed_leaves:
        parts = [_path_item_to_string(p) for p in path]
        _leaf_arrays["/".join(parts)] = leaf

    def _heatmap_for(group_path: list[str], leaf_name: str) -> list[list[float]] | None:
        """Look up the actual array for a leaf and return heatmap data."""
        # The tree path is: root_name / ... / group_parts / leaf_name
        # keyed_leaves paths start after root, so try joining group_path[1:] + leaf_name
        key = "/".join(group_path[1:] + [leaf_name])
        arr = _leaf_arrays.get(key)
        if arr is not None:
            return _downsample_for_heatmap(arr)
        return None

    def _flat_values_for(group_path: list[str], leaf_name: str) -> list[float] | None:
        key = "/".join(group_path[1:] + [leaf_name])
        arr = _leaf_arrays.get(key)
        if arr is not None:
            return flat_values_for_client(arr)
        return None

    op_nodes: list[dict[str, Any]] = []
    op_edges: list[dict[str, Any]] = []

    groups: list[dict[str, Any]] = []
    _collect_leaf_groups(root, [root.get("name", "Model")], groups)

    groups.sort(key=lambda grp: _natural_key("/".join(grp["path"])))

    node_count = 0
    depth_count = 0
    prev_id: str | None = None

    # --- Capture detection -------------------------------------------------
    # Find all Capture wrappers so their op nodes can be annotated.
    # This only applies when `model` is an Equinox module tree.
    _capture_at_path: dict[str, str] = {}
    if hasattr(model, "__dataclass_fields__"):
        for _iter_path, _mod in iter_module_paths(model):
            if isinstance(_mod, _Capture):
                _capture_at_path[_iter_path] = _mod.name

    def _strip_capture_module_segment(path: str) -> str:
        """Hide redundant Capture internals, e.g. `embedding/module/...` -> `embedding/...`."""
        for cap_path in _capture_at_path.keys():
            inner_prefix = cap_path + "/module"
            if path == inner_prefix:
                return cap_path
            if path.startswith(inner_prefix + "/"):
                return cap_path + path[len(inner_prefix) :]
        return path

    def _capture_for_group_id(gid: str) -> tuple[str, str] | None:
        """Return (capture_path, capture_name) if gid falls inside a Capture."""
        for cap_path, cap_name in _capture_at_path.items():
            if gid == cap_path or gid.startswith(cap_path + "/"):
                return cap_path, cap_name
        return None

    def add_op_node(
        label: str,
        path: str,
        *,
        group: str | None = None,
        capture_path: str | None = None,
        capture_name: str | None = None,
    ) -> str:
        nonlocal node_count, depth_count
        node_id = f"op:{node_count}"
        op_nodes.append(
            {
                "id": node_id,
                "label": label,
                "depth": depth_count,
                "type": "operation",
                "path": path,
                "group": group,
                "capture_path": capture_path,
                "capture_name": capture_name,
            }
        )
        node_count += 1
        depth_count += 1
        return node_id

    for idx, group in enumerate(groups):
        raw_group_path = "/".join(group["path"])
        # Use the path (minus model root) as group id for hierarchical collapsing
        raw_group_id = (
            "/".join(group["path"][1:]) if len(group["path"]) > 1 else raw_group_path
        )
        group_id = _strip_capture_module_segment(raw_group_id)
        group_path = _strip_capture_module_segment(raw_group_path)
        leaves = [leaf for leaf in group["leaves"] if leaf.get("shape") is not None]
        leaves.sort(key=lambda leaf: _natural_key(str(leaf.get("name", "tensor"))))

        _cap_info = _capture_for_group_id(group_id)
        _cap_path = _cap_info[0] if _cap_info else None
        _cap_name = _cap_info[1] if _cap_info else None

        node_label = group.get("name") or f"group_{idx}"
        op_id = add_op_node(
            node_label,
            f"{group_path}/op",
            group=group_id,
            capture_path=_cap_path,
            capture_name=_cap_name,
        )
        if prev_id is None:
            op_edges.append(
                {
                    "source": None,
                    "target": op_id,
                    "label": "input",
                    "kind": "flow",
                    "shape": None,
                    "dtype": None,
                }
            )
        else:
            op_edges.append(
                {
                    "source": prev_id,
                    "target": op_id,
                    "label": "x",
                    "kind": "flow",
                    "shape": None,
                    "dtype": None,
                }
            )

        for leaf in leaves:
            leaf_name = str(leaf.get("name", "tensor"))
            leaf_shape = leaf.get("shape")
            leaf_dtype = leaf.get("dtype")
            op_edges.append(
                {
                    "source": None,
                    "target": op_id,
                    "label": _edge_label(leaf_name, leaf_shape, leaf_dtype),
                    "kind": "parameter",
                    "shape": leaf_shape,
                    "dtype": leaf_dtype,
                    "values": _heatmap_for(group["path"], leaf_name),
                    "flat_values": _flat_values_for(group["path"], leaf_name),
                }
            )

        prev_id = op_id

    return {
        "summary": tree_payload["summary"],
        "graph": {
            "nodes": op_nodes,
            "edges": op_edges,
        },
    }
